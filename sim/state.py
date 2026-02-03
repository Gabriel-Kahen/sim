import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .models import Edge, Node, Patch, PatchOp, UpdateChange, now_ts


RANGE_FIELDS = {
    "risk_tolerance",
    "patience",
    "norm_sensitivity",
    "trust_disposition",
    "reputation_sensitivity",
    "enforcement_aversion",
    "adaptability",
    "rigidity",
    "reputation",
}


def _split_path(path: str) -> List[str]:
    return [p for p in path.split(".") if p]


def _edge_key(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((source, target)))


def _resolve_container(data: Dict[str, Any], path: str) -> Tuple[Any | None, str | None]:
    parts = _split_path(path)
    if not parts:
        return None, None
    cur = data
    for p in parts[:-1]:
        if isinstance(cur, list):
            try:
                idx = int(p)
            except Exception:
                return None, None
            if idx >= len(cur):
                return None, None
            cur = cur[idx]
        else:
            if p not in cur or not isinstance(cur[p], dict | list):
                cur[p] = {}
            cur = cur[p]
    return cur, parts[-1]


def _set_path(data: Dict[str, Any], path: str, value: Any) -> None:
    container, key = _resolve_container(data, path)
    if container is None or key is None:
        return
    if isinstance(container, list):
        # Ignore invalid set into list
        return
    container[key] = value


def _append_path(data: Dict[str, Any], path: str, item: Any) -> None:
    container, key = _resolve_container(data, path)
    if container is None or key is None:
        return
    if isinstance(container, list):
        # Path resolution landed in a list; skip to avoid type errors.
        return
    if key not in container or not isinstance(container[key], list):
        container[key] = []
    container[key].append(item)


def _update_item(data: Dict[str, Any], path: str, item_id: str, fields: Dict[str, Any]) -> None:
    container, key = _resolve_container(data, path)
    if container is None or key is None:
        return
    if key not in container or not isinstance(container[key], list):
        container[key] = []
    for idx, obj in enumerate(container[key]):
        if isinstance(obj, dict) and obj.get("id") == item_id:
            container[key][idx] = {**obj, **fields}
            return
    container[key].append({"id": item_id, **fields})


def _remove_item(data: Dict[str, Any], path: str, item_id: str) -> None:
    container, key = _resolve_container(data, path)
    if container is None or key is None:
        return
    if key not in container or not isinstance(container[key], list):
        return
    if item_id is None:
        return
    container[key] = [obj for obj in container[key] if not (isinstance(obj, dict) and obj.get("id") == item_id)]


def validate_range_fields(node: Node) -> None:
    traits = node.traits
    state = node.state
    for field in RANGE_FIELDS:
        for block in (traits, state):
            if field in block:
                val = block[field]
                try:
                    num = float(val)
                except Exception:
                    num = 0.5
                if num < 0:
                    num = 0.0
                if num > 1:
                    num = 1.0
                block[field] = num


@dataclass
class SimulationState:
    t: int
    environment: Dict[str, Any]
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[str, Edge] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def copy(self) -> "SimulationState":
        return SimulationState(
            t=self.t,
            environment=copy.deepcopy(self.environment),
            nodes={k: Node.from_dict(v.to_dict()) for k, v in self.nodes.items()},
            edges={k: Edge.from_dict(v.to_dict()) for k, v in self.edges.items()},
            events=copy.deepcopy(self.events),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "environment": self.environment,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "events": self.events,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SimulationState":
        return SimulationState(
            t=data["t"],
            environment=data["environment"],
            nodes={k: Node.from_dict(v) for k, v in data.get("nodes", {}).items()},
            edges={k: Edge.from_dict(v) for k, v in data.get("edges", {}).items()},
            events=data.get("events", []),
        )

    def apply_patch(self, patch: Patch, engine_cfg: Dict[str, Any]) -> None:
        """
        Apply patch ops with minimal semantics.
        """
        for op in patch.ops:
            if "op" not in op:
                raise ValueError(f"patch op missing 'op': {op}")
            match op["op"]:
                case "create_node":
                    node_data = op.get("node")
                    if (
                        not node_data
                        or "id" not in node_data
                        or "kind" not in node_data
                        or not isinstance(node_data["id"], str)
                        or not node_data["id"].startswith(("agent-", "inst-"))
                    ):
                        continue
                    try:
                        node = Node.from_dict(node_data)
                        validate_range_fields(node)
                    except Exception:
                        continue
                    self.nodes[node.id] = node
                case "update_node":
                    node_id = op["id"]
                    if node_id not in self.nodes:
                        continue
                    node = self.nodes[node_id]
                    self._apply_node_changes(node, op.get("changes", []))
                    validate_range_fields(node)
                    if isinstance(node.state, dict):
                        node.state["last_updated"] = now_ts()
                case "create_edge":
                    edge_data = op.get("edge")
                    if not edge_data or "id" not in edge_data or "source" not in edge_data or "target" not in edge_data:
                        continue
                    if edge_data["source"] not in self.nodes or edge_data["target"] not in self.nodes:
                        continue
                    # Enforce single edge per unordered pair.
                    key = _edge_key(edge_data["source"], edge_data["target"])
                    existing_id = None
                    for e_id, e in self.edges.items():
                        if _edge_key(e.source, e.target) == key:
                            existing_id = e_id
                            break
                    if existing_id:
                        updated = {**self.edges[existing_id].to_dict(), **edge_data}
                        self.edges[existing_id] = Edge.from_dict(updated)
                        continue
                    edge = Edge.from_dict(edge_data)
                    self.edges[edge.id] = edge
                case "update_edge":
                    edge_id = op["id"]
                    edge_payload = op.get("edge", {})
                    if edge_id not in self.edges:
                        src = edge_payload.get("source")
                        tgt = edge_payload.get("target")
                        if src and tgt:
                            key = _edge_key(src, tgt)
                            for e_id, e in self.edges.items():
                                if _edge_key(e.source, e.target) == key:
                                    edge_id = e_id
                                    break
                    if edge_id not in self.edges:
                        continue
                    edge = self.edges[edge_id]
                    updated = {**edge.to_dict(), **edge_payload}
                    self.edges[edge_id] = Edge.from_dict(updated)
                case "delete_edge":
                    edge_id = op["id"]
                    if edge_id in self.edges:
                        self.edges.pop(edge_id)
                case _:
                    raise ValueError(f"unsupported op {op}")
        self.events.append({"t": self.t, "notes": patch.notes})

    def _apply_node_changes(self, node: Node, changes: List[UpdateChange]) -> None:
        for change in changes:
            ctype = change["type"]
            path = change.get("path")
            if not path:
                continue
            if ctype == "set":
                _set_path(node.__dict__, path, change.get("value"))
            elif ctype == "append":
                item = change.get("item", change.get("value"))
                _append_path(node.__dict__, path, item)
            elif ctype == "update_item":
                item_id = change.get("id")
                if item_id is None:
                    continue
                _update_item(node.__dict__, path, item_id, change.get("fields", {}))
            elif ctype == "remove_item":
                item_id = change.get("id")
                if item_id is None:
                    continue
                _remove_item(node.__dict__, path, item_id)
            elif ctype == "increment":
                # Increment a numeric field; ignore if non-numeric.
                container, key = _resolve_container(node.__dict__, path)
                if container is None or key is None:
                    continue
                try:
                    current_val = float(container.get(key, 0))
                except Exception:
                    continue
                delta = change.get("value", 0) or 0
                try:
                    delta = float(delta)
                except Exception:
                    delta = 0
                container[key] = current_val + delta
            else:
                # Skip unknown change types to keep simulation running.
                continue


def save_state(path: Path, state: SimulationState) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)


def load_state(path: Path) -> SimulationState:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SimulationState.from_dict(data)
