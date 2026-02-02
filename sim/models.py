import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, TypedDict


def now_ts() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


NodeKind = Literal["agent", "institution"]


@dataclass
class Node:
    id: str
    kind: NodeKind
    name: str
    traits: Dict[str, Any]
    state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Node":
        return Node(
            id=data["id"],
            kind=data["kind"],
            name=data["name"],
            traits=data.get("traits", {}),
            state=data.get("state", {}),
        )


@dataclass
class Edge:
    id: str
    source: str
    target: str
    metadata: Dict[str, Any]
    characteristics: Dict[str, Any]
    lifecycle: Dict[str, Any]
    specifics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Edge":
        return Edge(
            id=data["id"],
            source=data["source"],
            target=data["target"],
            metadata=data.get("metadata", {}),
            characteristics=data.get("characteristics", {}),
            lifecycle=data.get("lifecycle", {}),
            specifics=data.get("specifics", {}),
        )


class UpdateChange(TypedDict, total=False):
    type: Literal["set", "append", "update_item", "remove_item"]
    path: str
    value: Any
    item: Any
    id: str
    fields: Dict[str, Any]


class PatchOp(TypedDict, total=False):
    op: Literal["create_node", "update_node", "create_edge", "update_edge", "delete_edge"]
    node: Dict[str, Any]
    edge: Dict[str, Any]
    id: str
    changes: List[UpdateChange]


@dataclass
class Patch:
    ops: List[PatchOp] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"ops": self.ops, "notes": self.notes}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Patch":
        return Patch(ops=data.get("ops", []), notes=data.get("notes"))
