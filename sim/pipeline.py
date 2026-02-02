import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import ensure_tick_events_dir
from .llm import LLMClient
from .models import Edge, Node, Patch, new_id, now_ts
from .state import SimulationState


class Simulator:
    def __init__(self, state: SimulationState, engine_cfg: Dict[str, Any], llm: LLMClient):
        self.state = state
        self.engine_cfg = engine_cfg
        self.llm = llm

    def step(self) -> SimulationState:
        current = self.state.copy()
        current.t += 1
        self.state = current
        obs_packets = {node_id: self._build_observation(node) for node_id, node in current.nodes.items()}
        event_log: List[str] = []

        pair_opps = self._sample_pair_opportunities(current)
        prod_opps = self._sample_production_opportunities(current)

        patches: List[Patch] = []
        for u, v in pair_opps:
            writer_out = self.llm.writer_pair(obs_packets[u], obs_packets[v])
            context = {
                "nodes": [current.nodes[u].to_dict(), current.nodes[v].to_dict()],
                "allowed_node_ids": [u, v],
                "reputations": {u: current.nodes[u].state.get("reputation", 0.5), v: current.nodes[v].state.get("reputation", 0.5)},
                "transfer_value": round(random.uniform(1, 6), 2),
                "t": current.t,
            }
            patch = self.llm.compiler(writer_out["outcome_summary"], context)
            patch = self._normalize_edge_ops(patch, (u, v), current)
            patches.append(patch)
            event_log.append(f"Pair {u} & {v}: {writer_out['outcome_summary']}")

        for node_id in prod_opps:
            writer_out = self.llm.writer_production(obs_packets[node_id])
            context = {
                "nodes": [current.nodes[node_id].to_dict()],
                "allowed_node_ids": [node_id],
                "production_value": round(random.uniform(1, 8), 2),
                "t": current.t,
            }
            patch = self.llm.compiler(writer_out["outcome_summary"], context)
            patches.append(patch)
            event_log.append(f"Production {node_id}: {writer_out['outcome_summary']}")

        # Apply patches sequentially with validation.
        for patch in patches:
            current.apply_patch(patch, self.engine_cfg)

        self._apply_environment_processes(current, event_log)
        self._memory_maintenance(current)
        self._write_tick_event(current, event_log)

        return current

    def _build_observation(self, node: Node) -> Dict[str, Any]:
        env = self.state.environment
        top_k = self.engine_cfg.get("top_k_edges", 3)
        top_m = self.engine_cfg.get("top_m_memories", 3)
        info_fidelity = env.get("information_fidelity", 1.0)
        bandwidth = node.traits.get("network_bandwidth", top_k)
        capacity = node.traits.get("memory_capacity", top_m)

        edges = self._incident_edges(node.id)
        edges_sorted = sorted(
            edges,
            key=lambda e: e.characteristics.get("strength", 0) * e.characteristics.get("activation_rate", 0),
            reverse=True,
        )
        edges_sampled = edges_sorted[: min(top_k, bandwidth)]
        neighbors = []
        for edge in edges_sampled:
            neighbor_id = edge.target if edge.source == node.id else edge.source
            neighbors.append(
                {
                    "id": neighbor_id,
                    "reputation": self.state.nodes[neighbor_id].state.get("reputation", 0.5),
                    "strength": edge.characteristics.get("strength", 0),
                }
            )

        mems_raw = node.state.get("memories", [])
        mems = [m for m in mems_raw if isinstance(m, dict)]
        mems = sorted(mems, key=lambda m: m.get("created_at", 0), reverse=True)
        mems_sampled = mems[: min(top_m, capacity)]

        def noisy(value: Any) -> Any:
            if random.random() > info_fidelity:
                return None
            return value

        return {
            "t": self.state.t,
            "id": node.id,
            "name": node.name,
            "self_summary": {
                "reputation": noisy(node.state.get("reputation", 0.5)),
                "assets_count": noisy(len(node.state.get("assets", []))),
                "debts_count": noisy(len(node.state.get("debts", []))),
                "claims_count": noisy(len(node.state.get("claims", []))),
                "job": noisy(node.state.get("job")),
            },
            "top_edges": [edge.to_dict() for edge in edges_sampled],
            "neighbor_summaries": neighbors,
            "selected_memories": mems_sampled,
            "env_snippet": {"information_fidelity": info_fidelity, "transaction_friction": env.get("transaction_friction", 0)},
        }

    def _incident_edges(self, node_id: str) -> List[Edge]:
        return [e for e in self.state.edges.values() if e.source == node_id or e.target == node_id]

    def _sample_pair_opportunities(self, state: SimulationState) -> List[Tuple[str, str]]:
        env = state.environment
        rate = env.get("interaction_opportunity_rate", 0.5)
        rho = env.get("novel_pair_chance", 0.25)
        max_pairs = self.engine_cfg.get("max_pair_opps_per_step", 10)
        threshold = self.engine_cfg.get("strength_activation_threshold", 0)
        candidates: List[Tuple[str, str]] = []

        node_ids = list(state.nodes.keys())
        for node_id in node_ids:
            m = np.random.poisson(rate)
            for _ in range(m):
                if random.random() < rho or not self._incident_edges(node_id):
                    other = random.choice([nid for nid in node_ids if nid != node_id])
                    pair = tuple(sorted((node_id, other)))
                    candidates.append(pair)
                else:
                    edges = [
                        e for e in self._incident_edges(node_id)
                        if e.characteristics.get("strength", 0) * e.characteristics.get("activation_rate", 0) >= threshold
                    ]
                    if not edges:
                        continue
                    weights = [
                        e.characteristics.get("strength", 0) * e.characteristics.get("activation_rate", 0) + 0.01 for e in edges
                    ]
                    edge = random.choices(edges, weights=weights, k=1)[0]
                    other = edge.target if edge.source == node_id else edge.source
                    pair = tuple(sorted((node_id, other)))
                    candidates.append(pair)

        deduped = list({pair for pair in candidates})
        random.shuffle(deduped)
        return deduped[:max_pairs]

    def _sample_production_opportunities(self, state: SimulationState) -> List[str]:
        env = state.environment
        rate = env.get("production_rate", 0.2)
        max_prod = self.engine_cfg.get("max_prod_ops_per_step", 5)
        picks = []
        for node_id, node in state.nodes.items():
            if random.random() < rate:
                picks.append(node_id)
        random.shuffle(picks)
        return picks[:max_prod]

    def _apply_environment_processes(self, state: SimulationState, event_log: List[str]) -> None:
        env = state.environment
        friction = env.get("transaction_friction", 0)
        max_prod_value = self.engine_cfg.get("max_prod_value_per_node", 50)

        # Apply transaction friction to newly added assets in this tick (approx: reduce last asset entries)
        for node in state.nodes.values():
            assets = node.state.get("assets", [])
            if not isinstance(assets, list):
                assets = []
                node.state["assets"] = assets
            last_asset = next((a for a in reversed(assets) if isinstance(a, dict)), None)
            if last_asset and last_asset.get("status") == "active":
                value = last_asset.get("value", 0) or 0
                last_asset["value"] = round(max(value - value * friction, 0), 2)
                last_asset["last_updated"] = now_ts()
                if value > max_prod_value:
                    last_asset["value"] = max_prod_value

        # Shocks
        shocks = env.get("shocks", {})
        shock_freq = shocks.get("frequency", 0)
        for node in state.nodes.values():
            if random.random() < shock_freq:
                magnitude = random.gauss(shocks.get("magnitude_mean", 0), shocks.get("magnitude_std", 1))
                node.state["assets"].append(
                    {
                        "id": new_id("shock"),
                        "created_at": now_ts(),
                        "last_updated": now_ts(),
                        "value": magnitude,
                        "status": "active",
                        "description": shocks.get("story", "Shock event"),
                    }
                )
                node.state["memories"].append(
                    {"id": new_id("mem"), "created_at": now_ts(), "description": shocks.get("story", "Shock event")}
                )
                event_log.append(f"Shock to {node.id}: {shocks.get('story', '')} (delta {round(magnitude,2)})")

        # Inflow
        inflow = env.get("inflow", {})
        if random.random() < inflow.get("frequency", 0):
            targets = list(state.nodes.values())
            random.shuffle(targets)
            amount = inflow.get("amount", 0)
            for node in targets[: max(1, len(targets) // 2)]:
                node.state["assets"].append(
                    {
                        "id": new_id("inflow"),
                        "created_at": now_ts(),
                        "last_updated": now_ts(),
                        "value": amount,
                        "status": "active",
                        "description": inflow.get("story", "Inflow"),
                    }
                )
                node.state["memories"].append(
                    {"id": new_id("mem"), "created_at": now_ts(), "description": inflow.get("story", "Inflow received")}
                )
                event_log.append(f"Inflow to {node.id}: +{amount} ({inflow.get('story', '')})")

        # Aging and simple succession stub
        for node in state.nodes.values():
            if node.kind == "agent":
                node.state["age"] = node.state.get("age", 0) + 1
                if node.state["age"] > node.traits.get("lifespan", math.inf):
                    node.state["succession_intent"] = "Succession pending"
                    event_log.append(f"Succession pending for {node.id} (age {node.state['age']})")

    def _memory_maintenance(self, state: SimulationState) -> None:
        for node in state.nodes.values():
            cap = node.traits.get("memory_capacity", 10)
            mems = node.state.get("memories", [])
            if len(mems) > cap:
                node.state["memories"] = mems[-cap:]

    def _normalize_edge_ops(self, patch: Patch, endpoints: tuple[str, str], state: SimulationState) -> Patch:
        """
        If compiler tries to update a non-existent edge between endpoints, convert to create_edge with defaults.
        """
        u, v = endpoints
        new_ops: List[Dict[str, Any]] = []
        for op in patch.ops:
            if op.get("op") == "create_node":
                # Skip LLM-created nodes; we only mutate existing nodes.
                continue
            if op.get("op") == "update_edge":
                edge_id = op.get("id")
                if edge_id and edge_id not in state.edges:
                    ts = now_ts()
                    edge_data = op.get("edge", {})
                    canonical_id = edge_id or f"edge-{min(u,v)}-{max(u,v)}"
                    default_edge = {
                        "id": canonical_id,
                        "source": min(edge_data.get("source", u), edge_data.get("target", v)),
                        "target": max(edge_data.get("source", u), edge_data.get("target", v)),
                        "metadata": edge_data.get("metadata", {"id": canonical_id, "created_at": ts, "last_updated": ts}),
                        "characteristics": edge_data.get(
                            "characteristics",
                            {
                                "strength": 0.2,
                                "consent": True,
                                "visibility": 0.5,
                                "activation_rate": 0.2,
                                "enforceability": 0.2,
                            },
                        ),
                        "lifecycle": edge_data.get("lifecycle", {"strength_decay_rate": None, "expiration": None}),
                        "specifics": edge_data.get(
                            "specifics",
                            {
                                "relationship_description": "New interaction edge",
                                "expectations": "Light coordination",
                                "history": "Formed during interaction",
                            },
                        ),
                    }
                    new_ops.append({"op": "create_edge", "edge": default_edge})
                    continue
            if op.get("op") == "create_edge":
                edge_data = op.get("edge", {})
                if "source" not in edge_data:
                    edge_data["source"] = endpoints[0]
                if "target" not in edge_data:
                    edge_data["target"] = endpoints[1]
                # Canonicalize id and direction to enforce one edge per unordered pair.
                src, tgt = sorted([edge_data["source"], edge_data["target"]])
                edge_data["source"], edge_data["target"] = src, tgt
                if "id" not in edge_data:
                    edge_data["id"] = f"edge-{src}-{tgt}"
                else:
                    edge_data["id"] = f"edge-{src}-{tgt}"
                if "metadata" not in edge_data:
                    ts = now_ts()
                    edge_data["metadata"] = {"id": edge_data["id"], "created_at": ts, "last_updated": ts}
                op["edge"] = edge_data
            new_ops.append(op)
        return Patch(ops=new_ops, notes=patch.notes)

    def _write_tick_event(self, state: SimulationState, event_log: List[str]) -> None:
        summary = self.llm.summarize_tick(event_log, state.t)
        payload = {"t": state.t, "summary": summary, "events": event_log}
        dir_path = ensure_tick_events_dir()
        path = Path(dir_path) / f"t{state.t}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
