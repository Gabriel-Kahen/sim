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
            # Institution formation chance
            self._maybe_form_institution(current, (u, v), writer_out["outcome_summary"], event_log)

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
                "assets_count": noisy(len(node.state.get("assets", []) if isinstance(node.state.get("assets"), list) else [])),
                "debts_count": noisy(len(node.state.get("debts", []) if isinstance(node.state.get("debts"), list) else [])),
                "claims_count": noisy(len(node.state.get("claims", []) if isinstance(node.state.get("claims"), list) else [])),
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
        job_change_rate = self.engine_cfg.get("job_change_rate", 0)
        job_options = self.engine_cfg.get("job_options", [])
        succession_rate = self.engine_cfg.get("succession_update_rate", 0)
        succession_templates = self.engine_cfg.get("succession_templates", [])
        birth_rate = self.engine_cfg.get("birth_rate", 0)
        birth_noise = self.engine_cfg.get("birth_trait_noise", 0.1)
        max_children = self.engine_cfg.get("max_children_per_parent", 2)
        death_jitter = self.engine_cfg.get("death_jitter", 5)
        child_rep_range = self.engine_cfg.get("initial_child_reputation_range", [0.4, 0.7])
        defection_rate = self.engine_cfg.get("defection_rate", 0)

        nodes_list = list(state.nodes.values())

        # Apply transaction friction to newly added assets in this tick (approx: reduce last asset entries)
        for node in nodes_list:
            assets = node.state.get("assets", [])
            if not isinstance(assets, list):
                assets = []
                node.state["assets"] = assets
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
        for node in nodes_list:
            if random.random() < shock_freq:
                magnitude = random.gauss(shocks.get("magnitude_mean", 0), shocks.get("magnitude_std", 1))
                assets_list = node.state.get("assets")
                if not isinstance(assets_list, list):
                    assets_list = []
                    node.state["assets"] = assets_list
                assets_list.append(
                    {
                        "id": new_id("shock"),
                        "created_at": now_ts(),
                        "last_updated": now_ts(),
                        "value": magnitude,
                        "status": "active",
                        "description": shocks.get("story", "Shock event"),
                    }
                )
                mems_list = node.state.get("memories")
                if not isinstance(mems_list, list):
                    mems_list = []
                    node.state["memories"] = mems_list
                mems_list.append(
                    {"id": new_id("mem"), "created_at": now_ts(), "description": shocks.get("story", "Shock event")}
                )
                event_log.append(f"Shock to {node.id}: {shocks.get('story', '')} (delta {round(magnitude,2)})")

        # Inflow
        inflow = env.get("inflow", {})
        if random.random() < inflow.get("frequency", 0):
            targets = nodes_list.copy()
            random.shuffle(targets)
            amount = inflow.get("amount", 0)
            for node in targets[: max(1, len(targets) // 2)]:
                assets_list = node.state.get("assets")
                if not isinstance(assets_list, list):
                    assets_list = []
                    node.state["assets"] = assets_list
                assets_list.append(
                    {
                        "id": new_id("inflow"),
                        "created_at": now_ts(),
                        "last_updated": now_ts(),
                        "value": amount,
                        "status": "active",
                        "description": inflow.get("story", "Inflow"),
                    }
                )
                mems_list = node.state.get("memories")
                if not isinstance(mems_list, list):
                    mems_list = []
                    node.state["memories"] = mems_list
                mems_list.append(
                    {"id": new_id("mem"), "created_at": now_ts(), "description": inflow.get("story", "Inflow received")}
                )
                event_log.append(f"Inflow to {node.id}: +{amount} ({inflow.get('story', '')})")

        # Aging and simple succession stub
        for node in nodes_list:
            if node.kind == "agent":
                node.state["age"] = node.state.get("age", 0) + 1
                age = node.state["age"]
                lifespan_val = math.inf
                if isinstance(node.traits, dict):
                    lifespan_val = node.traits.get("lifespan", math.inf)
                # Births
                children = node.state.setdefault("children", [])
                if (
                    birth_rate > 0
                    and node.state.get("status") != "deceased"
                    and age < lifespan_val - 20
                    and len(children) < max_children
                    and random.random() < birth_rate
                ):
                    child_id = self._create_child(state, node, birth_noise, child_rep_range, event_log)
                    if child_id:
                        children.append(child_id)
                        event_log.append(f"{node.id} had child {child_id}")
                # Defection (form new family)
                if defection_rate > 0 and node.state.get("status") != "deceased" and random.random() < defection_rate:
                    new_family_id = self._defect_from_family(state, node, event_log)
                    if new_family_id:
                        event_log.append(f"{node.id} defected to {new_family_id}")
                # Deaths
                if age > lifespan_val + random.uniform(0, death_jitter):
                    node.state["status"] = "deceased"
                    self._handle_death(state, node, event_log)
                    event_log.append(f"{node.id} died at age {age}")
                # Drift job and succession intent
                if job_options and random.random() < job_change_rate:
                    new_job = random.choice(job_options)
                    node.state["job"] = new_job
                    node.state.setdefault("memories", []).append(
                        {"id": new_id("mem"), "created_at": now_ts(), "description": f"Shifted job to {new_job}"}
                    )
                    event_log.append(f"{node.id} job set to {new_job}")
                if succession_templates and random.random() < succession_rate:
                    new_intent = random.choice(succession_templates)
                    node.state["succession_intent"] = new_intent
                    node.state.setdefault("memories", []).append(
                        {"id": new_id("mem"), "created_at": now_ts(), "description": f"Updated succession intent: {new_intent}"}
                    )
                    event_log.append(f"{node.id} succession intent updated")

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
                    src_val = edge_data.get("source", u)
                    tgt_val = edge_data.get("target", v)
                    if not isinstance(src_val, str) or not isinstance(tgt_val, str):
                        src_val, tgt_val = u, v
                    canonical_id = edge_id or f"edge-{min(src_val, tgt_val)}-{max(src_val, tgt_val)}"
                    default_edge = {
                        "id": canonical_id,
                        "source": min(src_val, tgt_val),
                        "target": max(src_val, tgt_val),
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
                src_val = edge_data.get("source")
                tgt_val = edge_data.get("target")
                if not isinstance(src_val, str) or not isinstance(tgt_val, str):
                    src_val, tgt_val = endpoints
                src, tgt = sorted([src_val, tgt_val])
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

    def _maybe_form_institution(
        self,
        state: SimulationState,
        pair: tuple[str, str],
        outcome_summary: str,
        event_log: list[str],
    ) -> None:
        rate = self.engine_cfg.get("institution_formation_rate", 0)
        if rate <= 0 or random.random() >= rate:
            return
        u, v = pair
        nodes = state.nodes
        if u not in nodes or v not in nodes:
            return
        # Avoid if already share an institution
        shared = self._shared_institution(nodes[u], nodes[v], state)
        if shared:
            return
        participants = [
            {"id": u, "name": nodes[u].name},
            {"id": v, "name": nodes[v].name},
        ]
        spec = self.llm.propose_institution(participants, outcome_summary)
        ts = now_ts()
        inst_id = new_id("inst")
        inst_traits = {
            "created_at": ts,
            "memory_capacity": 20,
            "rigidity": random.uniform(0.3, 0.7),
            "institution_type": spec.get("institution_type", "Organization"),
            "institution_description": spec.get("description", "A new organization"),
        }
        inst_state = {
            "last_updated": ts,
            "reputation": 0.5,
            "capabilities": spec.get("capabilities", "General coordination"),
            "policies": spec.get("policies", "None specified"),
            "governance": spec.get("governance", "Informal"),
            "memories": [],
            "succession_rules": "Members decide",
            "assets": [],
            "debts": [],
            "claims": [],
        }
        inst_node = Node(id=inst_id, kind="institution", name=spec.get("name", f"Institution {inst_id[-4:]}"), traits=inst_traits, state=inst_state)
        state.nodes[inst_id] = inst_node
        # Membership edges
        self._ensure_edge(state, u, inst_id, "Institution membership", strength=0.3, activation=0.3, enforceability=0.2, visibility=0.6)
        self._ensure_edge(state, v, inst_id, "Institution membership", strength=0.3, activation=0.3, enforceability=0.2, visibility=0.6)
        # Memories
        for nid in (u, v):
            nodes[nid].state.setdefault("memories", []).append(
                {"id": new_id("mem"), "created_at": ts, "description": f"Formed institution {inst_node.name}"}
            )
        event_log.append(f"Institution formed: {inst_node.name} by {u},{v}")

    def _shared_institution(self, a: Node, b: Node, state: SimulationState) -> bool:
        insts_a = {e.target if e.source == a.id else e.source for e in self._incident_edges(a.id) if state.nodes.get(e.target if e.source == a.id else e.source, None) and state.nodes.get(e.target if e.source == a.id else e.source).kind == "institution"}
        insts_b = {e.target if e.source == b.id else e.source for e in self._incident_edges(b.id) if state.nodes.get(e.target if e.source == b.id else e.source, None) and state.nodes.get(e.target if e.source == b.id else e.source).kind == "institution"}
        return bool(insts_a & insts_b)

    def _family_institution(self, state: SimulationState, family_id: str) -> str | None:
        for node in state.nodes.values():
            if node.kind == "institution" and family_id in node.name:
                return node.id
        return None

    def _create_child(
        self,
        state: SimulationState,
        parent: Node,
        noise: float,
        rep_range: list[float],
        event_log: list[str],
    ) -> str | None:
        ts = now_ts()
        child_id = new_id("agent")
        def perturb(val: float) -> float:
            return max(0.0, min(1.0, val + random.uniform(-noise, noise)))
        traits = {}
        if isinstance(parent.traits, dict):
            for k, v in parent.traits.items():
                if isinstance(v, (int, float)) and 0 <= v <= 1:
                    traits[k] = perturb(float(v))
                else:
                    traits[k] = v
        traits["created_at"] = ts
        traits["birth_family"] = parent.traits.get("birth_family")
        traits["personality"] = parent.traits.get("personality", "mixed")
        state_block = {
            "age": 0,
            "last_updated": ts,
            "job": "Child",
            "memories": [{"id": new_id("mem"), "created_at": ts, "description": f"Born to {parent.id}"}],
            "assets": [],
            "debts": [],
            "claims": [],
            "reputation": random.uniform(*rep_range) if len(rep_range) == 2 else 0.5,
            "succession_intent": "Undecided",
            "children": [],
        }
        child = Node(id=child_id, kind="agent", name=f"Agent {child_id[-4:]}", traits=traits, state=state_block)
        state.nodes[child_id] = child

        # Connect to family members and institution.
        family_id = traits.get("birth_family")
        siblings = [n for n in state.nodes.values() if n.traits.get("birth_family") == family_id and n.id != child_id and n.id != parent.id]
        targets = [parent] + siblings
        family_inst = self._family_institution(state, family_id) if family_id else None
        for other in targets:
            self._ensure_edge(state, child_id, other.id, "Family tie")
        if family_inst:
            self._ensure_edge(state, child_id, family_inst, "Family institution membership", strength=0.65, enforceability=0.5, visibility=0.7)
        return child_id

    def _ensure_edge(
        self,
        state: SimulationState,
        src: str,
        tgt: str,
        description: str,
        strength: float = 0.7,
        activation: float = 0.5,
        enforceability: float = 0.6,
        visibility: float = 0.8,
    ) -> None:
        ts = now_ts()
        a, b = sorted([src, tgt])
        edge_id = f"edge-{a}-{b}"
        if edge_id in state.edges:
            return
        edge = Edge(
            id=edge_id,
            source=a,
            target=b,
            metadata={"id": edge_id, "created_at": ts, "last_updated": ts},
            characteristics={
                "strength": strength,
                "consent": True,
                "visibility": visibility,
                "activation_rate": activation,
                "enforceability": enforceability,
            },
            lifecycle={"strength_decay_rate": None, "expiration": None},
            specifics={
                "relationship_description": description,
                "expectations": "Support",
                "history": "Family link",
            },
        )
        state.edges[edge_id] = edge

    def _handle_death(self, state: SimulationState, node: Node, event_log: list[str]) -> None:
        node.state["status"] = "deceased"
        # Transfer assets to family institution if possible.
        family_id = node.traits.get("birth_family")
        inst_id = self._family_institution(state, family_id) if family_id else None
        if inst_id and inst_id in state.nodes:
            inst = state.nodes[inst_id]
            inst.state.setdefault("assets", [])
            for asset in node.state.get("assets", []):
                if isinstance(asset, dict):
                    inst.state["assets"].append(asset)
            if node.state.get("assets"):
                event_log.append(f"Assets of {node.id} moved to {inst_id}")
        node.state["assets"] = []

    def _defect_from_family(self, state: SimulationState, node: Node, event_log: list[str]) -> str | None:
        old_family = node.traits.get("birth_family")
        new_family = new_id("fam")
        ts = now_ts()
        # Update trait and state
        node.traits["birth_family"] = new_family
        node.state.setdefault("memories", []).append(
            {"id": new_id("mem"), "created_at": ts, "description": f"Left {old_family} to form {new_family}"}
        )
        # Remove edges to old family members/institution
        to_delete = []
        for e_id, e in state.edges.items():
            if node.id in (e.source, e.target):
                other = e.target if e.source == node.id else e.source
                other_node = state.nodes.get(other)
                if other_node and other_node.traits.get("birth_family") == old_family:
                    to_delete.append(e_id)
                if other_node and other_node.kind == "institution" and old_family in other_node.name:
                    to_delete.append(e_id)
        for e_id in to_delete:
            state.edges.pop(e_id, None)
        # Create new institution
        inst_id = new_id("inst")
        inst_traits = {
            "created_at": ts,
            "memory_capacity": 20,
            "rigidity": 0.45,
            "institution_type": "Family",
            "institution_description": f"Institutional form of family {new_family}",
        }
        inst_state = {
            "last_updated": ts,
            "reputation": 0.6,
            "capabilities": "Family coordination and support",
            "policies": "Mutual aid and resource sharing",
            "governance": "Family council",
            "memories": [],
            "succession_rules": "Eldest available member presides",
            "assets": [],
            "debts": [],
            "claims": [],
        }
        inst_node = Node(id=inst_id, kind="institution", name=f"Family {new_family}", traits=inst_traits, state=inst_state)
        state.nodes[inst_id] = inst_node
        # Connect membership edge
        self._ensure_edge(state, node.id, inst_id, "Family institution membership", strength=0.65, enforceability=0.5, visibility=0.7)
        return new_family

    def _write_tick_event(self, state: SimulationState, event_log: List[str]) -> None:
        summary = self.llm.summarize_tick(event_log, state.t)
        payload = {"t": state.t, "summary": summary, "events": event_log}
        dir_path = ensure_tick_events_dir()
        path = Path(dir_path) / f"t{state.t}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
