import random
from typing import Any, Dict, List, Tuple

from .models import Edge, Node, new_id, now_ts
from .state import SimulationState


def _sample_range(bounds: List[float]) -> float:
    low, high = bounds
    return random.uniform(low, high)


def _family_sizes(population: int, sizes: List[int]) -> List[int]:
    result = []
    idx = 0
    remaining = population
    while remaining > 0:
        size = sizes[idx % len(sizes)]
        size = min(size, remaining)
        result.append(size)
        remaining -= size
        idx += 1
    return result


def seed_state(params: Dict[str, Any]) -> SimulationState:
    seeding = params["seeding"]
    env_cfg = params["environment"]
    t0 = SimulationState(t=0, environment=env_cfg, nodes={}, edges={}, events=[])

    population = seeding["population_size"]
    family_sizes = _family_sizes(population, seeding["family_size_distribution"])
    ts = now_ts()

    agents: List[Node] = []
    edges: List[Edge] = []
    family_members: Dict[str, List[Node]] = {}
    for family_idx, fam_size in enumerate(family_sizes):
        family_id = new_id(f"fam{family_idx}")
        family_members[family_id] = []
        members = []
        for _ in range(fam_size):
            agent_id = new_id("agent")
            traits = {
                "created_at": ts,
                "birth_family": family_id,
                "risk_tolerance": _sample_range(seeding["risk_tolerance_range"]),
                "patience": _sample_range(seeding["patience_range"]),
                "norm_sensitivity": _sample_range(seeding["norm_sensitivity_range"]),
                "trust_disposition": _sample_range(seeding["trust_disposition_range"]),
                "reputation_sensitivity": _sample_range(seeding["reputation_sensitivity_range"]),
                "enforcement_aversion": _sample_range(seeding["enforcement_aversion_range"]),
                "adaptability": _sample_range(seeding["adaptability_range"]),
                "memory_capacity": int(_sample_range(seeding["memory_capacity_range"])),
                "network_bandwidth": int(_sample_range(seeding["network_bandwidth_range"])),
                "exploration_rate": _sample_range(seeding["exploration_rate_range"]),
                "reproduction_rate": _sample_range(seeding["reproduction_rate_range"]),
                "shock_exposure": _sample_range(seeding["shock_exposure_range"]),
                "lifespan": int(_sample_range(seeding["lifespan_range"])),
                "personality": "cautious" if random.random() < 0.5 else "bold",
            }
            state = {
                "age": int(random.randint(*seeding["base_age_range"])),
                "last_updated": ts,
                "job": "Informal",
                "memories": [{"id": new_id("mem"), "created_at": ts, "description": f"Born into family {family_id}"}],
                "assets": [
                    {
                        "id": new_id("asset"),
                        "created_at": ts,
                        "last_updated": ts,
                        "value": round(_sample_range(seeding["base_wealth_range"]), 2),
                        "status": "active",
                        "description": "Initial endowment",
                    }
                ],
                "debts": [],
                "claims": [],
                "reputation": _sample_range(seeding["initial_reputation"]),
                "succession_intent": "Undecided",
            }
            node = Node(id=agent_id, kind="agent", name=f"Agent {agent_id[-4:]}", traits=traits, state=state)
            agents.append(node)
            members.append(node)
            family_members[family_id].append(node)
        # connect family members
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                e_id = new_id("edge")
                edge = Edge(
                    id=e_id,
                    source=members[i].id,
                    target=members[j].id,
                    metadata={"id": e_id, "created_at": ts, "last_updated": ts},
                    characteristics={
                        "strength": 0.7,
                        "consent": True,
                        "visibility": 0.8,
                        "activation_rate": 0.5,
                        "enforceability": 0.6,
                    },
                    lifecycle={"strength_decay_rate": None, "expiration": None},
                    specifics={
                        "relationship_description": "Family tie",
                        "expectations": "Mutual support",
                        "history": "Born into same family",
                    },
                )
                edges.append(edge)
                # add reverse edge to keep directionality
                e_id2 = new_id("edge")
                edges.append(
                    Edge(
                        id=e_id2,
                        source=members[j].id,
                        target=members[i].id,
                        metadata={"id": e_id2, "created_at": ts, "last_updated": ts},
                        characteristics={
                            "strength": 0.7,
                            "consent": True,
                            "visibility": 0.8,
                            "activation_rate": 0.5,
                            "enforceability": 0.6,
                        },
                        lifecycle={"strength_decay_rate": None, "expiration": None},
                        specifics={
                            "relationship_description": "Family tie",
                            "expectations": "Mutual support",
                            "history": "Born into same family",
                        },
                    )
                )

    # Institutions: one per family, representing the family institution.
    institutions: List[Node] = []
    for family_id, members in family_members.items():
        inst_id = new_id("inst")
        traits = {
            "created_at": ts,
            "memory_capacity": 20,
            "rigidity": 0.45,
            "institution_type": "Family",
            "institution_description": f"Institutional form of family {family_id}",
        }
        state = {
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
        inst_node = Node(id=inst_id, kind="institution", name=f"Family {family_id}", traits=traits, state=state)
        institutions.append(inst_node)

        # Create membership edges between institution and members.
        for member in members:
            e_id_out = new_id("edge")
            edges.append(
                Edge(
                    id=e_id_out,
                    source=inst_id,
                    target=member.id,
                    metadata={"id": e_id_out, "created_at": ts, "last_updated": ts},
                    characteristics={
                        "strength": 0.65,
                        "consent": True,
                        "visibility": 0.7,
                        "activation_rate": 0.5,
                        "enforceability": 0.5,
                    },
                    lifecycle={"strength_decay_rate": None, "expiration": None},
                    specifics={
                        "relationship_description": "Family institution membership",
                        "expectations": "Support and coordination",
                        "history": "Member of this family",
                    },
                )
            )
            e_id_in = new_id("edge")
            edges.append(
                Edge(
                    id=e_id_in,
                    source=member.id,
                    target=inst_id,
                    metadata={"id": e_id_in, "created_at": ts, "last_updated": ts},
                    characteristics={
                        "strength": 0.65,
                        "consent": True,
                        "visibility": 0.7,
                        "activation_rate": 0.5,
                        "enforceability": 0.5,
                    },
                    lifecycle={"strength_decay_rate": None, "expiration": None},
                    specifics={
                        "relationship_description": "Family institution membership",
                        "expectations": "Support and coordination",
                        "history": "Member of this family",
                    },
                )
            )

    t0.nodes = {n.id: n for n in agents + institutions}
    t0.edges = {e.id: e for e in edges}
    t0.events.append({"t": 0, "notes": "Seeded initial population"})
    return t0
