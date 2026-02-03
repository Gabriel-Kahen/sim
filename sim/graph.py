from typing import Dict, List

from .models import Edge, Node
from .state import SimulationState


def _safe(s: str) -> str:
    return s.replace('"', '\\"')


def _node_label(node: Node) -> str:
    rep = node.state.get("reputation", 0.5)
    assets_list = node.state.get("assets", [])
    if not isinstance(assets_list, list):
        assets_list = []
    debts_list = node.state.get("debts", [])
    if not isinstance(debts_list, list):
        debts_list = []
    claims_list = node.state.get("claims", [])
    if not isinstance(claims_list, list):
        claims_list = []
    assets = len(assets_list)
    debts = len(debts_list)
    claims = len(claims_list)
    return f"{node.name}\\n{node.kind}\\nrep={rep:.2f}, A={assets}, D={debts}, C={claims}"


def _edge_label(edge: Edge) -> str:
    c = edge.characteristics
    s = c.get("strength", 0)
    a = c.get("activation_rate", 0)
    e = c.get("enforceability", 0)
    desc = edge.specifics.get("relationship_description", "")
    return f"s={s:.2f}, a={a:.2f}, e={e:.2f}\\n{desc}"


def _edge_color(strength: float) -> str:
    if strength >= 0.66:
        return "#2c7be5"
    if strength >= 0.33:
        return "#6c757d"
    return "#adb5bd"


def state_to_dot(state: SimulationState) -> str:
    lines: List[str] = [
        "digraph G {",
        '  rankdir=LR;',
        '  node [shape=box, style=filled, fillcolor="#f5f5f5", color="#444", fontname="Helvetica"];',
        '  edge [color="#777", fontname="Helvetica", dir="none"];',
    ]

    for node in state.nodes.values():
        fill = "#e8f4ff" if node.kind == "agent" else "#fff4e5"
        lines.append(
            f'  "{_safe(node.id)}" [label="{_node_label(node)}", fillcolor="{fill}"];'
        )

    seen_pairs = set()
    for edge in state.edges.values():
        if edge.source not in state.nodes or edge.target not in state.nodes:
            continue
        key = tuple(sorted((edge.source, edge.target)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        s = edge.characteristics.get("strength", 0)
        color = _edge_color(s)
        lines.append(
            f'  "{_safe(edge.source)}" -> "{_safe(edge.target)}" '
            f'[label="{_edge_label(edge)}", color="{color}"];'
        )

    lines.append("}")
    return "\n".join(lines)
