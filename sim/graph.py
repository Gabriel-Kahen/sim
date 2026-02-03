from typing import Dict, List

from .models import Edge, Node
from .state import SimulationState


def _safe(s: str) -> str:
    return s.replace('"', '\\"')


def _node_label(node: Node) -> str:
    rep = node.state.get("reputation", 0.5)
    job = node.state.get("job") if isinstance(node.state, dict) else None
    job_line = job if isinstance(job, str) and job.strip() else ""
    base = f"{node.name}"
    if job_line:
        base += f"\\n{job_line}"
    base += f"\\nrep={rep:.2f}"
    return base


def _edge_label(edge: Edge) -> str:
    c = edge.characteristics
    s = c.get("strength", 0)
    desc = edge.specifics.get("relationship_description", "")
    if desc == "New interaction edge":
        desc = "Relationship"
    return f"{desc} (s={s:.2f})"


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
        if node.kind == "agent":
            fill = "#c8f7c5"
            shape = "circle"
            size_attrs = ', width="1.2", height="1.2", fixedsize=true, fontsize="9"'
        else:
            inst_type = node.traits.get("institution_type") if isinstance(node.traits, dict) else ""
            fill = "#fff5c2" if str(inst_type).lower() == "family" else "#dce9ff"
            shape = "box"
            size_attrs = ', width="1.8", height="1.1", fixedsize=true'
        lines.append(
            f'  "{_safe(node.id)}" [label="{_node_label(node)}", fillcolor="{fill}", shape="{shape}"{size_attrs}];'
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
