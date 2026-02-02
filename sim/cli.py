import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_PARAMS_PATH, load_env_key, load_params
from .graph import state_to_dot
from .llm import LLMClient
from .persistence import get_state, list_states, persist_state, set_current
from .pipeline import Simulator
from .seed import seed_state


def cmd_init(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    state = seed_state(params)
    state_id = persist_state(state)
    print(f"Initialized state {state_id} at t={state.t}")


def cmd_step(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    engine_cfg = params.get("engine", {})
    state = get_state()
    # Allow environment tweaks between runs by reloading from params.
    state.environment = params.get("environment", state.environment)
    api_key = load_env_key()
    llm = LLMClient(api_key=api_key)
    for _ in range(args.steps):
        sim = Simulator(state, engine_cfg, llm)
        state = sim.step()
        state_id = persist_state(state)
        print(f"Step -> t={state.t} saved as {state_id}")


def cmd_list(_: argparse.Namespace) -> None:
    states = list_states()
    if not states:
        print("No states saved yet.")
        return
    for entry in states:
        print(f"{entry['id']} (t={entry['t']}) -> {entry['path']}")


def cmd_show(args: argparse.Namespace) -> None:
    state = get_state(args.id)
    summary = {
        "t": state.t,
        "environment": state.environment,
        "node_count": len(state.nodes),
        "edge_count": len(state.edges),
        "sample_nodes": list(state.nodes.keys())[:5],
    }
    print(json.dumps(summary, indent=2))


def cmd_rewind(args: argparse.Namespace) -> None:
    set_current(args.to)
    print(f"Current state set to {args.to}")


def cmd_graph(args: argparse.Namespace) -> None:
    state = get_state(args.id)
    dot = state_to_dot(state)
    if args.out:
        Path(args.out).write_text(dot, encoding="utf-8")
        print(f"Wrote graph DOT to {args.out}")
    else:
        print(dot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Socio-Econ Sim CLI")
    parser.set_defaults(func=None)

    sub = parser.add_subparsers()

    p_init = sub.add_parser("init", help="Seed initial state")
    p_init.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH, help="Path to params.yaml")
    p_init.set_defaults(func=cmd_init)

    p_step = sub.add_parser("step", help="Advance simulation by N steps")
    p_step.add_argument("--steps", type=int, default=1, help="Number of ticks to run")
    p_step.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH, help="Path to params.yaml")
    p_step.set_defaults(func=cmd_step)

    p_list = sub.add_parser("list", help="List saved states")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show summary of current or specific state")
    p_show.add_argument("--id", help="State id to show")
    p_show.set_defaults(func=cmd_show)

    p_rewind = sub.add_parser("rewind", help="Set current pointer to an existing state id")
    p_rewind.add_argument("--to", required=True, help="State id to set as current")
    p_rewind.set_defaults(func=cmd_rewind)

    p_graph = sub.add_parser("graph", help="Output GraphViz DOT for current/specific state")
    p_graph.add_argument("--id", help="State id (defaults to current)")
    p_graph.add_argument("--out", type=Path, help="Path to write DOT (stdout if omitted)")
    p_graph.set_defaults(func=cmd_graph)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
