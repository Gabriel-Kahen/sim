# Socio-Econ Sim (CLI)

Python CLI implementing the described socio-economic simulation with a Gemini Writer → Compiler pipeline, persistent state history, and tunable parameters in `params.yaml`.

## Setup
- Python 3.10+ recommended.
 - Install deps: `pip install -r requirements.txt`
 - Auth for Vertex AI (application-default credentials), e.g. `gcloud auth application-default login` and ensure the right project is selected.
 - Set `.env` with `VERTEX_PROJECT=<your_project>`, optional `VERTEX_LOCATION` (default `us-central1`), and `VERTEX_MODEL` (default `gemini-2.5-flash-lite`). LLM failures will raise; no mock fallback.

## Config
- `params.yaml` holds seeding, environment, and engine defaults. Edit and rerun steps to apply.

## Commands
- `python3 -m sim.cli init` — seed initial state and save snapshot under `data/`.
 - `python3 -m sim.cli step --steps 3` — advance the sim; saves each tick as a new state file and updates the current pointer.
- `python3 -m sim.cli list` — list saved states.
- `python3 -m sim.cli show [--id <state_id>]` — show summary of current or specified state.
- `python3 -m sim.cli rewind --to <state_id>` — point "current" to an existing snapshot (history is preserved).
- `python3 -m sim.cli graph [--id <state_id>] [--out out.dot]` — emit GraphViz DOT for the current (or specified) state; defaults to current state and writes `out.dot`.
- After each `step`, a tick event log is written to `data/tick_events/t<t>.json` with the raw event list and an LLM-generated summary for that tick.

## Notes
- LLM pipeline: thin observation packets → Gemini Writer (narrative) → Gemini Compiler (JSON patches; falls back to deterministic memory appends if unusable) → engine validation/apply → environment processes (shocks, inflow, friction/clamps) → memory maintenance.
- Persistence: each tick saved as `data/t{t}-<uuid>.json` with `data/index.json` tracking history and current pointer.
- Seeding: agents created in family clusters with initial assets and family edges; one institution per family is added.
- Shocks/inflows: environment may trigger LLM-generated, storied shocks (with keyword-based targeting) or inflows; effects are added as assets and memories per affected node.
- Edges: enforced single, undirected edge per unordered pair (`edge-<src>-<tgt>`); weakest edges are pruned beyond per-type caps. LLM edge updates merge into the existing edge; bad endpoints are skipped.
- State drift: jobs and succession intent change via LLM patches and light engine drift (random job/intention updates; jobs are LLM-proposed when no preset options). Births and deaths are simulated: agents can have children (inherit family/institution links) and die (assets shift to family institution). Defection: agents can rarely leave their family and form a new family/institution. Memories accrue from interactions, shocks, inflows, and fallback summaries, capped by memory_capacity.
