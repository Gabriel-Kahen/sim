import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ensure_data_dir
from .models import new_id
from .state import SimulationState, load_state, save_state


INDEX_FILE = ensure_data_dir() / "index.json"


def _default_index() -> Dict[str, Any]:
    return {"current": None, "states": []}


def load_index() -> Dict[str, Any]:
    if not INDEX_FILE.exists():
        return _default_index()
    with INDEX_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_index(index: Dict[str, Any]) -> None:
    ensure_data_dir()
    with INDEX_FILE.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def persist_state(state: SimulationState) -> str:
    ensure_data_dir()
    index = load_index()
    state_id = new_id(f"t{state.t}")
    path = INDEX_FILE.parent / f"{state_id}.json"
    save_state(path, state)
    meta = {
        "id": state_id,
        "t": state.t,
        "path": str(path),
    }
    index["states"].append(meta)
    index["current"] = state_id
    save_index(index)
    return state_id


def list_states() -> List[Dict[str, Any]]:
    index = load_index()
    return index.get("states", [])


def get_state(state_id: Optional[str] = None) -> SimulationState:
    index = load_index()
    sid = state_id or index.get("current")
    if not sid:
        raise FileNotFoundError("no state found; run init first")
    for entry in index.get("states", []):
        if entry["id"] == sid:
            return load_state(Path(entry["path"]))
    raise FileNotFoundError(f"state {sid} not found")


def set_current(state_id: str) -> None:
    index = load_index()
    if not any(entry["id"] == state_id for entry in index.get("states", [])):
        raise FileNotFoundError(f"state {state_id} not found")
    index["current"] = state_id
    save_index(index)
