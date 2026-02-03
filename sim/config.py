import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS_PATH = ROOT / "params.yaml"
DATA_DIR = ROOT / "data"
TICK_EVENTS_DIR = DATA_DIR / "tick_events"


def load_env_key() -> str | None:
    """
    Legacy helper to load a Gemini API key from environment or .env file.
    Vertex AI is now used by default; this is kept for backward compatibility.
    """
    load_dotenv()
    key = os.getenv("Gemini_Key") or os.getenv("GEMINI_KEY")
    return key


def load_params(path: Path | str | None = None) -> Dict[str, Any]:
    params_path = Path(path) if path else DEFAULT_PARAMS_PATH
    with params_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def ensure_tick_events_dir() -> Path:
    TICK_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    return TICK_EVENTS_DIR
