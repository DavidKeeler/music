"""Download state tracking via .download_state.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = ".download_state.json"


def load_state(data_dir: Path) -> dict:
    path = data_dir / STATE_FILE
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(data_dir: Path, state: dict) -> None:
    path = data_dir / STATE_FILE
    path.write_text(json.dumps(state, indent=2) + "\n")


def mark_complete(data_dir: Path, state: dict, name: str) -> None:
    state[name] = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_state(data_dir, state)


def mark_failed(data_dir: Path, state: dict, name: str, error: str) -> None:
    state[name] = {"status": "failed", "error": error}
    save_state(data_dir, state)


def is_complete(state: dict, name: str) -> bool:
    return state.get(name, {}).get("status") == "complete"
