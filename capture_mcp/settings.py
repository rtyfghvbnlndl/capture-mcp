"""Choices the agent makes at runtime (select_device), remembered across server restarts."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def settings_path() -> Path:
    custom = os.environ.get("CAPTURE_SETTINGS_FILE", "").strip()
    return Path(custom) if custom else Path.home() / ".capture-mcp" / "settings.json"


def load() -> dict:
    p = settings_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.warning("ignoring unreadable settings file %s: %s", p, e)
        return {}


def save(data: dict) -> Path:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p
