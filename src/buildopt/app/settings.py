"""Persisted desktop-app settings (~/.buildopt/settings.json)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def app_dir() -> Path:
    return Path(os.environ.get("BUILDOPT_HOME", Path.home() / ".buildopt"))


@dataclass
class Settings:
    auto_import: bool = True  # off = show the loadout read-only
    page_prefix: str = "BO:"
    owned_page_id: int | None = None  # page slot the user granted when the page limit was reached
    asked_for_page: bool = False
    bundle_source: str = ""  # URL or directory with index.json
    bundle_dir: str = field(default_factory=lambda: str(app_dir() / "bundles"))
    league_path: str | None = None
    final_write_lead_ms: int = 3000  # final write this long before finalization ends
    path: str = field(default="", repr=False)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        path = Path(path) if path else app_dir() / "settings.json"
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                data = {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "path"}
        s = cls(**known)
        s.path = str(path)
        return s

    def save(self) -> None:
        if not self.path:
            return
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("path")
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
