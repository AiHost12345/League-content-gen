"""Bundle download and local cache.

Bundles are fetched when a patch lands (app start or on demand), never in the
champ select critical path: scoring always reads the local copy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

import requests

from buildopt.bundle import bundle_filename
from buildopt.scoring import Scorer

log = logging.getLogger(__name__)


def _read(source: str, name: str) -> bytes:
    if source.startswith(("http://", "https://")):
        resp = requests.get(source.rstrip("/") + "/" + name, timeout=30)
        resp.raise_for_status()
        return resp.content
    return (Path(source) / name).read_bytes()


def sync_bundles(source: str, dest: str | Path) -> list[str]:
    """Download every bundle whose hash differs from the local copy. Returns updated keys."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    remote = json.loads(_read(source, "index.json"))
    local_index = dest / "index.json"
    local = json.loads(local_index.read_text()) if local_index.exists() else {"bundles": {}}
    updated = []
    for key, entry in remote["bundles"].items():
        target = dest / entry["file"]
        if local["bundles"].get(key, {}).get("sha256") == entry["sha256"] and target.exists():
            continue
        data = _read(source, entry["file"])
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            log.warning("hash mismatch for %s; skipping", entry["file"])
            continue
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(data)
        shutil.move(tmp, target)
        local["bundles"][key] = entry
        updated.append(key)
    local_index.write_text(json.dumps(local, indent=1))
    return updated


class BundleCache:
    def __init__(self, bundle_dir: str | Path):
        self.dir = Path(bundle_dir)
        self._scorers: dict[tuple[int, str], Scorer] = {}

    def available(self) -> dict[tuple[int, str], dict]:
        idx = self.dir / "index.json"
        if not idx.exists():
            return {}
        entries = json.loads(idx.read_text())["bundles"].values()
        return {(e["champion_id"], e["role"]): e for e in entries}

    def scorer(self, champion_id: int, role: str | None) -> Scorer | None:
        """Scorer for a champion and role. In blind pick (no role) use the champion's best-covered bundle."""
        if role is None:
            roles = [r for (c, r) in self.available() if c == champion_id]
            if not roles:
                return None
            role = roles[0]
        key = (champion_id, role)
        if key not in self._scorers:
            path = self.dir / bundle_filename(champion_id, role)
            if not path.exists():
                return None
            self._scorers[key] = Scorer(json.loads(path.read_text(encoding="utf-8")))
        return self._scorers[key]

    def clear(self) -> None:
        self._scorers.clear()
