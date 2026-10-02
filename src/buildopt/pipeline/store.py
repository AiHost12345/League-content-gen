"""SQLite storage for crawl state and the per-player row table."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    puuid TEXT NOT NULL,
    platform TEXT NOT NULL,
    tier TEXT NOT NULL,
    division TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    PRIMARY KEY (puuid, platform)
);
CREATE INDEX IF NOT EXISTS players_tier ON players(platform, tier, status);
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    patch TEXT,
    status TEXT NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rows (
    match_id TEXT NOT NULL,
    participant_id INTEGER NOT NULL,
    champion_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    patch TEXT NOT NULL,
    tier TEXT NOT NULL,
    win INTEGER NOT NULL,
    has_timeline INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (match_id, participant_id)
);
CREATE INDEX IF NOT EXISTS rows_champ ON rows(champion_id, role, patch);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()

    def close(self) -> None:
        self.conn.close()

    # ---- players -------------------------------------------------------
    def add_players(self, platform: str, tier: str, division: str, puuids: Iterable[str]) -> int:
        with self.lock, self.conn:
            cur = self.conn.executemany(
                "INSERT OR IGNORE INTO players(puuid, platform, tier, division) VALUES (?,?,?,?)",
                [(p, platform, tier, division) for p in puuids],
            )
            return cur.rowcount

    def pending_players(self, platform: str, tier: str, limit: int = 100) -> list[tuple[str, str]]:
        with self.lock:
            return self.conn.execute(
                "SELECT puuid, division FROM players WHERE platform=? AND tier=? AND status='pending' ORDER BY division LIMIT ?",
                (platform, tier, limit),
            ).fetchall()

    def mark_player(self, platform: str, puuid: str, status: str) -> None:
        with self.lock, self.conn:
            self.conn.execute("UPDATE players SET status=? WHERE puuid=? AND platform=?", (status, puuid, platform))

    def tier_seeded(self, platform: str, tier: str) -> bool:
        return self.get_meta(f"seeded:{platform}:{tier}") == "1"

    def set_tier_seeded(self, platform: str, tier: str) -> None:
        self.set_meta(f"seeded:{platform}:{tier}", "1")

    # ---- matches -------------------------------------------------------
    def has_match(self, match_id: str) -> bool:
        with self.lock:
            return self.conn.execute("SELECT 1 FROM matches WHERE match_id=?", (match_id,)).fetchone() is not None

    def match_patch(self, match_id: str) -> str | None:
        with self.lock:
            r = self.conn.execute("SELECT patch FROM matches WHERE match_id=?", (match_id,)).fetchone()
            return r[0] if r else None

    def mark_match(self, match_id: str, platform: str, patch: str | None, status: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO matches(match_id, platform, patch, status, fetched_at) VALUES (?,?,?,?,?)",
                (match_id, platform, patch, status, time.time()),
            )

    def add_rows(self, rows: list[dict]) -> None:
        with self.lock, self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO rows(match_id, participant_id, champion_id, role, patch, tier, win, has_timeline, data)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (r["match_id"], r["participant_id"], r["champion_id"], r["role"], r["patch"], r["tier"],
                     int(r["win"]), int(r["has_timeline"]), json.dumps(r, separators=(",", ":")))
                    for r in rows
                ],
            )

    def iter_rows(self, champion_id: int | None = None, role: str | None = None,
                  patches: Iterable[str] | None = None, timeline_only: bool = False) -> Iterator[dict]:
        q, args = "SELECT data FROM rows WHERE 1=1", []
        if champion_id is not None:
            q += " AND champion_id=?"
            args.append(champion_id)
        if role:
            q += " AND role=?"
            args.append(role)
        if patches:
            patches = list(patches)
            q += f" AND patch IN ({','.join('?' * len(patches))})"
            args.extend(patches)
        if timeline_only:
            q += " AND has_timeline=1"
        with self.lock:
            cur = self.conn.execute(q, args)
            batch = cur.fetchall()
        for (data,) in batch:
            yield json.loads(data)

    def count_rows(self, champion_id: int | None = None, role: str | None = None, patch: str | None = None) -> int:
        q, args = "SELECT COUNT(*) FROM rows WHERE has_timeline=1", []
        if champion_id is not None:
            q += " AND champion_id=?"
            args.append(champion_id)
        if role:
            q += " AND role=?"
            args.append(role)
        if patch:
            q += " AND patch=?"
            args.append(patch)
        with self.lock:
            return self.conn.execute(q, args).fetchone()[0]

    def patches(self) -> list[str]:
        with self.lock:
            ps = [r[0] for r in self.conn.execute("SELECT DISTINCT patch FROM rows")]
        return sorted(ps, key=patch_key)

    def champion_roles(self, min_games: int = 1) -> list[tuple[int, str, int]]:
        with self.lock:
            return self.conn.execute(
                "SELECT champion_id, role, COUNT(*) c FROM rows WHERE has_timeline=1 GROUP BY champion_id, role HAVING c>=?"
                " ORDER BY c DESC",
                (min_games,),
            ).fetchall()

    # ---- meta ----------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self.lock:
            r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return r[0] if r else None

    def set_meta(self, key: str, value: str) -> None:
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value))


def patch_key(patch: str) -> tuple:
    out = []
    for part in patch.split("."):
        out.append(int(part) if part.isdigit() else 0)
    return tuple(out)
