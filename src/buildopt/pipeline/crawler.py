"""Rank-cascade crawler.

Starts at Challenger, then Grandmaster, then Master, then Diamond I..IV and
Emerald I..IV. Each step is crawled until its players' games on the target
patches are exhausted before moving down. Every row keeps its rank tier.

All state lives in SQLite, so the crawl can be stopped and resumed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from buildopt.ddragon import StaticData
from buildopt.pipeline.parse import is_valid_game, needs_timeline, parse_match, patch_of
from buildopt.pipeline.store import Store, patch_key
from buildopt.riot.api import RiotApiError, RiotClient

log = logging.getLogger(__name__)

APEX = ("CHALLENGER", "GRANDMASTER", "MASTER")
DIVISIONS = ("I", "II", "III", "IV")
CASCADE: tuple[tuple[str, str], ...] = (
    *((t, "I") for t in APEX),
    *(("DIAMOND", d) for d in DIVISIONS),
    *(("EMERALD", d) for d in DIVISIONS),
)


@dataclass
class CrawlConfig:
    platforms: list[str]
    patches: list[str]  # patches to keep, e.g. ["16.19", "16.18"]
    since: int | None = None  # epoch seconds; only list matches after this
    targets: set[tuple[int, str]] = field(default_factory=set)  # (champion_id, role); empty = all
    max_target_rows: int | None = None  # stop once this many target rows are stored
    max_pages_per_division: int = 50
    max_matches_per_player: int = 300
    lowest: tuple[str, str] = ("EMERALD", "IV")


class Crawler:
    def __init__(self, client: RiotClient, store: Store, static: StaticData, config: CrawlConfig):
        self.client = client
        self.store = store
        self.static = static
        self.cfg = config
        self.stop_event = threading.Event()
        self.oldest_patch = min(config.patches, key=patch_key)

    # ---- public --------------------------------------------------------
    def run(self) -> None:
        threads = [threading.Thread(target=self._run_platform, args=(p,), name=f"crawl-{p}", daemon=True)
                   for p in self.cfg.platforms]
        for t in threads:
            t.start()
        try:
            for t in threads:
                while t.is_alive():
                    t.join(timeout=1.0)
        except KeyboardInterrupt:
            self.stop_event.set()
            raise

    def target_rows(self) -> int:
        if not self.cfg.targets:
            return self.store.count_rows()
        return sum(self.store.count_rows(c, r or None) for c, r in self.cfg.targets)

    def done(self) -> bool:
        if self.stop_event.is_set():
            return True
        if self.cfg.max_target_rows and self.target_rows() >= self.cfg.max_target_rows:
            self.stop_event.set()
            return True
        return False

    # ---- cascade -------------------------------------------------------
    def steps(self):
        for step in CASCADE:
            yield step
            if step == tuple(self.cfg.lowest):
                return

    def _run_platform(self, platform: str) -> None:
        for tier, division in self.steps():
            if self.done():
                return
            log.info("[%s] crawling %s %s", platform, tier, division)
            self.seed(platform, tier, division)
            self.crawl_tier(platform, tier, division)

    def seed(self, platform: str, tier: str, division: str) -> None:
        key = tier if tier in APEX else f"{tier}_{division}"
        if self.store.tier_seeded(platform, key):
            return
        if tier in APEX:
            entries = self.client.apex_league(platform, tier)
            self.store.add_players(platform, tier, division, [e["puuid"] for e in entries if e.get("puuid")])
        else:
            for page in range(1, self.cfg.max_pages_per_division + 1):
                entries = self.client.league_entries(platform, tier, division, page)
                if not entries:
                    break
                self.store.add_players(platform, tier, division, [e["puuid"] for e in entries if e.get("puuid")])
        self.store.set_tier_seeded(platform, key)

    def crawl_tier(self, platform: str, tier: str, division: str) -> None:
        while not self.done():
            batch = [(p, d) for p, d in self.store.pending_players(platform, tier, 200) if d == division or tier in APEX]
            if not batch:
                return
            for puuid, _ in batch:
                if self.done():
                    return
                try:
                    self.crawl_player(platform, puuid, tier)
                    self.store.mark_player(platform, puuid, "done")
                except RiotApiError as e:
                    log.warning("[%s] player failed: %s", platform, e)
                    self.store.mark_player(platform, puuid, "error")

    def crawl_player(self, platform: str, puuid: str, tier: str) -> None:
        start = 0
        while start < self.cfg.max_matches_per_player:
            ids = self.client.match_ids(platform, puuid, start=start, count=100, start_time=self.cfg.since)
            if not ids:
                return
            for mid in ids:
                if self.done():
                    return
                patch = self.process_match(platform, mid, tier)
                if patch and patch_key(patch) < patch_key(self.oldest_patch):
                    return  # ids are newest first; everything after is older
            if len(ids) < 100:
                return
            start += 100

    def process_match(self, platform: str, match_id: str, tier: str) -> str | None:
        if self.store.has_match(match_id):
            return self.store.match_patch(match_id)
        try:
            match = self.client.match(platform, match_id)
        except RiotApiError as e:
            if e.status == 404:
                self.store.mark_match(match_id, platform, None, "missing")
                return None
            raise
        info = match["info"]
        patch = patch_of(info.get("gameVersion", "0.0"))
        if patch not in self.cfg.patches:
            self.store.mark_match(match_id, platform, patch, "other_patch")
            return patch
        if not is_valid_game(info):
            self.store.mark_match(match_id, platform, patch, "invalid")
            return patch
        timeline = self.client.timeline(platform, match_id) if needs_timeline(match, self.cfg.targets) else None
        rows = parse_match(match, timeline, self.static, tier, platform)
        self.store.add_rows(rows)
        self.store.mark_match(match_id, platform, patch, "stored" if rows else "unparsed")
        return patch
