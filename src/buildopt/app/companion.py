"""Champ select companion: re-score on every enemy lock and import the loadout.

Flow (design doc):
  1. Session starts -> load the bundle for your hovered/locked champion and role.
  2. Each enemy lock -> rebuild the trait vector and re-score (all local).
  3. Your champion locks -> write the rune page and item set.
  4. Later enemy locks -> re-score; if the top loadout changes, rewrite both and
     say what changed.
  5. Final write near the end of finalization so the last enemy pick counts.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from buildopt.analysis import jungle, wincon
from buildopt.app.bundles import BundleCache
from buildopt.itemset import build_item_set
from buildopt.lcu.champselect import ChampSelectState, assign_roles, parse_session
from buildopt.lcu.connection import LcuError
from buildopt.lcu.importer import ImportBlocked, ImportFailed, Importer
from buildopt.scoring import Recommendation, Scorer

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _names() -> dict:
    from buildopt import ddragon

    return {int(k): c["name"] for k, c in ddragon.load(offline=True).champions.items()}


def champion_name(cid: int) -> str:
    return _names().get(int(cid), f"champion {cid}")


class UI(Protocol):
    def show(self, view: dict) -> None: ...
    def status(self, message: str) -> None: ...
    def ask_page(self, pages: list[dict]) -> int | None: ...


@dataclass
class View:
    """Everything the window renders, as plain data."""

    header: str
    runes: str
    build: str
    win: str
    why: list[str]
    runner_up: str | None
    change: str | None
    extras: list[str] = field(default_factory=list)
    other_builds: list[str] = field(default_factory=list)
    read_only: bool = False
    score_ms: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Companion:
    def __init__(self, cache: BundleCache, ui: UI, settings, importer: Importer | None = None, read_only: bool = False,
                 clock=time.monotonic):
        self.cache = cache
        self.ui = ui
        self.settings = settings
        self.importer = importer
        self.read_only = read_only or importer is None
        self.clock = clock
        self.lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        self.scorer: Scorer | None = None
        self.rec: Recommendation | None = None
        self.state: ChampSelectState | None = None
        self.written: tuple | None = None  # (loadout key, enemy key) last written
        self.final_timer: threading.Timer | None = None
        self.change: str | None = None
        self.last_import: dict | None = None

    @property
    def imports_enabled(self) -> bool:
        return self.settings.auto_import and not self.read_only

    # ---- events ------------------------------------------------------------
    def on_session(self, event_type: str, session: dict | None) -> None:
        with self.lock:
            if event_type == "Delete" or not session:
                if self.final_timer:
                    self.final_timer.cancel()
                self.reset()
                self.ui.status("Waiting for champ select")
                return
            self.update(parse_session(session))

    def update(self, state: ChampSelectState) -> None:
        self.state = state
        if not state.my_champion:
            self.ui.status("Waiting for your pick")
            return
        scorer = self.cache.scorer(state.my_champion, state.my_role)
        if scorer is None:
            self.scorer = None
            who = f"{champion_name(state.my_champion)} {(state.my_role or '').lower()}".strip()
            self.ui.status(f"No recommendations for {who} yet. Collect games for it in the Get Data tab "
                           "(only champions you've built recommendations for get imports).")
            return
        if scorer is not self.scorer:
            self.scorer, self.rec, self.written = scorer, None, None

        t0 = self.clock()
        roles = assign_roles(state.enemies, scorer.profiles)
        rec = scorer.recommend(state.enemies, roles, rank_by=getattr(self.settings, "rank_by", "win_rate"),
                               min_games=getattr(self.settings, "min_build_games", 20))
        ms = (self.clock() - t0) * 1000
        if self.rec is not None and rec.best.key != self.rec.best.key:
            self.change = scorer.change_reason(self.rec, rec)
        self.rec = rec
        self.ui.show(self.view(scorer, rec, state, roles, ms).as_dict())

        if state.my_locked and self.imports_enabled:
            if self.written is None or self.written[0] != rec.best.key:
                self.write()
            if state.phase == "FINALIZATION":
                self.schedule_final_write(state.time_left_ms)

    def schedule_final_write(self, time_left_ms: int) -> None:
        if self.final_timer is not None:
            return
        delay = max(0.0, (time_left_ms - self.settings.final_write_lead_ms) / 1000)
        self.final_timer = threading.Timer(delay, self.final_write)
        self.final_timer.daemon = True
        self.final_timer.start()

    def final_write(self) -> None:
        with self.lock:
            if self.rec and self.state and self.imports_enabled:
                if self.written != (self.rec.best.key, self.state.enemy_key):
                    self.write()

    # ---- import ------------------------------------------------------------
    def write(self) -> bool:
        """Write the rune page and the item set independently, so one failing never blocks the other."""
        scorer, rec, state = self.scorer, self.rec, self.state
        if not (scorer and rec and state and self.importer):
            return False
        page = rec.best.page
        name = f"{scorer.bundle['champion_name']} {scorer.perk(page[2])}"
        item_set = build_item_set(scorer, rec, self.settings.page_prefix)
        result = {"runes": self._attempt(lambda: self.importer.write_rune_page(
                      name, page[0], page[1], list(page[2:]), list(rec.shards)), f"'{self.settings.page_prefix} {name}'"),
                  "items": self._attempt(lambda: self.importer.write_item_set(item_set), f"'{item_set['title']}'")}
        self.last_import = result
        report = getattr(self.ui, "import_result", None)
        if report:
            report(result)
        ok = result["runes"][0] or result["items"][0]
        if ok:
            self.written = (rec.best.key, state.enemy_key)
        parts = [f"Runes {'written' if result['runes'][0] else 'FAILED'}",
                 f"item set {'written' if result['items'][0] else 'FAILED'}"]
        self.ui.status(", ".join(parts) + (f" · {self.change}" if self.change and ok else ""))
        return result["runes"][0] and result["items"][0]

    @staticmethod
    def _attempt(fn, what: str) -> tuple[bool, str]:
        try:
            fn()
        except (ImportBlocked, ImportFailed) as e:
            log.warning("import of %s skipped: %s", what, e)
            return False, f"{what}: {e}"
        except LcuError as e:
            log.warning("import of %s failed: %s", what, e)
            return False, f"{what}: the client refused it ({e})"
        except Exception as e:  # never let one write take the app down
            log.exception("import of %s crashed", what)
            return False, f"{what}: {e}"
        return True, what

    # ---- view --------------------------------------------------------------
    def view(self, scorer: Scorer, rec: Recommendation, state: ChampSelectState, roles: dict, ms: float) -> View:
        v = recommendation_view(scorer, rec, roles, allies=state.allies, my_champion=state.my_champion,
                                enemies=state.enemies, my_role=state.my_role)
        v.change = self.change
        v.read_only = not self.imports_enabled
        v.score_ms = ms
        return v


def recommendation_view(scorer: Scorer, rec: Recommendation, roles: dict, allies: dict | None = None,
                        my_champion: int | None = None, enemies: list[int] | None = None,
                        my_role: str | None = None) -> View:
    """Plain-data view of a recommendation, shared by the companion and the GUI."""
    b = rec.best
    enemies = enemies or []
    allies = allies or {}
    runner = None
    if rec.runner_up:
        r = rec.runner_up
        runner = (f"{scorer.perk(r.keystone)} · {scorer.path_label(r.path)} · {r.win_rate:.1%} "
                  f"({(r.win_rate - b.win_rate) * 100:+.1f} pts)")
    extras = []
    wc = scorer.bundle.get("wincon")
    team = list(allies.values())
    if my_champion and my_champion not in team:
        team.append(my_champion)
    if wc and enemies and len(team) >= 2:  # needs at least some of your team to say anything useful
        a = wincon.analyze(team, enemies, wc)
        extras.append(f"Win condition: {a['your_archetype']} vs {a['enemy_archetype']} · {a['advice']}")
    if scorer.bundle.get("jungle") and (my_role or scorer.bundle["role"]) == "JUNGLE":
        enemy_jg = next((c for c, r in roles.items() if r == "JUNGLE"), None)
        routes = jungle.best_routes(scorer.bundle["jungle"], enemy_jg, top=1)
        if routes:
            vs = f" vs {scorer.champion(routes[0]['vs'])}" if routes[0]["vs"] else ""
            extras.append(f"First clear{vs}: {routes[0]['route']} ({routes[0]['win_rate']:.1%})")
        if allies:
            enemies_by_role = {r: c for c, r in roles.items()}
            ganks = jungle.gank_priority(allies, enemies_by_role, scorer.profiles)
            if ganks:
                extras.append("Gank priority: " + " > ".join(g["lane"].lower() for g in ganks))
    return View(
        header=f"{scorer.bundle['champion_name']} {scorer.bundle['role'].lower()} · {rec.data_version}",
        runes=f"{scorer.page_label(b.page)} · {', '.join(scorer.perk(s) for s in rec.shards)}",
        build=scorer.path_label(b.path),
        win=f"{b.win_rate:.1%} (95% {b.lo:.1%}–{b.hi:.1%}) · {b.confidence} confidence",
        why=rec.why,
        runner_up=runner,
        change=None,
        extras=extras,
        other_builds=[f"{scorer.path_label(l.path)} · {l.win_rate:.1%} ({l.games:,} games)"
                      for l in Scorer.top_builds([l for l in rec.ranked if l.games >= rec.min_games], 6)
                      if l.path != b.path][:5],
    )
