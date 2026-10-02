"""Load the per-player rows for one champion and role, with weights.

Weights encode the patch policy (current patch primary, previous patch at
reduced weight until the current one has enough games, rows touching items or
runes changed in the patch notes zeroed) and optional rank weighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from buildopt.analysis.traits import trait_vector
from buildopt.pipeline.store import Store, patch_key

TIER_ORDER = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER")


@dataclass
class DatasetConfig:
    patch: str | None = None  # current patch; default = newest in the store
    prev_weight: float = 0.3
    min_current_games: int = 5000  # once the current patch has this many games, drop the previous one
    changed_items: set[int] = field(default_factory=set)  # changed in `patch` per patch notes
    changed_runes: set[int] = field(default_factory=set)
    min_tier: str = "EMERALD"
    tier_weights: dict[str, float] = field(default_factory=dict)  # e.g. {"MASTER": 1.5}


@dataclass
class Game:
    row: dict
    win: int
    weight: float
    path: tuple[int, ...]  # completed legendaries in order (no boots)
    path_min: tuple[float, ...]
    boots: int | None
    boots_slot: int | None  # number of legendaries finished before boots
    first_lane_gd: float
    first_team_gd: float
    first_min: float
    page: tuple  # (primary_style, sub_style, perks...)
    shards: tuple[int, int, int]
    keystone: int
    c_raw: list[float]
    enemies: list[dict]
    decision_min: float = 0.0  # set by path comparisons: minute the decision item finished

    @property
    def duration_s(self) -> int:
        return self.row["duration_s"]


def page_key(row: dict) -> tuple:
    return (row["primary_style"], row["sub_style"], *row["perks"])


def to_game(row: dict, profiles: dict, weight: float) -> Game | None:
    legends = [it for it in row["items"] if not it.get("boots")]
    if not legends or len(row["perks"]) != 6:
        return None
    boots = next((it for it in row["items"] if it.get("boots")), None)
    boots_slot = None
    if boots:
        boots_slot = sum(1 for it in legends if it["min"] <= boots["min"])
    first = legends[0]
    return Game(
        row=row,
        win=int(row["win"]),
        weight=weight,
        path=tuple(it["id"] for it in legends),
        path_min=tuple(it["min"] for it in legends),
        boots=boots["id"] if boots else None,
        boots_slot=boots_slot,
        first_lane_gd=first["lane_gd"],
        first_team_gd=first["team_gd"],
        first_min=first["min"],
        page=page_key(row),
        shards=tuple(row["shards"]),
        keystone=row["keystone"],
        c_raw=trait_vector([e["champion_id"] for e in row["enemies"]], profiles),
        enemies=row["enemies"],
    )


def row_weight(row: dict, cfg: DatasetConfig, current: str, previous: str | None, use_previous: bool) -> float:
    if row["tier"] in TIER_ORDER and TIER_ORDER.index(row["tier"]) < TIER_ORDER.index(cfg.min_tier):
        return 0.0
    if row["patch"] == current:
        w = 1.0
    elif use_previous and row["patch"] == previous:
        w = cfg.prev_weight
        early = {it["id"] for it in row["items"][:4]}
        if early & cfg.changed_items or set(row["perks"]) & cfg.changed_runes:
            w = 0.0
    else:
        return 0.0
    return w * cfg.tier_weights.get(row["tier"], 1.0)


def load_games(store: Store, champion_id: int, role: str, profiles: dict, cfg: DatasetConfig | None = None) -> tuple[list[Game], dict]:
    cfg = cfg or DatasetConfig()
    patches = store.patches()
    if not patches:
        return [], {"patch": None}
    current = cfg.patch or patches[-1]
    older = [p for p in patches if patch_key(p) < patch_key(current)]
    previous = older[-1] if older else None
    n_current = store.count_rows(champion_id, role, current)
    use_previous = previous is not None and n_current < cfg.min_current_games
    wanted = [current] + ([previous] if use_previous else [])
    games = []
    for row in store.iter_rows(champion_id, role, wanted, timeline_only=True):
        w = row_weight(row, cfg, current, previous, use_previous)
        if w <= 0:
            continue
        g = to_game(row, profiles, w)
        if g:
            games.append(g)
    info = {
        "patch": current,
        "previous_patch": previous if use_previous else None,
        "previous_weight": cfg.prev_weight if use_previous else 0.0,
        "games_current": n_current,
        "games_used": len(games),
    }
    return games, info


def iter_all_rows(store: Store, patches: Iterable[str]) -> Iterable[dict]:
    return store.iter_rows(patches=list(patches))
