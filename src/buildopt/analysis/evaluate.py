"""Held-out evaluation (milestone 3 gate): does the joint model beat simple baselines?

Games are split by match id. The model is scored the way it is used in champ
select: game-state controls at their average (they are unknown before the
game), so only the loadout and the enemy comp inform the prediction.
"""

from __future__ import annotations

import zlib
from collections import defaultdict

import numpy as np

from buildopt.analysis.candidates import candidate_pages, candidate_paths
from buildopt.analysis.dataset import Game
from buildopt.analysis.model import Penalties, build_model, tune_penalties
from buildopt.stats import MIN_GAMES


def split(games: list[Game], test_fraction: int = 5) -> tuple[list[Game], list[Game]]:
    train, test = [], []
    for g in games:
        (test if zlib.crc32(g.row["match_id"].encode()) % test_fraction == 0 else train).append(g)
    return train, test


def _ll(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def evaluate(games: list[Game], min_games: int = MIN_GAMES, penalties: Penalties | None = None) -> dict:
    train, test = split(games)
    paths = candidate_paths(train, min_games)
    pages = candidate_pages(train, min_games)
    tuning = {}
    if penalties is None:
        penalties, tuning = tune_penalties(train, paths, pages)
    model = build_model(train, paths, pages).fit(train, penalties)
    test = model.usable(test)
    if not test:
        raise ValueError("No held-out games cover the candidate paths and pages.")
    y = np.array([g.win for g in test], float)
    w = np.array([g.weight for g in test], float)
    pi = {p: i for i, p in enumerate(model.paths)}
    ri = {p: i for i, p in enumerate(model.pages)}

    def predict(with_state: bool) -> np.ndarray:
        X = np.vstack([
            model.row(pi[g.path[:3]], ri[g.page], model.standardise_traits(g.c_raw),
                      model.controls(g) if with_state else None,
                      {(e["champion_id"], e["role"]) for e in g.enemies})
            for g in test
        ])
        eta = X @ model.beta
        if with_state:
            return 1 / (1 + np.exp(-eta))
        return model.marginal(eta)[0]  # state unknown pre-game: average over it

    tr = model.usable(train)
    base = sum(g.weight * g.win for g in tr) / sum(g.weight for g in tr)
    acc: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for g in tr:
        a = acc[(g.path[:3], g.page)]
        a[0] += g.weight * g.win
        a[1] += g.weight
    strength = 20.0
    loadout_rate = np.array([(acc[(g.path[:3], g.page)][0] + base * strength) / (acc[(g.path[:3], g.page)][1] + strength)
                             for g in test])
    results = {
        "test_games": len(test),
        "train_games": len(tr),
        "penalty_scales": tuning.get("scales"),
        "log_likelihood": {
            "champion_average": _ll(np.full(len(test), base), y, w),
            "loadout_win_rate": _ll(loadout_rate, y, w),
            "model_comp_only": _ll(predict(False), y, w),
            "model_with_game_state": _ll(predict(True), y, w),
        },
    }
    ll = results["log_likelihood"]
    results["gate_passed"] = ll["model_comp_only"] > max(ll["champion_average"], ll["loadout_win_rate"])
    return results
