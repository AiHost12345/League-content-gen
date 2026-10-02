"""Build-path views: prefix tree, pair synergy, and conditional path comparison.

The comparison removes the two biases called out in the design:
  * survivorship - paths are compared at the decision point (games whose
    first k completed items are exactly path A or path B), with the minute the
    last item finished as a covariate and a reported split;
  * selection - games are re-weighted by inverse probability of choosing the
    path given the gold lead at first-item completion, so both paths are
    compared at the same lead.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from buildopt.analysis.dataset import Game
from buildopt.analysis.model import fit_logistic
from buildopt.analysis.traits import conditions
from buildopt.stats import LOW_CONFIDENCE_GAMES, MIN_GAMES, pair_lift, weighted_win_rate


# ---- prefix tree -----------------------------------------------------------

def prefix_tree(games: list[Game], max_depth: int = 3, min_games: int = MIN_GAMES) -> list[dict]:
    """Win rate for every ordered prefix of completed items, as a tree."""
    groups: dict[tuple, list[Game]] = defaultdict(list)
    for g in games:
        for k in range(1, min(max_depth, len(g.path)) + 1):
            groups[g.path[:k]].append(g)

    def node(prefix: tuple) -> dict:
        gs = groups[prefix]
        wr = weighted_win_rate([g.win for g in gs], [g.weight for g in gs])
        children = sorted(
            (c for c in groups if len(c) == len(prefix) + 1 and c[:-1] == prefix and len(groups[c]) >= min_games),
            key=lambda c: -len(groups[c]),
        )
        return {"items": list(prefix), **wr.to_dict(), "children": [node(c) for c in children]}

    roots = sorted((p for p in groups if len(p) == 1 and len(groups[p]) >= min_games), key=lambda p: -len(groups[p]))
    return [node(p) for p in roots]


# ---- pair synergy ----------------------------------------------------------

def pair_synergy(games: list[Game], within: int = 3, min_games: int = MIN_GAMES, strength: float = 20.0) -> list[dict]:
    """Lift for every item pair among the first ``within`` completed items."""
    tot_w = sum(g.weight for g in games)
    if tot_w == 0:
        return []
    champ = sum(g.weight * g.win for g in games) / tot_w
    single: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    pair: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for g in games:
        items = sorted(set(g.path[:within]))
        for a in items:
            s = single[a]
            s[0] += g.weight * g.win
            s[1] += g.weight
            s[2] += 1
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                s = pair[(a, b)]
                s[0] += g.weight * g.win
                s[1] += g.weight
                s[2] += 1

    def rate(wins: float, w: float) -> float:
        return (wins + champ * strength) / (w + strength)

    out = []
    for (a, b), (wins, w, n) in pair.items():
        if n < min_games:
            continue
        wr_ab, wr_a, wr_b = rate(wins, w), rate(*single[a][:2]), rate(*single[b][:2])
        out.append({
            "a": a, "b": b, "games": n,
            "wr_ab": wr_ab, "wr_a": wr_a, "wr_b": wr_b,
            "lift": pair_lift(wr_ab, wr_a, wr_b, champ),
        })
    out.sort(key=lambda d: -d["lift"])
    return out


# ---- conditional comparison ------------------------------------------------

@dataclass
class ConditionResult:
    condition: str
    a: dict | None  # WinRate.to_dict() or None when hidden (< min games)
    b: dict | None
    games_a: int
    games_b: int

    @property
    def difference(self) -> float | None:
        if self.a is None or self.b is None:
            return None
        return self.a["win_rate"] - self.b["win_rate"]

    @property
    def confidence(self) -> str | None:
        if self.a is None or self.b is None:
            return None
        order = {"low": 0, "medium": 1, "high": 2}
        return min(self.a["confidence"], self.b["confidence"], key=order.get)

    def to_dict(self) -> dict:
        return {"condition": self.condition, "a": self.a, "b": self.b, "games_a": self.games_a,
                "games_b": self.games_b, "difference": self.difference, "confidence": self.confidence}


def propensity_weights(games_a: list[Game], games_b: list[Game], clip: float = 0.05) -> tuple[list[float], list[float]]:
    """Stabilised inverse-probability weights for choosing path A over B.

    Covariates: lane and team gold difference and minute at first-item
    completion, plus the minute the decision item finished.
    """
    games = games_a + games_b
    if not games_a or not games_b:
        return [g.weight for g in games_a], [g.weight for g in games_b]
    raw = np.array([[g.first_lane_gd, g.first_team_gd, g.first_min, g.decision_min] for g in games], float)
    sd = raw.std(axis=0)
    Z = (raw - raw.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    X = np.hstack([np.ones((len(games), 1)), Z])
    y = np.array([1.0] * len(games_a) + [0.0] * len(games_b))
    w = np.array([g.weight for g in games], float)
    beta, _ = fit_logistic(X, y, w, np.array([1e-6] + [0.5] * Z.shape[1]))
    p = np.clip(1 / (1 + np.exp(-(X @ beta))), clip, 1 - clip)
    pa = float(np.sum(w * y) / np.sum(w))
    ipw = np.where(y == 1, pa / p, (1 - pa) / (1 - p)) * w
    return ipw[: len(games_a)].tolist(), ipw[len(games_a):].tolist()


def decision_point(games: list[Game], path: tuple[int, ...]) -> list[Game]:
    k = len(path)
    out = []
    for g in games:
        if g.path[:k] == path:
            g.decision_min = g.path_min[k - 1]
            out.append(g)
    return out


def compare_paths(games: list[Game], a: tuple[int, ...], b: tuple[int, ...], levels: dict,
                  min_games: int = MIN_GAMES, adjust: bool = True) -> dict:
    if len(a) != len(b):
        raise ValueError("Paths must have the same length to be compared at the decision point.")
    ga, gb = decision_point(games, a), decision_point(games, b)
    if adjust:
        wa, wb = propensity_weights(ga, gb)
    else:
        wa, wb = [g.weight for g in ga], [g.weight for g in gb]

    conds = conditions(levels)
    mins = sorted(g.decision_min for g in ga + gb)
    if len(mins) >= 3:
        t1, t2 = mins[len(mins) // 3], mins[(2 * len(mins)) // 3]
        conds += [
            (f"item {len(a)} done before {t1:.0f} min", lambda c, r, t1=t1: r["_dmin"] < t1),
            (f"item {len(a)} done {t1:.0f}–{t2:.0f} min", lambda c, r, t1=t1, t2=t2: t1 <= r["_dmin"] < t2),
            (f"item {len(a)} done after {t2:.0f} min", lambda c, r, t2=t2: r["_dmin"] >= t2),
        ]

    results = []
    for label, pred in conds:
        sel_a = [(g, w) for g, w in zip(ga, wa) if pred(g.c_raw, {**g.row, "_dmin": g.decision_min})]
        sel_b = [(g, w) for g, w in zip(gb, wb) if pred(g.c_raw, {**g.row, "_dmin": g.decision_min})]

        def summary(sel):
            if len(sel) < min_games:
                return None
            return weighted_win_rate([g.win for g, _ in sel], [w for _, w in sel]).to_dict()

        results.append(ConditionResult(label, summary(sel_a), summary(sel_b), len(sel_a), len(sel_b)))

    return {
        "path_a": list(a),
        "path_b": list(b),
        "adjusted": adjust,
        "games_a": len(ga),
        "games_b": len(gb),
        "conditions": [r.to_dict() for r in results],
        "headline": headline(results),
    }


def headline(results: list[ConditionResult], names: tuple[str, str] = ("A", "B")) -> dict:
    """Overall leader, and the conditions under which the trailing path wins instead."""
    overall = next((r for r in results if r.condition == "all games"), None)
    if overall is None or overall.difference is None:
        return {"leader": None, "difference": None, "flips": []}
    leader = "a" if overall.difference >= 0 else "b"
    flips = []
    for r in results:
        if r.condition == "all games" or r.difference is None:
            continue
        lead_here = "a" if r.difference >= 0 else "b"
        if lead_here != leader and min(r.games_a, r.games_b) >= MIN_GAMES:
            flips.append({"condition": r.condition, "difference": r.difference, "confidence": r.confidence,
                          "reliable": min(r.games_a, r.games_b) > LOW_CONFIDENCE_GAMES})
    flips.sort(key=lambda f: -abs(f["difference"]))
    return {"leader": leader, "difference": abs(overall.difference), "flips": flips}


def stability(games: list[Game], a: tuple, b: tuple, levels: dict, min_games: int = MIN_GAMES) -> dict:
    """Milestone 2 gate: run the comparison on two disjoint halves of the data and check agreement."""
    halves = ([], [])
    for g in games:
        halves[zlib.crc32(g.row["match_id"].encode()) % 2].append(g)
    runs = [compare_paths(h, a, b, levels, min_games) for h in halves]
    agree, total = 0, 0
    for c1, c2 in zip(runs[0]["conditions"], runs[1]["conditions"]):
        if c1["difference"] is None or c2["difference"] is None:
            continue
        total += 1
        agree += (c1["difference"] >= 0) == (c2["difference"] >= 0)
    return {"conditions_compared": total, "sign_agreement": agree / total if total else None, "runs": runs}
