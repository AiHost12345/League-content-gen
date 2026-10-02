"""Win conditions (modelled on YordleDiff's analyzer, with fitted weights).

Each champion gets a seven-dimension signal vector. A team vector is the sum
of its five champions. The archetype is the best cosine match against one
shared archetype list; the fatal flaw is the dimension with the largest gap in
your favour. Unlike a hand-graded table, dimension weights can be fitted from
match data: which team-vector gaps actually predict wins.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from buildopt.ddragon import StaticData

SIGNALS = ("engage", "mobility", "disengage", "waveclear", "scaling", "frontline", "tempo")

# One archetype list used everywhere. Weights are over SIGNALS.
ARCHETYPES = {
    "teamfight": (1.0, 0.1, 0.3, 0.6, 0.6, 0.9, 0.1),
    "pick": (0.7, 0.8, 0.1, 0.1, 0.1, 0.1, 0.8),
    "siege": (0.1, 0.1, 0.8, 1.0, 0.5, 0.3, 0.3),
    "split push": (0.1, 0.9, 0.3, 0.5, 0.6, 0.3, 0.6),
    "protect the carry": (0.2, 0.1, 1.0, 0.4, 1.0, 0.7, 0.0),
    "dive": (0.9, 0.9, 0.0, 0.2, 0.2, 0.5, 0.8),
}

SIGNAL_PHRASES = {
    "engage": "start fights on your terms",
    "mobility": "play around picks and flanks",
    "disengage": "deny their engages and kite",
    "waveclear": "siege towers and stall with waves",
    "scaling": "play for the late game",
    "frontline": "front-to-back teamfights",
    "tempo": "snowball early objectives",
}


def _c(x: float) -> float:
    return max(0.0, min(1.0, x))


def champion_signals(profile: dict, tags: Iterable[str]) -> list[float]:
    tags = set(tags)
    tank, cc, ranged, poke = profile["tank"], profile["cc"], profile["ranged"], profile["poke"]
    mob = profile.get("mobility", 0.4)
    scaling = profile.get("scaling", {"Marksman": 0.8, "Mage": 0.6, "Tank": 0.5}.get(next(iter(tags), ""), 0.4))
    wave = 0.3 + 0.35 * ("Mage" in tags) + 0.3 * ("Marksman" in tags) + 0.1 * poke
    return [round(v, 3) for v in (
        _c(0.4 * tank + 0.45 * cc + 0.15 * (1 - ranged)),
        _c(mob),
        _c(0.5 * cc + 0.3 * ranged + 0.2 * poke),
        _c(wave),
        _c(scaling),
        _c(tank),
        _c(0.5 * (1 - scaling) + 0.3 * profile.get("burst", 0.4) + 0.2 * mob),
    )]


def signals_from_profiles(profiles: dict, static: StaticData) -> dict:
    out = {}
    for cid, prof in profiles.items():
        c = static.champion(int(cid))
        out[str(cid)] = champion_signals(prof, c["tags"] if c else [])
    return {"signals": list(SIGNALS), "champions": out, "weights": [1.0] * len(SIGNALS)}


def team_vector(champions: Sequence[int], table: dict) -> np.ndarray:
    vecs = [table["champions"].get(str(c)) for c in champions if c]
    vecs = [v for v in vecs if v]
    return np.sum(vecs, axis=0) if vecs else np.zeros(len(SIGNALS))


def archetype(vec: np.ndarray) -> tuple[str, float]:
    best, score = "unknown", -1.0
    n = float(np.linalg.norm(vec))
    if n == 0:
        return best, 0.0
    for name, w in ARCHETYPES.items():
        w = np.asarray(w)
        cos = float(vec @ w / (n * np.linalg.norm(w)))
        if cos > score:
            best, score = name, cos
    return best, score


def analyze(allies: Sequence[int], enemies: Sequence[int], table: dict) -> dict:
    mine, theirs = team_vector(allies, table), team_vector(enemies, table)
    weights = np.asarray(table.get("weights", [1.0] * len(SIGNALS)))
    gap = (mine - theirs) * weights
    flaw = int(np.argmax(gap))
    arch_me, _ = archetype(mine)
    arch_them, _ = archetype(theirs)
    edge = float(gap.sum()) * table.get("scale", 0.0)
    return {
        "your_archetype": arch_me,
        "enemy_archetype": arch_them,
        "fatal_flaw": SIGNALS[flaw],
        "advice": f"Exploit their weak {SIGNALS[flaw]}: {SIGNAL_PHRASES[SIGNALS[flaw]]}.",
        "gaps": dict(zip(SIGNALS, (round(float(x), 2) for x in gap))),
        "comp_edge_logit": round(edge, 3),
    }


def fit_signal_weights(rows: Iterable[dict], table: dict, ridge: float = 5.0) -> dict:
    """Fit dimension weights from which team-vector gaps predict wins (one sample per team per game)."""
    from buildopt.analysis.model import fit_logistic

    teams: dict[tuple, dict] = defaultdict(lambda: {"champs": [], "win": 0})
    for r in rows:
        t = teams[(r["match_id"], r["team_id"])]
        t["champs"].append(r["champion_id"])
        t["win"] = int(r["win"])
    X, y = [], []
    for (mid, team), t in teams.items():
        other = teams.get((mid, 300 - team))
        if not other or len(t["champs"]) != 5 or len(other["champs"]) != 5:
            continue
        X.append(team_vector(t["champs"], table) - team_vector(other["champs"], table))
        y.append(t["win"])
    if len(y) < 200:
        return table
    X = np.array(X)
    beta, _ = fit_logistic(X, np.array(y, float), np.ones(len(y)), np.full(X.shape[1], ridge))
    scale = float(np.abs(beta).max()) or 1.0
    table = dict(table)
    table["weights"] = [round(float(b / scale), 4) for b in beta]
    table["scale"] = round(scale, 5)
    table["fitted_games"] = len(y) // 2
    return table


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0
