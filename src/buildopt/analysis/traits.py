"""Enemy comp trait vector ``c``.

Five enemies are summarised as a handful of traits so that sparse
champion-by-champion combinations share statistical strength. The same vector
feeds the item model, the rune model, and the situational item choices.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

TRAITS = ("tanks", "ap_share", "hard_cc", "healing", "burst", "ranged", "poke")

TRAIT_LABELS = {
    "tanks": "tanks",
    "ap_share": "AP damage",
    "hard_cc": "hard CC",
    "healing": "healing",
    "burst": "burst",
    "ranged": "ranged",
    "poke": "poke",
}


def _soft_count(score: float, threshold: float = 0.6, sharpness: float = 12.0) -> float:
    return 1.0 / (1.0 + math.exp(-(score - threshold) * sharpness))


def trait_vector(enemies: Iterable[int], profiles: dict) -> list[float]:
    """Raw (unstandardised) trait vector for up to five enemy champion ids."""
    profs = [profiles.get(int(c)) or profiles.get(str(c)) for c in enemies if c]
    profs = [p for p in profs if p]
    if not profs:
        return [0.0] * len(TRAITS)
    k = len(profs)
    # Scale sums to a full team when only some enemies are known (mid champ select).
    scale = 5.0 / k
    tanks = sum(_soft_count(p["tank"]) for p in profs) * scale
    ap = sum(p["ap_share"] for p in profs) / k
    cc = sum(p["cc"] for p in profs) * scale
    heal = sum(p["heal"] for p in profs) * scale
    burst = sum(p["burst"] for p in profs) * scale
    ranged = sum(p["ranged"] for p in profs) * scale
    poke = sum(p["poke"] for p in profs) * scale
    return [tanks, ap, cc, heal, burst, ranged, poke]


def standardise(c: Sequence[float], mean: Sequence[float], std: Sequence[float]) -> list[float]:
    return [(x - m) / s if s > 0 else 0.0 for x, m, s in zip(c, mean, std)]


def describe(trait: str, value: float, levels: dict | None = None, known: int = 5) -> str:
    """Human phrasing of one trait value, e.g. '3 tanks' or 'heavy healing'.

    Count traits are scaled to a full team in the vector; ``known`` (enemies
    locked so far) turns them back into the count actually on screen.
    """
    if trait in ("tanks", "ranged"):
        n = int(round(value * max(known, 1) / 5))
        if trait == "ranged":
            return f"{n} ranged"
        return f"{n} tank" if n == 1 else f"{n} tanks"
    if trait == "ap_share":
        return f"{round(value * 100)}% AP damage"
    level = trait_level(trait, value, levels)
    word = {"high": "heavy", "low": "little", "mid": "some"}[level]
    return f"{word} {TRAIT_LABELS[trait]}"


def trait_level(trait: str, value: float, levels: dict | None) -> str:
    if not levels or trait not in levels:
        return "mid"
    lo, hi = levels[trait]
    if value >= hi:
        return "high"
    if value <= lo:
        return "low"
    return "mid"


def conditions(levels: dict) -> list[tuple[str, callable]]:
    """Named comp conditions used when splitting comparisons.

    Each returns (label, predicate(raw_trait_vector, row) -> bool).
    """
    idx = {t: i for i, t in enumerate(TRAITS)}
    out = [
        ("all games", lambda c, r: True),
        ("0–1 tanks", lambda c, r: round(c[idx["tanks"]]) <= 1),
        ("2+ tanks", lambda c, r: round(c[idx["tanks"]]) >= 2),
    ]
    for t in ("healing", "ap_share", "hard_cc", "burst", "poke"):
        if t in levels:
            lo, hi = levels[t]
            name = TRAIT_LABELS[t]
            out.append((f"high {name}", lambda c, r, i=idx[t], hi=hi: c[i] >= hi))
            out.append((f"low {name}", lambda c, r, i=idx[t], lo=lo: c[i] <= lo))
    out += [
        ("games under 25 min", lambda c, r: r["duration_s"] < 25 * 60),
        ("games past 30 min", lambda c, r: r["duration_s"] >= 30 * 60),
    ]
    return out


def tercile_levels(vectors: Sequence[Sequence[float]]) -> dict:
    """Bottom/top tercile cut points per trait from the training rows."""
    out = {}
    if not vectors:
        return out
    for i, t in enumerate(TRAITS):
        vals = sorted(v[i] for v in vectors)
        n = len(vals)
        out[t] = (vals[max(0, n // 3 - 1)], vals[min(n - 1, (2 * n) // 3)])
    return out
