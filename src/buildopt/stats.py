"""Small statistics helpers shared by the pipeline and the desktop app."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

Z95 = 1.959963984540054

# Display thresholds from the design: hide a path below 50 games in a
# condition, label 50-200 as low confidence.
MIN_GAMES = 50
LOW_CONFIDENCE_GAMES = 200


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def expit(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def effective_n(weights: Iterable[float]) -> float:
    """Kish effective sample size for weighted observations."""
    s = s2 = 0.0
    for w in weights:
        s += w
        s2 += w * w
    return (s * s / s2) if s2 > 0 else 0.0


def wilson_interval(p: float, n: float, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion ``p`` observed over ``n`` trials.

    ``n`` may be fractional (an effective sample size), which lets the same
    interval be applied to weighted or model-adjusted win rates.
    """
    if n <= 0:
        return 0.0, 1.0
    p = min(max(p, 0.0), 1.0)
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_lower(p: float, n: float, z: float = Z95) -> float:
    return wilson_interval(p, n, z)[0]


def confidence_label(games: float, lo: float, hi: float) -> str:
    """High / medium / low from sample size and interval width."""
    if games < LOW_CONFIDENCE_GAMES:
        return "low"
    width = hi - lo
    if width <= 0.08:
        return "high"
    if width <= 0.14:
        return "medium"
    return "low"


@dataclass
class WinRate:
    wins: float
    games: float  # sum of weights
    n_eff: float  # effective sample size
    raw_games: int

    @property
    def rate(self) -> float:
        return self.wins / self.games if self.games > 0 else float("nan")

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.rate, self.n_eff) if self.games > 0 else (0.0, 1.0)

    @property
    def lower(self) -> float:
        return self.interval[0]

    @property
    def confidence(self) -> str:
        lo, hi = self.interval
        return confidence_label(self.raw_games, lo, hi)

    def to_dict(self) -> dict:
        lo, hi = self.interval
        return {
            "win_rate": self.rate,
            "lo": lo,
            "hi": hi,
            "games": self.raw_games,
            "n_eff": self.n_eff,
            "confidence": self.confidence,
        }


def weighted_win_rate(wins: Sequence[float], weights: Sequence[float] | None = None) -> WinRate:
    if weights is None:
        weights = [1.0] * len(wins)
    tot = sum(weights)
    won = sum(w * y for w, y in zip(weights, wins))
    return WinRate(wins=won, games=tot, n_eff=effective_n(weights), raw_games=len(wins))


def smoothed_logit_rate(wins: float, games: float, prior: float, strength: float = 20.0) -> float:
    """Logit of a win rate shrunk toward ``prior`` with ``strength`` pseudo-games."""
    return logit((wins + prior * strength) / (games + strength))


def pair_lift(wr_ab: float, wr_a: float, wr_b: float, wr_champ: float) -> float:
    """Pair synergy on the log-odds scale (design doc formula).

    lift(A,B) = logit WR(A and B) - logit WR(A) - logit WR(B) + logit WR(champ)
    Positive: the items complement each other. Negative: redundant.
    """
    return logit(wr_ab) - logit(wr_a) - logit(wr_b) + logit(wr_champ)
