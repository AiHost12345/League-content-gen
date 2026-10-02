"""Jungle planner v1 (modelled on YordleDiff's decision engine).

* First-clear routes are clustered from per-minute timeline positions
  (minutes 2-4), normalised to the blue side so both teams share labels, then
  ranked by win rate against the enemy jungler.
* Gank priority by lane combines ally CC with enemy laners' escape tools.

v2 (a per-minute win-probability model to score drake vs gank vs farm) is not
implemented yet.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from buildopt.stats import weighted_win_rate

MAP_SIZE = 14800
ZONES = {  # anchor points on the blue side's view of the map
    "own blue side": [(3800, 7900), (2100, 8400), (3800, 6500)],
    "own red side": [(7000, 5400), (7800, 4000), (8400, 2700)],
    "enemy blue side": [(11000, 6900), (12700, 6400), (11000, 8300)],
    "enemy red side": [(7800, 9400), (7000, 10800), (6400, 12100)],
    "top river": [(4400, 9600), (5000, 10400)],
    "bot river": [(10500, 5100), (9800, 4400)],
    "base": [(500, 500), (1500, 1500)],
}
ROUTE_MINUTES = (2, 3, 4)  # indices into row["positions"], which start at minute 1


def zone(pos: Sequence[float] | None, team_id: int) -> str | None:
    if not pos:
        return None
    x, y = pos
    if team_id == 200:
        x, y = MAP_SIZE - x, MAP_SIZE - y
    best, dist = None, float("inf")
    for name, anchors in ZONES.items():
        for ax, ay in anchors:
            d = math.hypot(x - ax, y - ay)
            if d < dist:
                best, dist = name, d
    return best if dist < 3500 else "lane"


def route_of(row: dict) -> str | None:
    pos = row.get("positions") or []
    zones = []
    for m in ROUTE_MINUTES:
        if m - 1 < len(pos):
            z = zone(pos[m - 1], row["team_id"])
            if z and (not zones or zones[-1] != z):
                zones.append(z)
    return " → ".join(zones) if zones else None


def route_stats(games, min_games: int = 20) -> dict:
    """Route win rates overall and against each enemy jungler."""
    overall: dict[str, list] = defaultdict(list)
    vs: dict[tuple[str, int], list] = defaultdict(list)
    for g in games:
        route = route_of(g.row)
        if not route:
            continue
        overall[route].append(g)
        jg = next((e["champion_id"] for e in g.enemies if e["role"] == "JUNGLE"), None)
        if jg:
            vs[(route, jg)].append(g)

    def summ(gs):
        return weighted_win_rate([g.win for g in gs], [g.weight for g in gs]).to_dict()

    return {
        "routes": {r: summ(gs) for r, gs in overall.items() if len(gs) >= min_games},
        "vs_jungler": {f"{r}|{c}": summ(gs) for (r, c), gs in vs.items() if len(gs) >= min_games},
    }


def best_routes(stats: dict, enemy_jungler: int | None, top: int = 3) -> list[dict]:
    rows = []
    if enemy_jungler:
        for key, s in stats.get("vs_jungler", {}).items():
            route, champ = key.rsplit("|", 1)
            if int(champ) == enemy_jungler:
                rows.append({"route": route, "vs": enemy_jungler, **s})
    if not rows:
        rows = [{"route": r, "vs": None, **s} for r, s in stats.get("routes", {}).items()]
    rows.sort(key=lambda d: -d["lo"])
    return rows[:top]


def gank_priority(allies: dict[str, int], enemies: dict[str, int], profiles: dict) -> list[dict]:
    """Rank lanes for ganks: ally CC to lock targets down, enemy laners with weak escapes."""

    def prof(cid):
        return profiles.get(str(cid)) or profiles.get(cid) or {"cc": 0.4, "mobility": 0.4, "tank": 0.3}

    lanes = {"TOP": ["TOP"], "MIDDLE": ["MIDDLE"], "BOTTOM": ["BOTTOM", "UTILITY"]}
    out = []
    for lane, roles in lanes.items():
        ally_cc = sum(prof(allies[r])["cc"] for r in roles if r in allies)
        enemy = [prof(enemies[r]) for r in roles if r in enemies]
        if not enemy:
            continue
        escape = sum(p.get("mobility", 0.4) for p in enemy) / len(enemy)
        squishy = sum(1 - p["tank"] for p in enemy) / len(enemy)
        score = ally_cc / len(roles) + (1 - escape) + 0.5 * squishy
        out.append({"lane": lane, "score": round(score, 3), "ally_cc": round(ally_cc, 2), "enemy_escape": round(escape, 2)})
    out.sort(key=lambda d: -d["score"])
    return out
