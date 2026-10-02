"""Read the champ select session and assign roles to enemy picks.

Only what the champ select screen already shows is used: champions and your
own assigned position. Enemy summoner names, ranks and histories are never
read.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from buildopt import ROLES

POSITION_TO_ROLE = {"top": "TOP", "jungle": "JUNGLE", "middle": "MIDDLE", "bottom": "BOTTOM", "utility": "UTILITY"}


@dataclass
class ChampSelectState:
    phase: str  # PLANNING, BAN_PICK, FINALIZATION, GAME_STARTING
    my_champion: int  # locked or hovered, 0 if none
    my_locked: bool
    my_role: str | None
    allies: dict[str, int] = field(default_factory=dict)  # role -> champion (where known)
    enemies: list[int] = field(default_factory=list)  # locked enemy champions in pick order
    time_left_ms: int = 0

    @property
    def enemy_key(self) -> tuple[int, ...]:
        return tuple(sorted(self.enemies))


def parse_session(session: dict) -> ChampSelectState:
    local = session.get("localPlayerCellId")
    me = next((p for p in session.get("myTeam", []) if p.get("cellId") == local), {})
    actions = [a for group in session.get("actions", []) for a in group]
    my_pick = [a for a in actions if a.get("actorCellId") == local and a.get("type") == "pick"]
    locked = any(a.get("completed") for a in my_pick)
    champ = me.get("championId") or me.get("championPickIntent") or 0
    if not champ:
        champ = next((a.get("championId", 0) for a in my_pick if a.get("championId")), 0)
    allies = {}
    for p in session.get("myTeam", []):
        role = POSITION_TO_ROLE.get((p.get("assignedPosition") or "").lower())
        if role and p.get("championId"):
            allies[role] = p["championId"]
    enemies = [p["championId"] for p in session.get("theirTeam", []) if p.get("championId")]
    timer = session.get("timer", {})
    return ChampSelectState(
        phase=timer.get("phase", ""),
        my_champion=int(champ),
        my_locked=locked,
        my_role=POSITION_TO_ROLE.get((me.get("assignedPosition") or "").lower()),
        allies=allies,
        enemies=enemies,
        time_left_ms=int(timer.get("adjustedTimeLeftInPhase", 0) or 0),
    )


def assign_roles(enemies: list[int], profiles: dict) -> dict[int, str]:
    """Most likely role for each enemy pick, from role play-rates, as an assignment problem.

    With at most five champions and five roles, exhaustive search over the
    permutations (<=120) is exact and fast.
    """
    if not enemies:
        return {}
    enemies = enemies[:5]

    def rate(cid: int, role: str) -> float:
        prof = profiles.get(str(cid)) or profiles.get(cid) or {}
        return max(prof.get("role_rates", {}).get(role, 0.2), 1e-4)

    best, best_score = None, -math.inf
    for roles in itertools.permutations(ROLES, len(enemies)):
        score = sum(math.log(rate(c, r)) for c, r in zip(enemies, roles))
        if score > best_score:
            best, best_score = roles, score
    return dict(zip(enemies, best))
