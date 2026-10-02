"""Reduce a match-v5 match + timeline to one row per player.

The completed-item sequence holds only finished legendaries and tier-2 boots,
with undos removed and quick sell-backs dropped, and records the gold state at
the moment each item was completed.
"""

from __future__ import annotations

from bisect import bisect_right

from buildopt import ROLES
from buildopt.ddragon import StaticData

START_WINDOW_MS = 75_000  # purchases before this are starting items
FLIP_WINDOW_MS = 180_000  # completed items sold back within 3 min are discarded
POSITION_MINUTES = (1, 2, 3, 4, 5, 6)


def patch_of(game_version: str) -> str:
    parts = game_version.split(".")
    return ".".join(parts[:2])


def is_valid_game(info: dict) -> bool:
    if info.get("queueId") != 420:
        return False
    if info.get("gameDuration", 0) < 300:  # remakes
        return False
    return not any(p.get("gameEndedInEarlySurrender") for p in info.get("participants", []))


def _rune_page(perks: dict) -> dict:
    styles = {s["description"]: s for s in perks.get("styles", [])}
    prim = styles.get("primaryStyle", {})
    sub = styles.get("subStyle", {})
    prim_sel = [s["perk"] for s in prim.get("selections", [])]
    sub_sel = [s["perk"] for s in sub.get("selections", [])]
    sp = perks.get("statPerks", {})
    return {
        "primary_style": prim.get("style", 0),
        "sub_style": sub.get("style", 0),
        "keystone": prim_sel[0] if prim_sel else 0,
        "perks": prim_sel + sub_sel,
        "shards": [sp.get("offense", 0), sp.get("flex", 0), sp.get("defense", 0)],
    }


class GoldSeries:
    """Per-minute total gold for every participant, interpolated in time."""

    def __init__(self, frames: list[dict]):
        self.ts = [f["timestamp"] for f in frames]
        self.gold = [{int(k): v.get("totalGold", 0) for k, v in f["participantFrames"].items()} for f in frames]

    def at(self, pid: int, t: float) -> float:
        if not self.ts:
            return 0.0
        i = bisect_right(self.ts, t) - 1
        if i < 0:
            return float(self.gold[0].get(pid, 0))
        if i >= len(self.ts) - 1:
            return float(self.gold[-1].get(pid, 0))
        t0, t1 = self.ts[i], self.ts[i + 1]
        g0, g1 = self.gold[i].get(pid, 0), self.gold[i + 1].get(pid, 0)
        return g0 + (g1 - g0) * (t - t0) / max(1, t1 - t0)


def resolve_items(events: list[dict], pid: int, static: StaticData) -> tuple[list[dict], list[int]]:
    """Return (completed item sequence, starting items) for one participant."""
    completed: list[dict] = []
    start: list[int] = []
    evs = sorted((e for e in events if e.get("participantId") == pid and e["type"].startswith("ITEM_")),
                 key=lambda e: e["timestamp"])
    for e in evs:
        t = e["timestamp"]
        typ = e["type"]
        if typ == "ITEM_PURCHASED":
            item = e["itemId"]
            if t < START_WINDOW_MS and not completed:
                start.append(item)
            if static.is_completed(item):
                if static.is_boots(item) and any(static.is_boots(c["id"]) for c in completed if c.get("sold") is None):
                    continue  # second pair of boots after a sell; keep first slot
                completed.append({"id": item, "ms": t})
        elif typ == "ITEM_UNDO":
            before, after = e.get("beforeId", 0), e.get("afterId", 0)
            if before and not after:  # undo a purchase
                if t < START_WINDOW_MS + 30_000 and before in start:
                    start.remove(before)
                for i in range(len(completed) - 1, -1, -1):
                    if completed[i]["id"] == before and completed[i].get("sold") is None:
                        del completed[i]
                        break
            elif after and not before:  # undo a sell
                for c in reversed(completed):
                    if c["id"] == after and c.get("sold") is not None:
                        c["sold"] = None
                        break
        elif typ == "ITEM_SOLD":
            item = e["itemId"]
            for c in reversed(completed):
                if c["id"] == item and c.get("sold") is None:
                    c["sold"] = t
                    break
    seq = [c for c in completed if c.get("sold") is None or c["sold"] - c["ms"] > FLIP_WINDOW_MS]
    return seq, start


def parse_match(match: dict, timeline: dict | None, static: StaticData, tier: str, platform: str) -> list[dict]:
    info = match["info"]
    match_id = match["metadata"]["matchId"]
    patch = patch_of(info["gameVersion"])
    duration = info["gameDuration"]
    if duration > 100_000:  # very old matches reported milliseconds
        duration //= 1000
    parts = info["participants"]
    by_pid = {p["participantId"]: p for p in parts}
    role_of = {p["participantId"]: (p.get("teamPosition") or "") for p in parts}

    gold = events = frames = None
    if timeline:
        frames = timeline["info"]["frames"]
        gold = GoldSeries(frames)
        events = [e for f in frames for e in f.get("events", [])]

    rows = []
    for p in parts:
        pid = p["participantId"]
        team = p["teamId"]
        role = role_of[pid]
        if role not in ROLES:
            continue
        allies = [q for q in parts if q["teamId"] == team and q["participantId"] != pid]
        enemies = [q for q in parts if q["teamId"] != team]
        opp = next((q for q in enemies if role_of[q["participantId"]] == role), None)
        page = _rune_page(p.get("perks", {}))
        row = {
            "match_id": match_id,
            "platform": platform,
            "patch": patch,
            "game_start": info.get("gameStartTimestamp", 0),
            "duration_s": duration,
            "tier": tier,
            "puuid": p.get("puuid", ""),
            "participant_id": pid,
            "team_id": team,
            "champion_id": p["championId"],
            "role": role,
            "win": bool(p["win"]),
            "allies": [{"champion_id": q["championId"], "role": role_of[q["participantId"]]} for q in allies],
            "enemies": [{"champion_id": q["championId"], "role": role_of[q["participantId"]]} for q in enemies],
            **page,
            "final_items": [p.get(f"item{i}", 0) for i in range(7)],
            "stats": {
                "dmg_total": p.get("totalDamageDealtToChampions", 0),
                "dmg_magic": p.get("magicDamageDealtToChampions", 0),
                "dmg_physical": p.get("physicalDamageDealtToChampions", 0),
                "dmg_true": p.get("trueDamageDealtToChampions", 0),
                "heal": p.get("totalHeal", 0),
                "cc_time": p.get("timeCCingOthers", 0),
                "dmg_taken": p.get("totalDamageTaken", 0),
                "dmg_mitigated": p.get("damageSelfMitigated", 0),
                "kills": p.get("kills", 0),
                "deaths": p.get("deaths", 0),
                "assists": p.get("assists", 0),
            },
            "has_timeline": timeline is not None,
            "items": [],
            "start_items": [],
            "positions": [],
        }
        if timeline is not None:
            seq, start = resolve_items(events, pid, static)
            team_pids = [q["participantId"] for q in parts if q["teamId"] == team]
            enemy_pids = [q["participantId"] for q in enemies]
            for c in seq:
                t = c["ms"]
                lane_gd = gold.at(pid, t) - gold.at(opp["participantId"], t) if opp else 0.0
                team_gd = sum(gold.at(q, t) for q in team_pids) - sum(gold.at(q, t) for q in enemy_pids)
                row["items"].append({
                    "id": c["id"],
                    "min": round(t / 60000, 2),
                    "lane_gd": round(lane_gd),
                    "team_gd": round(team_gd),
                    "boots": static.is_boots(c["id"]),
                })
            row["start_items"] = start
            for m in POSITION_MINUTES:
                if m < len(frames):
                    pos = frames[m]["participantFrames"].get(str(pid), {}).get("position")
                    row["positions"].append([pos["x"], pos["y"]] if pos else None)
        rows.append(row)
    if len(rows) != len(parts):
        # Missing positions (e.g. unranked swaps) make lane matchups unreliable.
        return []
    return rows


def needs_timeline(match: dict, targets: set[tuple[int, str]] | None) -> bool:
    if not targets:
        return True
    for p in match["info"]["participants"]:
        if (p["championId"], p.get("teamPosition", "")) in targets or (p["championId"], "") in targets:
            return True
    return False
