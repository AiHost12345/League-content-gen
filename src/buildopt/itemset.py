"""Turn a recommendation into a League client item set (in-game shop's recommended tab)."""

from __future__ import annotations

import uuid
from typing import Sequence

from buildopt.ddragon import CATEGORIES
from buildopt.scoring import Recommendation, Scorer

SUMMONERS_RIFT_MAP_ID = 11
POTION = 2003


def _block(title: str, items: Sequence[int]) -> dict:
    return {"type": title, "items": [{"id": str(i), "count": 1} for i in items], "hideIfSummonerSpell": "", "showIfSummonerSpell": ""}


def core_with_boots(path: Sequence[int], boots: int | None, slot: int) -> list[int]:
    items = list(path)
    if boots:
        items.insert(max(0, min(slot, len(items))), boots)
    return items


def build_blocks(scorer: Scorer, rec: Recommendation) -> list[dict]:
    c = rec.traits
    best = rec.best
    boots = scorer.boots_pick(c)
    core = core_with_boots(best.path, boots, scorer.boots_slot(best.path))
    names = " → ".join(str(i + 1) for i in range(len(best.path)))
    start = list(scorer.bundle.get("start_items") or [])
    if POTION not in start and start:
        start.append(POTION)
    blocks = [_block("Start", start), _block(f"Core: item {names}", core)]
    used = set(core)
    for cat in CATEGORIES:
        item = scorer.situational_pick(cat, c, exclude=used)
        if not item:
            continue
        reason = scorer.situational_reason(cat, c)
        label = {"anti-heal": "anti-heal", "armor": "armor", "mr": "MR", "survive": "survive"}[cat]
        blocks.append(_block(f"Situational: {label}" + (f" ({reason})" if reason else ""), [item]))
        used.add(item)
    late = scorer.late_game(used)
    if late:
        blocks.append(_block("Late game", late))
    if rec.alt_path:
        blocks.append(_block(f"Alt path ({scorer.perk(rec.alt_path.keystone)})" if rec.alt_path.keystone != best.keystone
                             else "Alt path", core_with_boots(rec.alt_path.path, boots, scorer.boots_slot(rec.alt_path.path))))
    return blocks


def item_set_title(scorer: Scorer, prefix: str) -> str:
    dv = scorer.bundle["data_version"]
    return f"{prefix} {scorer.bundle['champion_name']} {dv['patch']} #{dv['id'][-6:]}"


def build_item_set(scorer: Scorer, rec: Recommendation, prefix: str = "BO:") -> dict:
    return {
        "title": item_set_title(scorer, prefix),
        "associatedChampions": [scorer.bundle["champion_id"]],
        "associatedMaps": [SUMMONERS_RIFT_MAP_ID],
        "blocks": build_blocks(scorer, rec),
        "map": "SR",
        "mode": "any",
        "preferredItemSlots": [],
        "sortrank": 0,
        "startedFrom": "blank",
        "type": "custom",
        "uid": str(uuid.uuid4()),
    }
