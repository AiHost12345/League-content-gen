"""Data Dragon metadata: items, champions and runes.

Loads from a local cache, then the Data Dragon CDN, and falls back to the
snapshot shipped with the package so tests and the desktop app work offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from importlib import resources
from pathlib import Path
from typing import Iterable

DDRAGON = "https://ddragon.leagueoflegends.com"
SUMMONERS_RIFT = "11"

# Stat shards are not in runesReforged.json. Rows: offense, flex, defense.
SHARD_ROWS = (
    {5008: "Adaptive Force", 5005: "Attack Speed", 5007: "Ability Haste"},
    {5008: "Adaptive Force", 5010: "Move Speed", 5001: "Health Scaling"},
    {5011: "Health", 5013: "Tenacity and Slow Resist", 5001: "Health Scaling"},
)
SHARD_NAMES = {k: v for row in SHARD_ROWS for k, v in row.items()}

# Situational item categories shown in the item set.
CATEGORIES = ("anti-heal", "armor", "mr", "survive")


def _snapshot() -> dict:
    with resources.files("buildopt").joinpath("data/ddragon_snapshot.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _trim_items(raw: dict) -> dict:
    out = {}
    for k, v in raw["data"].items():
        if int(k) >= 100000 or not v.get("maps", {}).get(SUMMONERS_RIFT):
            continue
        desc = v.get("description", "")
        out[k] = {
            "name": v["name"],
            "from": v.get("from") or [],
            "into": v.get("into") or [],
            "tags": v.get("tags", []),
            "depth": v.get("depth"),
            "gold": {"total": v["gold"]["total"], "purchasable": v["gold"]["purchasable"]},
            "requiredAlly": v.get("requiredAlly"),
            "inStore": v.get("inStore"),
            "consumed": v.get("consumed", False),
            "specialRecipe": v.get("specialRecipe"),
            "grievous": bool(re.search(r"Grievous|Wounds", desc)),
        }
    return out


def _trim_champions(raw: dict) -> dict:
    out = {}
    for v in raw["data"].values():
        s = v["stats"]
        out[v["key"]] = {
            "id": v["id"],
            "name": v["name"],
            "tags": v["tags"],
            "info": v["info"],
            "partype": v["partype"],
            "stats": {k: s[k] for k in ("hp", "hpperlevel", "armor", "spellblock", "movespeed", "attackrange", "attackdamage")},
        }
    return out


def fetch(version: str | None = None, cache_dir: Path | None = None, session=None) -> dict:
    """Fetch (and cache) Data Dragon data. Returns the same shape as the snapshot."""
    import requests

    http = session or requests
    if version is None:
        version = http.get(f"{DDRAGON}/api/versions.json", timeout=20).json()[0]
    if cache_dir:
        cached = Path(cache_dir) / f"ddragon_{version}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
    base = f"{DDRAGON}/cdn/{version}/data/en_US"
    data = {
        "version": version,
        "items": _trim_items(http.get(f"{base}/item.json", timeout=30).json()),
        "champions": _trim_champions(http.get(f"{base}/champion.json", timeout=30).json()),
        "runes": http.get(f"{base}/runesReforged.json", timeout=30).json(),
    }
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / f"ddragon_{version}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


def load(version: str | None = None, cache_dir: Path | None = None, offline: bool = False) -> "StaticData":
    if cache_dir and version:
        p = Path(cache_dir) / f"ddragon_{version}.json"
        if p.exists():
            return StaticData(json.loads(p.read_text(encoding="utf-8")))
    if not offline:
        try:
            return StaticData(fetch(version, cache_dir))
        except Exception:  # network down: use the shipped snapshot
            pass
    return StaticData(_snapshot())


@dataclass
class StaticData:
    raw: dict
    _completed: dict = field(default_factory=dict, repr=False)

    @property
    def version(self) -> str:
        return self.raw["version"]

    # ---- items ---------------------------------------------------------
    @property
    def items(self) -> dict:
        return self.raw["items"]

    def item(self, item_id: int) -> dict | None:
        return self.items.get(str(item_id))

    def item_name(self, item_id: int) -> str:
        it = self.item(item_id)
        return it["name"] if it else f"Item {item_id}"

    def _into_real(self, it: dict) -> list[str]:
        # Ornn masterwork upgrades (requiredAlly) don't make an item a component.
        return [i for i in it.get("into", []) if (self.items.get(i) or {}).get("requiredAlly") is None]

    def is_boots(self, item_id: int) -> bool:
        """Tier-2 boots. Tier-3 upgrades are treated as the same boots slot."""
        it = self.item(item_id)
        return bool(it and "Boots" in it["tags"] and (it.get("depth") or 1) == 2)

    def is_legendary(self, item_id: int) -> bool:
        it = self.item(item_id)
        if not it or "Boots" in it["tags"] or it.get("consumed") or "Consumable" in it["tags"]:
            return False
        if "Trinket" in it["tags"] or it.get("requiredAlly"):
            return False
        if self._into_real(it):
            return False
        if not it["gold"]["purchasable"] and not it.get("specialRecipe"):
            return False
        return it["gold"]["total"] >= 2000 or (it.get("depth") or 0) >= 3

    def is_completed(self, item_id: int) -> bool:
        if item_id not in self._completed:
            self._completed[item_id] = self.is_legendary(item_id) or self.is_boots(item_id)
        return self._completed[item_id]

    def item_categories(self, item_id: int) -> list[str]:
        it = self.item(item_id)
        if not it or not self.is_legendary(item_id):
            return []
        tags = set(it["tags"])
        cats = []
        if it.get("grievous"):
            cats.append("anti-heal")
        if "Armor" in tags:
            cats.append("armor")
        if "SpellBlock" in tags:
            cats.append("mr")
        if "Health" in tags and ({"LifeSteal", "SpellVamp", "Armor", "SpellBlock"} & tags or "HealthRegen" in tags):
            cats.append("survive")
        return cats

    # ---- champions -----------------------------------------------------
    @property
    def champions(self) -> dict:
        return self.raw["champions"]

    def champion(self, champ_id: int) -> dict | None:
        return self.champions.get(str(champ_id))

    def champion_name(self, champ_id: int) -> str:
        c = self.champion(champ_id)
        return c["name"] if c else f"Champion {champ_id}"

    @cached_property
    def _champ_by_name(self) -> dict:
        out = {}
        for key, c in self.champions.items():
            out[c["name"].lower()] = int(key)
            out[c["id"].lower()] = int(key)
            out[re.sub(r"[^a-z]", "", c["name"].lower())] = int(key)
        return out

    def champion_id(self, name_or_id: str | int) -> int:
        if isinstance(name_or_id, int) or str(name_or_id).isdigit():
            return int(name_or_id)
        key = str(name_or_id).lower()
        if key in self._champ_by_name:
            return self._champ_by_name[key]
        key = re.sub(r"[^a-z]", "", key)
        if key in self._champ_by_name:
            return self._champ_by_name[key]
        raise KeyError(f"Unknown champion: {name_or_id}")

    @cached_property
    def _item_by_name(self) -> dict:
        out = {}
        for k, it in self.items.items():
            if self.is_completed(int(k)):
                out.setdefault(re.sub(r"[^a-z]", "", it["name"].lower()), int(k))
        return out

    def item_id(self, name_or_id: str | int) -> int:
        if isinstance(name_or_id, int) or str(name_or_id).isdigit():
            return int(name_or_id)
        key = re.sub(r"[^a-z]", "", str(name_or_id).lower())
        if key in self._item_by_name:
            return self._item_by_name[key]
        matches = [v for k, v in self._item_by_name.items() if k.startswith(key)]
        if len(matches) == 1:
            return matches[0]
        aliases = {"bc": "blackcleaver", "dd": "deathsdance", "titanic": "titanichydra", "collector": "thecollector",
                   "ga": "guardianangel", "steraks": "sterakgage", "ie": "infinityedge", "bork": "bladeoftheruinedking"}
        if key in aliases:
            return self.item_id(aliases[key])
        raise KeyError(f"Unknown or ambiguous item: {name_or_id}")

    # ---- runes ---------------------------------------------------------
    @cached_property
    def perk_style(self) -> dict[int, int]:
        out = {}
        for style in self.raw["runes"]:
            for slot in style["slots"]:
                for r in slot["runes"]:
                    out[r["id"]] = style["id"]
        return out

    @cached_property
    def keystones(self) -> set[int]:
        return {r["id"] for style in self.raw["runes"] for r in style["slots"][0]["runes"]}

    @cached_property
    def perk_slot(self) -> dict[int, int]:
        return {r["id"]: i for style in self.raw["runes"] for i, slot in enumerate(style["slots"]) for r in slot["runes"]}

    @cached_property
    def perk_names(self) -> dict[int, str]:
        out = {r["id"]: r["name"] for style in self.raw["runes"] for slot in style["slots"] for r in slot["runes"]}
        out.update(SHARD_NAMES)
        return out

    @cached_property
    def style_names(self) -> dict[int, str]:
        return {s["id"]: s["name"] for s in self.raw["runes"]}

    def perk_name(self, perk_id: int) -> str:
        return self.perk_names.get(perk_id, str(perk_id))

    def completed_items(self, ids: Iterable[int]) -> list[int]:
        return [i for i in ids if i and self.is_completed(i)]
