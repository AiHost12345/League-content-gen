"""Stats bundle: everything the desktop app needs to score loadouts offline.

One bundle per champion and role per patch. Every bundle carries a single
data version that is stamped on every output (CLI, window, item set title).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from buildopt.analysis import jungle, wincon
from buildopt.analysis.candidates import candidate_pages, candidate_paths, model_paths, page_shards
from buildopt.analysis.dataset import DatasetConfig, Game, load_games
from buildopt.analysis.model import LoadoutModel, Penalties, build_model, tune_penalties
from buildopt.analysis.profiles import build_profiles
from buildopt.analysis.traits import TRAITS, tercile_levels, trait_level
from buildopt.ddragon import CATEGORIES, StaticData
from buildopt.pipeline.store import Store
from buildopt.stats import MIN_GAMES, weighted_win_rate

SCHEMA_VERSION = 1

# Which comp trait makes each situational category relevant, and in which direction.
CATEGORY_TRAIT = {"anti-tank": ("tanks", "high"), "anti-heal": ("healing", "high"), "armor": ("ap_share", "low"), "mr": ("ap_share", "high"),
                  "survive": ("burst", "high")}


def _item_stats(entries: list[tuple[int, Game]]) -> dict:
    acc: dict[int, list] = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for item, g in entries:
        a = acc[item]
        a[0] += g.weight * g.win
        a[1] += g.weight
        a[2] += g.weight * g.weight
        a[3] += 1
    return {str(k): {"wins": v[0], "w": v[1], "w2": v[2], "n": v[3]} for k, v in acc.items()}


def situational_stats(games: list[Game], static: StaticData, levels: dict) -> dict:
    """Per category: item win stats from slot 3 onward, split by the level of the relevant trait."""
    out = {}
    tidx = {t: i for i, t in enumerate(TRAITS)}
    for cat in CATEGORIES:
        trait, _ = CATEGORY_TRAIT[cat]
        by_level: dict[str, list] = defaultdict(list)
        for g in games:
            level = trait_level(trait, g.c_raw[tidx[trait]], levels)
            for item in g.path[2:6]:
                if cat in static.item_categories(item):
                    by_level[level].append((item, g))
                    by_level["all"].append((item, g))
        out[cat] = {lvl: _item_stats(entries) for lvl, entries in by_level.items()}
    return out


def late_items(games: list[Game], min_games: int) -> list[dict]:
    """Items bought 4th-6th, ranked by Wilson lower bound and shown relative to all games reaching item 4."""
    reach = [g for g in games if len(g.path) >= 4]
    if not reach:
        return []
    base = weighted_win_rate([g.win for g in reach], [g.weight for g in reach]).rate
    per: dict[int, list[Game]] = defaultdict(list)
    for g in reach:
        for item in set(g.path[3:6]):
            per[item].append(g)
    out = []
    for item, gs in per.items():
        if len(gs) < min_games:
            continue
        wr = weighted_win_rate([g.win for g in gs], [g.weight for g in gs])
        out.append({"id": item, **wr.to_dict(), "vs_slot_average": wr.rate - base})
    out.sort(key=lambda d: -d["lo"])
    return out


def first_item_stats(games: list[Game], min_games: int = 20) -> list[dict]:
    """Every first item players finished, with win rates split by how many tanks the enemy had."""
    tidx = TRAITS.index("tanks")
    per: dict[int, list[Game]] = defaultdict(list)
    for g in games:
        per[g.path[0]].append(g)
    out = []
    for item, gs in per.items():
        if len(gs) < min_games:
            continue
        few = [g for g in gs if round(g.c_raw[tidx]) <= 1]
        many = [g for g in gs if round(g.c_raw[tidx]) >= 2]

        def wr(sel):
            return weighted_win_rate([g.win for g in sel], [g.weight for g in sel]).to_dict() if sel else None

        out.append({"id": item, **wr(gs), "vs_0_1_tanks": wr(few), "vs_2plus_tanks": wr(many)})
    out.sort(key=lambda d: -d["games"])
    return out


def boots_stats(games: list[Game], levels: dict) -> dict:
    tidx = TRAITS.index("ap_share")
    by_level: dict[str, list] = defaultdict(list)
    for g in games:
        if g.boots:
            lvl = trait_level("ap_share", g.c_raw[tidx], levels)
            by_level[lvl].append((g.boots, g))
            by_level["all"].append((g.boots, g))
    return {lvl: _item_stats(e) for lvl, e in by_level.items()}


def path_info(games: list[Game], paths: list[tuple]) -> list[dict]:
    by_path: dict[tuple, list[Game]] = defaultdict(list)
    for g in games:
        by_path[g.path[:3]].append(g)
    out = []
    for p in paths:
        gs = by_path.get(tuple(p), [])
        slots = Counter(g.boots_slot for g in gs if g.boots_slot is not None)
        wr = weighted_win_rate([g.win for g in gs], [g.weight for g in gs])
        out.append({"path": list(p), "games": len(gs), "raw_win_rate": wr.rate,
                    "boots_slot": slots.most_common(1)[0][0] if slots else 1})
    return out


def build_bundle(store: Store, static: StaticData, champion_id: int, role: str, cfg: DatasetConfig | None = None,
                 min_games: int = MIN_GAMES, penalties: Penalties | None = None, profiles: dict | None = None) -> dict:
    cfg = cfg or DatasetConfig()
    patches = store.patches()
    if profiles is None:
        recent = patches[-2:] if patches else []
        profiles = build_profiles(store.iter_rows(patches=recent), static)
    games, info = load_games(store, champion_id, role, profiles, cfg)
    if not games:
        raise ValueError(f"No games for champion {champion_id} {role}.")
    # Every build players finished is scored: common ones get their own model term, rarer ones are
    # fitted through their group and then scored individually (rare_paths).
    paths, rare = model_paths(games)
    pages = candidate_pages(games, min_games)
    if not paths or not pages:
        raise ValueError(f"Not enough games for any path/page above {min_games} games "
                         f"({len(games)} games for {static.champion_name(champion_id)} {role}).")
    tuning = {}
    if penalties is None:
        penalties, tuning = tune_penalties(games, paths, pages)
    model = build_model(games, paths, pages).fit(games, penalties)
    model.meta["penalty_scales"] = tuning.get("scales")
    used = model.usable(games)
    rare_paths = model.rare_effects(used, rare)
    levels = tercile_levels([g.c_raw for g in used])

    page_counts = Counter(g.page for g in used)
    path_counts = Counter(g.path[:3] for g in used)
    popular = {"path": model.path_index(path_counts.most_common(1)[0][0]), "page": pages.index(page_counts.most_common(1)[0][0])}
    start = Counter(tuple(sorted(g.row["start_items"])) for g in games if g.row.get("start_items"))
    shards = page_shards(games, pages)

    situational = situational_stats(games, static, levels)
    late = late_items(games, min_games)
    referenced = {i for p in candidate_paths(used) for i in p} | {int(k) for c in situational.values() for lv in c.values() for k in lv}
    first_items = first_item_stats(games)
    referenced |= {d["id"] for d in late} | {i for s in start for i in s} | {d["id"] for d in first_items}
    boots = boots_stats(games, levels)
    referenced |= {int(k) for lv in boots.values() for k in lv}

    generated = dt.datetime.now(dt.timezone.utc)
    bundle = {
        "schema": SCHEMA_VERSION,
        "champion_id": champion_id,
        "champion_name": static.champion_name(champion_id),
        "role": role,
        "data_version": {
            **info,
            "ddragon": static.version,
            "generated": generated.isoformat(timespec="seconds"),
            "games_model": model.meta["train_games"],
        },
        "model": model.to_dict(),
        "popular": popular,
        "page_shards": [list(shards.get(p, (5008, 5008, 5011))) for p in pages],
        "path_info": path_info(used, candidate_paths(used)),
        "rare_paths": rare_paths,
        "start_items": list(start.most_common(1)[0][0]) if start else [],
        "boots": boots,
        "situational": situational,
        "late_items": late[:10],
        "first_items": first_items,
        "levels": {k: list(v) for k, v in levels.items()},
        "profiles": {str(k): v for k, v in profiles.items()},
        "item_names": {str(i): static.item_name(i) for i in sorted(referenced)},
        "perk_names": {str(k): v for k, v in static.perk_names.items()},
        "style_names": {str(k): v for k, v in static.style_names.items()},
        "champion_names": {k: c["name"] for k, c in static.champions.items()},
    }
    if role == "JUNGLE":
        bundle["jungle"] = jungle.route_stats(games, min_games=max(10, min_games // 5))
    bundle["wincon"] = wincon.signals_from_profiles(profiles, static)
    digest = hashlib.sha256(json.dumps(bundle["model"]["beta"]).encode()).hexdigest()[:6]
    bundle["data_version"]["id"] = f"{info['patch']}-{generated:%Y%m%d}-{digest}"
    return bundle


def data_version_label(bundle: dict) -> str:
    dv = bundle["data_version"]
    prev = f" + {dv['previous_patch']}×{dv['previous_weight']:g}" if dv.get("previous_patch") else ""
    return f"data {dv['id']} · patch {dv['patch']}{prev} · {dv['games_used']:,} games"


def bundle_filename(champion_id: int, role: str) -> str:
    return f"{champion_id}_{role}.json"


def save_bundle(bundle: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / bundle_filename(bundle["champion_id"], bundle["role"])
    text = json.dumps(bundle, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"bundles": {}}
    index["bundles"][f"{bundle['champion_id']}_{bundle['role']}"] = {
        "file": path.name,
        "champion_id": bundle["champion_id"],
        "champion_name": bundle["champion_name"],
        "role": bundle["role"],
        "data_version": bundle["data_version"]["id"],
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    index["updated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return path


def load_bundle(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_of(bundle: dict) -> LoadoutModel:
    return LoadoutModel.from_dict(bundle["model"])
