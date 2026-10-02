"""Synthetic ranked games with known effects, for demos and tests.

Produces rows in exactly the shape the parser emits, so the whole pipeline
(bundle, comparisons, scoring, item sets, companion) can run without a Riot API
key. The planted ground truth mirrors the design doc's Briar question:

* Collector -> BC wins with 0-1 enemy tanks; Titanic -> BC pulls ahead at 2+.
* Players buy Collector more often when already ahead (selection bias), so the
  raw Collector win rate is inflated.
* Chempunk third is better into heavy healing; Lethal Tempo pairs with
  Collector; Conqueror pairs with Titanic; Cut Down gains vs tanks.
* A per-champion matchup term: Briar does worse into an enemy Lee Sin.
"""

from __future__ import annotations

import math
import random

from buildopt.analysis.profiles import prior_profile
from buildopt.analysis.traits import trait_vector
from buildopt.ddragon import StaticData
from buildopt.pipeline.store import Store

POOLS = {
    "TOP": ["Darius", "Garen", "Ornn", "Malphite", "Sion", "K'Sante", "Fiora", "Jax", "Aatrox", "Camille", "Renekton",
            "Teemo", "Gnar", "Mordekaiser", "Sett"],
    "JUNGLE": ["Lee Sin", "Vi", "Sejuani", "Zac", "Kha'Zix", "Viego", "Amumu", "Elise", "Graves", "Kindred", "Nidalee",
               "Warwick", "Jarvan IV", "Diana", "Briar"],
    "MIDDLE": ["Ahri", "Syndra", "Orianna", "Zed", "Yasuo", "Viktor", "Vladimir", "Sylas", "Lux", "Xerath", "Galio",
               "Akali", "Katarina", "Hwei", "LeBlanc"],
    "BOTTOM": ["Jinx", "Caitlyn", "Kai'Sa", "Ezreal", "Jhin", "Ashe", "Varus", "Miss Fortune", "Samira", "Draven",
               "Sivir", "Xayah", "Lucian", "Aphelios", "Smolder"],
    "UTILITY": ["Thresh", "Leona", "Nautilus", "Lulu", "Soraka", "Nami", "Rakan", "Braum", "Milio", "Karma",
                "Blitzcrank", "Morgana", "Senna", "Zyra", "Rell"],
}
ROLES = list(POOLS)

COLLECTOR, BC, TITANIC, DD, STERAKS, ECLIPSE, CHEMPUNK, SV = 6676, 3071, 3748, 6333, 3053, 6692, 6609, 3065
STEELCAPS, MERCS = 3047, 3111
LATE_POOL = [3026, 3156, 6694, 3143, 4401, 3075, 3053, 3065, 6333, 6609, 3074, 6610]
PATHS = [  # (path, popularity)
    ((COLLECTOR, BC, DD), 1.2), ((TITANIC, BC, DD), 1.2), ((COLLECTOR, BC, STERAKS), 0.8),
    ((TITANIC, BC, STERAKS), 0.8), ((ECLIPSE, BC, DD), 0.6), ((TITANIC, BC, SV), 0.5),
    ((TITANIC, BC, CHEMPUNK), 0.5), ((COLLECTOR, BC, CHEMPUNK), 0.4),
]
ROUTES = {  # route name -> zone anchors at minutes 1..6 (blue-side coordinates)
    "red start": [(7800, 4000), (8400, 2700), (7000, 5400), (3800, 7900), (4400, 9600), (5000, 9000)],
    "blue start": [(3800, 7900), (2100, 8400), (3800, 6500), (7800, 4000), (10500, 5100), (9000, 4800)],
    "invade": [(11000, 6900), (12700, 6400), (7800, 4000), (8400, 2700), (10500, 5100), (9000, 4800)],
}


def _pages(static: StaticData) -> list[dict]:
    # primary style, sub style, perks (keystone, 3 minors, 2 secondary), popularity
    return [
        {"p": (8000, 8100, (8010, 9111, 9104, 8014, 8139, 8135)), "w": 1.6},  # Conqueror / Coup de Grace
        {"p": (8000, 8100, (8008, 9111, 9104, 8014, 8139, 8135)), "w": 1.0},  # Lethal Tempo
        {"p": (8000, 8100, (8010, 9111, 9104, 8017, 8139, 8135)), "w": 0.8},  # Cut Down swap
        {"p": (8000, 8400, (8010, 9111, 9104, 8014, 8473, 8242)), "w": 0.7},  # Resolve secondary
        {"p": (8100, 8000, (8112, 8139, 8140, 8135, 9111, 9104)), "w": 0.5},  # Electrocute
        {"p": (8100, 8000, (9923, 8139, 8140, 8135, 9111, 9104)), "w": 0.4},  # Hail of Blades
    ]


def _ids(static: StaticData) -> dict[str, list[int]]:
    return {r: [static.champion_id(n) for n in names] for r, names in POOLS.items()}


def _true_effect(path: tuple, page: tuple, c: list[float], lee_sin: bool, lane_gd: float, team_gd: float,
                 boots: int) -> float:
    tanks, ap, cc, heal, burst, ranged, poke = c
    z_tank, z_heal, z_ap, z_cc, z_burst = (tanks - 1.6) / 1.0, (heal - 2.0) / 0.5, (ap - 0.42) / 0.1, (cc - 2.6) / 0.4, (burst - 2.6) / 0.4
    eta = 0.00025 * lane_gd + 0.00010 * team_gd
    first, third = path[0], path[2]
    keystone = page[2][0]
    if first == COLLECTOR:
        eta += 0.30 - 0.25 * tanks
    elif first == TITANIC:
        eta += -0.08 + 0.05 * tanks
    else:
        eta -= 0.10
    eta += {CHEMPUNK: -0.05 + 0.20 * z_heal, STERAKS: 0.08 * z_burst, SV: -0.05 + 0.12 * z_ap}.get(third, 0.0)
    eta += {8010: 0.05, 8008: -0.03, 8112: -0.10, 9923: -0.08}.get(keystone, 0.0)
    if keystone == 8008 and first == COLLECTOR:
        eta += 0.15
    if keystone == 8010 and first == TITANIC:
        eta += 0.08
    perks = page[2]
    if 8017 in perks:
        eta += 0.08 * z_tank
    if 8242 in perks:
        eta += 0.08 * z_cc
    if boots == MERCS:
        eta += 0.10 * z_ap
    if lee_sin:
        eta -= 0.20
    return eta


def _stats(prof: dict, minutes: float, rnd: random.Random) -> dict:
    total = rnd.gauss(22000, 5000) * minutes / 30
    ap = min(1, max(0, rnd.gauss(prof["ap_share"], 0.06)))
    return {
        "dmg_total": int(total), "dmg_magic": int(total * ap), "dmg_physical": int(total * (1 - ap) * 0.9),
        "dmg_true": int(total * (1 - ap) * 0.1), "heal": int(max(0, rnd.gauss(prof["heal"] * 450, 60)) * minutes),
        "cc_time": int(max(0, rnd.gauss(prof["cc"] * 2.0, 0.3)) * minutes),
        "dmg_taken": int(max(0, rnd.gauss(prof["tank"] * 1600 + 500, 150)) * minutes),
        "dmg_mitigated": int(max(0, rnd.gauss(prof["tank"] * 1400 + 200, 150)) * minutes),
        "kills": rnd.randint(0, 12), "deaths": rnd.randint(0, 10), "assists": rnd.randint(0, 15),
    }


def generate(store: Store, static: StaticData, n_matches: int = 20000, seed: int = 7, patch: str = "16.19",
             prev_patch: str | None = "16.18", prev_frac: float = 0.15, target: str = "Briar", target_role: str = "JUNGLE",
             target_frac: float = 0.7, tiers: tuple = ("MASTER", "DIAMOND", "EMERALD")) -> int:
    rnd = random.Random(seed)
    pools = _ids(static)
    target_id = static.champion_id(target)
    lee = static.champion_id("Lee Sin")
    profiles = {cid: prior_profile(static, cid) for ids in pools.values() for cid in ids}
    pages = _pages(static)
    batch, written = [], 0
    for m in range(n_matches):
        match_id = f"SYN_{seed}_{m}"
        game_patch = prev_patch if prev_patch and rnd.random() < prev_frac else patch
        minutes = min(45.0, max(16.0, rnd.gauss(29, 5.5)))
        has_target = rnd.random() < target_frac
        picks: dict[int, dict[str, int]] = {}
        used: set[int] = set()
        for team in (100, 200):
            picks[team] = {}
            for role in ROLES:
                if has_target and team == 100 and role == target_role:
                    cid = target_id
                else:
                    cid = rnd.choice([c for c in pools[role] if c not in used and c != target_id])
                used.add(cid)
                picks[team][role] = cid
        tier = rnd.choice(tiers)
        win100 = rnd.random() < 0.5
        target_row = None
        if has_target:
            enemies = list(picks[200].values())
            c = trait_vector(enemies, profiles)
            lane_gd = rnd.gauss(0, 900)
            team_gd = rnd.gauss(lane_gd * 1.2, 1500)
            # Selection: Collector-first more likely when already ahead.
            weights = [w * (math.exp(0.0012 * lane_gd) if p[0] == COLLECTOR else 1.0) for p, w in PATHS]
            path = rnd.choices([p for p, _ in PATHS], weights)[0]
            page = rnd.choices([pg["p"] for pg in pages], [pg["w"] for pg in pages])[0]
            boots = MERCS if rnd.random() < 0.25 + 0.5 * c[1] else STEELCAPS
            eta = _true_effect(path, page, c, picks[200]["JUNGLE"] == lee, lane_gd, team_gd, boots)
            win100 = rnd.random() < 1 / (1 + math.exp(-eta))
            t = max(9.0, rnd.gauss(13.5, 1.5) - lane_gd / 2000)
            items, n_items = [], min(6, int((minutes - t) / 6.5) + 1)
            seq = list(path) + rnd.sample([i for i in LATE_POOL if i not in path], 3)
            boots_slot = 1 if rnd.random() < 0.75 else 2
            for k in range(min(n_items, 6)):
                if k == boots_slot:
                    items.append({"id": boots, "min": round(t - 1.0, 2), "lane_gd": int(lane_gd), "team_gd": int(team_gd), "boots": True})
                gd = lane_gd if k == 0 else lane_gd + rnd.gauss(0, 500) * k
                items.append({"id": seq[k], "min": round(t, 2), "lane_gd": int(gd), "team_gd": int(team_gd * (1 + 0.3 * k)), "boots": False})
                t += max(4.0, rnd.gauss(7, 1.5))
            route = rnd.choices(list(ROUTES), [1.0, 1.0, 0.4])[0]
            positions = [[x + rnd.gauss(0, 400), y + rnd.gauss(0, 400)] for x, y in ROUTES[route]]
            target_row = {"items": items, "start_items": [1101, 2003], "positions": positions, "page": page}
        for team in (100, 200):
            for i, role in enumerate(ROLES):
                cid = picks[team][role]
                other = 300 - team
                won = win100 if team == 100 else not win100
                pid = i + 1 + (0 if team == 100 else 5)
                is_target = has_target and team == 100 and role == target_role
                page = target_row["page"] if is_target else pages[0]["p"]
                row = {
                    "match_id": match_id, "platform": "syn1", "patch": game_patch, "game_start": 0,
                    "duration_s": int(minutes * 60), "tier": tier, "puuid": f"p{m}_{pid}", "participant_id": pid,
                    "team_id": team, "champion_id": cid, "role": role, "win": bool(won),
                    "allies": [{"champion_id": picks[team][r], "role": r} for r in ROLES if r != role],
                    "enemies": [{"champion_id": picks[other][r], "role": r} for r in ROLES],
                    "primary_style": page[0], "sub_style": page[1], "keystone": page[2][0], "perks": list(page[2]),
                    "shards": [5008, 5008, 5011], "final_items": [],
                    "stats": _stats(profiles[cid], minutes, rnd), "has_timeline": is_target,
                    "items": target_row["items"] if is_target else [],
                    "start_items": target_row["start_items"] if is_target else [],
                    "positions": target_row["positions"] if is_target else [],
                }
                batch.append(row)
        if len(batch) >= 5000:
            store.add_rows(batch)
            written += len(batch)
            batch = []
    store.add_rows(batch)
    return written + len(batch)
