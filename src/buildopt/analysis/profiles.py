"""Per-champion trait profiles and role play-rates.

Each champion gets scores in [0, 1] for the dimensions the enemy trait vector
is built from. Scores are measured from match data (damage split, damage
absorbed, CC time, healing) as percentiles across champions, and blended with
a Data Dragon prior so rarely played champions still get sensible values.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from buildopt import ROLES
from buildopt.ddragon import StaticData

PRIOR_STRENGTH = 30.0  # games-worth of weight given to the Data Dragon prior
MEASURED = ("ap_share", "tank", "cc", "heal", "scaling")

KNOWN_HEALERS = {
    "Soraka", "Aatrox", "Vladimir", "Sylas", "Yuumi", "Dr. Mundo", "Warwick", "Swain", "Briar", "Nidalee", "Sona",
    "Seraphine", "Senna", "Illaoi", "Olaf", "Fiddlesticks", "Kayn", "Trundle", "Volibear", "Nami", "Milio", "Renata Glasc",
    "Zac", "Maokai", "Aphelios", "Samira", "Bel'Veth", "Viego", "Gwen", "Fiora", "Irelia", "Sett", "Darius",
}
KNOWN_POKE = {"Xerath", "Jayce", "Ziggs", "Varus", "Ezreal", "Lux", "Zoe", "Vel'Koz", "Caitlyn", "Nidalee", "Hwei",
              "Corki", "Jhin", "Karma", "Zyra", "Brand", "Kog'Maw", "Syndra", "Mel", "Smolder"}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def prior_profile(static: StaticData, champ_id: int) -> dict:
    c = static.champion(champ_id)
    if c is None:
        return {"ap_share": 0.4, "tank": 0.3, "cc": 0.4, "heal": 0.3, "burst": 0.4, "ranged": 0.0, "poke": 0.2,
                "mobility": 0.4, "scaling": 0.5, "n": 0, "role_rates": {r: 0.2 for r in ROLES}}
    tags = set(c["tags"])
    info = c["info"]
    st = c["stats"]
    ap = info["magic"] / max(1, info["magic"] + info["attack"])
    tank_tag = 0.9 if "Tank" in tags else 0.5 if "Fighter" in tags else 0.15
    cc = max({"Tank": 0.8, "Support": 0.7, "Mage": 0.5, "Fighter": 0.4, "Assassin": 0.3, "Marksman": 0.2}.get(t, 0.3) for t in tags)
    heal = 0.85 if c["name"] in KNOWN_HEALERS else 0.35 if "Fighter" in tags else 0.2
    burst = max({"Assassin": 0.9, "Mage": 0.6, "Fighter": 0.4}.get(t, 0.2) for t in tags)
    ranged = 1.0 if st["attackrange"] >= 325 else 0.0
    poke = 1.0 if c["name"] in KNOWN_POKE else (0.5 if ranged and tags & {"Mage", "Marksman"} else 0.1)
    mobility = _clamp((st["movespeed"] - 325) / 30 * 0.5 + (0.4 if "Assassin" in tags else 0.0) + 0.1)
    scaling = max({"Marksman": 0.75, "Mage": 0.6, "Tank": 0.5, "Fighter": 0.45, "Support": 0.4, "Assassin": 0.35}.get(t, 0.4)
                  for t in tags)
    return {
        "ap_share": ap,
        "tank": _clamp((tank_tag + info["defense"] / 10) / 2),
        "cc": cc,
        "heal": heal,
        "burst": burst,
        "ranged": ranged,
        "poke": poke,
        "mobility": mobility,
        "scaling": scaling,
        "n": 0,
        "role_rates": prior_role_rates(tags),
    }


def prior_role_rates(tags: set[str]) -> dict[str, float]:
    w = {r: 0.02 for r in ROLES}
    for t in tags:
        for role, v in {
            "Marksman": {"BOTTOM": 1.0}, "Support": {"UTILITY": 1.0}, "Tank": {"TOP": 0.5, "JUNGLE": 0.3, "UTILITY": 0.3},
            "Fighter": {"TOP": 0.6, "JUNGLE": 0.5}, "Mage": {"MIDDLE": 0.8, "UTILITY": 0.2}, "Assassin": {"MIDDLE": 0.6, "JUNGLE": 0.5},
        }.get(t, {}).items():
            w[role] += v
    s = sum(w.values())
    return {r: v / s for r, v in w.items()}


def build_profiles(rows: Iterable[dict], static: StaticData) -> dict[int, dict]:
    """Aggregate per-champion stats from every row (all ten players of every game)."""
    acc: dict[int, dict] = defaultdict(lambda: {"n": 0, "ap": 0.0, "tank": 0.0, "cc": 0.0, "heal": 0.0, "roles": defaultdict(int),
                                                "long": [0, 0], "short": [0, 0]})
    for r in rows:
        s = r["stats"]
        minutes = max(r["duration_s"] / 60, 10)
        a = acc[r["champion_id"]]
        a["n"] += 1
        a["ap"] += s["dmg_magic"] / max(1, s["dmg_total"])
        a["tank"] += (s["dmg_taken"] + s["dmg_mitigated"]) / minutes
        a["cc"] += s["cc_time"] / minutes
        a["heal"] += s["heal"] / minutes
        a["roles"][r["role"]] += 1
        if r["duration_s"] >= 30 * 60:
            a["long"][0] += int(r["win"])
            a["long"][1] += 1
        elif r["duration_s"] < 25 * 60:
            a["short"][0] += int(r["win"])
            a["short"][1] += 1

    def slope(a: dict) -> float:
        # win rate in long games minus short games, each shrunk toward 50%
        return (a["long"][0] + 5) / (a["long"][1] + 10) - (a["short"][0] + 5) / (a["short"][1] + 10)

    means = {cid: {**{k: a[k] / a["n"] for k in ("ap", "tank", "cc", "heal")}, "scaling": slope(a)}
             for cid, a in acc.items() if a["n"]}
    reliable = [cid for cid, a in acc.items() if a["n"] >= 20] or list(means)

    def pct(key: str, value: float) -> float:
        ref = sorted(means[c][key] for c in reliable)
        if not ref:
            return 0.5
        below = sum(1 for v in ref if v < value)
        return below / max(1, len(ref) - 1) if len(ref) > 1 else 0.5

    profiles: dict[int, dict] = {}
    for key in static.champions:
        cid = int(key)
        prof = prior_profile(static, cid)
        a = acc.get(cid)
        if a and a["n"]:
            n = a["n"]
            w = n / (n + PRIOR_STRENGTH)
            m = means[cid]
            measured = {"ap_share": m["ap"], "tank": pct("tank", m["tank"]), "cc": pct("cc", m["cc"]),
                        "heal": pct("heal", m["heal"]), "scaling": pct("scaling", m["scaling"])}
            for k in MEASURED:
                prof[k] = w * measured[k] + (1 - w) * prof[k]
            tot = sum(a["roles"].values())
            prior_rr = prof["role_rates"]
            prof["role_rates"] = {r: (a["roles"].get(r, 0) + 5 * prior_rr[r]) / (tot + 5) for r in ROLES}
            prof["n"] = n
        profiles[cid] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in prof.items()}
    return profiles
