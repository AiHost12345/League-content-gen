"""Score every rune page x build path against the current enemy comp.

Runs locally from a stats bundle. Loadouts are ranked by the Wilson lower
bound of the model-adjusted win rate, where the interval's effective sample
size comes from the model's uncertainty for that exact loadout and comp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from buildopt.analysis.model import LoadoutModel
from buildopt.analysis.traits import TRAITS, describe, trait_level, trait_vector
from buildopt.bundle import CATEGORY_TRAIT, data_version_label
from buildopt.stats import confidence_label, wilson_interval


@dataclass
class Loadout:
    path_i: int
    page_i: int
    path: tuple[int, ...]
    page: tuple
    win_rate: float
    lo: float
    hi: float
    n_eff: float
    games: int  # games with this exact path + page
    confidence: str

    @property
    def keystone(self) -> int:
        return self.page[2]

    @property
    def key(self) -> tuple:
        return (self.path, self.page)

    def to_dict(self) -> dict:
        return {"path": list(self.path), "page": list(self.page), "win_rate": self.win_rate, "lo": self.lo,
                "hi": self.hi, "n_eff": self.n_eff, "games": self.games, "confidence": self.confidence}


@dataclass
class Recommendation:
    best: Loadout
    runner_up: Loadout | None  # best loadout with a different keystone
    alt_path: Loadout | None  # best different path on the chosen page
    why: list[str]
    traits: list[float]
    data_version: str
    shards: tuple[int, int, int]
    ranked: list[Loadout] = field(default_factory=list)
    known: int = 5  # enemies locked when this was scored

    def to_dict(self) -> dict:
        return {
            "best": self.best.to_dict(),
            "runner_up": self.runner_up.to_dict() if self.runner_up else None,
            "alt_path": self.alt_path.to_dict() if self.alt_path else None,
            "why": self.why,
            "traits": dict(zip(TRAITS, self.traits)),
            "shards": list(self.shards),
            "data_version": self.data_version,
        }


class Scorer:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = LoadoutModel.from_dict(bundle["model"])
        self.profiles = bundle["profiles"]
        self.levels = {k: tuple(v) for k, v in bundle.get("levels", {}).items()}
        self.item_names = bundle.get("item_names", {})
        self.perk_names = bundle.get("perk_names", {})
        self.champion_names = bundle.get("champion_names", {})
        self._base = self._precompute()

    # ---- names -------------------------------------------------------------
    def item(self, item_id: int) -> str:
        return self.item_names.get(str(item_id), str(item_id))

    def perk(self, perk_id: int) -> str:
        return self.perk_names.get(str(perk_id), str(perk_id))

    def champion(self, cid: int) -> str:
        return self.champion_names.get(str(cid), str(cid))

    def path_label(self, path: Sequence[int]) -> str:
        return " → ".join(self.item(i) for i in path)

    def page_label(self, page: Sequence[int]) -> str:
        return f"{self.perk(page[2])} ({', '.join(self.perk(p) for p in page[3:])})"

    @property
    def data_version(self) -> str:
        return data_version_label(self.bundle)

    # ---- scoring -----------------------------------------------------------
    def _precompute(self):
        m = self.model
        combos = [(pi, ri) for pi in range(len(m.paths)) for ri in range(len(m.pages))]
        zero = np.zeros(m.T)
        A = np.vstack([m.row(pi, ri, zero) for pi, ri in combos])  # comp-independent part
        # Trait-dependent columns: x = A + sum_j cz_j * B_j
        lay = m.layout()
        B = np.zeros((m.T, len(combos), m.n_features))
        for c, (pi, ri) in enumerate(combos):
            k = m.keystones.index(m.pages[ri][2])
            for j in range(m.T):
                B[j, c, lay["trait"][0] + j] = 1.0
                B[j, c, lay["path_trait"][0] + pi * m.T + j] = 1.0
                B[j, c, lay["keystone_trait"][0] + k * m.T + j] = 1.0
        info = self.bundle.get("path_info", [])
        games = {}
        for d in info:
            games[tuple(d["path"])] = d["games"]
        return combos, A, B, games

    def traits(self, enemies: Sequence[int]) -> list[float]:
        return trait_vector(enemies, self.profiles)

    def score(self, enemies: Sequence[int], enemy_roles: dict[int, str] | None = None) -> list[Loadout]:
        m = self.model
        combos, A, B, path_games = self._base
        c_raw = self.traits(enemies)
        cz = m.standardise_traits(c_raw)
        X = A + np.tensordot(cz, B, axes=1)
        if enemy_roles:
            terms = {(int(c), r) for c, r in enemy_roles.items()}
            a = m.layout()["enemy"][0]
            for j, t in enumerate(m.enemy_terms):
                if t in terms:
                    X[:, a + j] = 1.0
        eta = X @ m.beta
        var = np.einsum("ij,jk,ik->i", X, m.cov, X)
        probs, slopes = m.marginal(eta)
        out = []
        for (pi, ri), p, d, v in zip(combos, probs, slopes, var):
            p = float(p)
            # Effective sample size implied by the model's uncertainty (delta method on the marginal rate).
            n_eff = p * (1 - p) / max(d * d * v, 1e-12)
            lo, hi = wilson_interval(p, n_eff)
            games = path_games.get(m.paths[pi], 0)
            out.append(Loadout(pi, ri, m.paths[pi], m.pages[ri], p, lo, hi, n_eff, games,
                               confidence_label(min(games, n_eff), lo, hi)))
        out.sort(key=lambda l: -l.lo)
        return out

    def recommend(self, enemies: Sequence[int], enemy_roles: dict[int, str] | None = None) -> Recommendation:
        ranked = self.score(enemies, enemy_roles)
        best = ranked[0]
        runner = next((l for l in ranked if l.keystone != best.keystone), None)
        alt = next((l for l in ranked if l.page_i == best.page_i and l.path[:2] != best.path[:2]), None) or \
            next((l for l in ranked if l.page_i == best.page_i and l.path != best.path), None)
        c_raw = self.traits(enemies)
        return Recommendation(
            best=best, runner_up=runner, alt_path=alt, why=self.explain(best, c_raw, ranked),
            traits=c_raw, data_version=self.data_version, shards=tuple(self.bundle["page_shards"][best.page_i]),
            ranked=ranked, known=len([e for e in enemies if e]),
        )

    # ---- explanations ------------------------------------------------------
    def contributions(self, a: Loadout, b: Loadout, c_raw: Sequence[float]) -> list[tuple[str, float]]:
        """Per-trait contribution to logit(a) - logit(b) for this comp."""
        m = self.model
        lay = m.layout()
        cz = m.standardise_traits(c_raw)
        T = m.T

        def coefs(l: Loadout) -> np.ndarray:
            k = m.keystones.index(l.keystone)
            p0 = lay["path_trait"][0] + l.path_i * T
            k0 = lay["keystone_trait"][0] + k * T
            return m.beta[p0:p0 + T] + m.beta[k0:k0 + T]

        d = (coefs(a) - coefs(b)) * cz
        return sorted(zip(TRAITS, d.tolist()), key=lambda t: -t[1])

    def _diff_label(self, a: Loadout, b: Loadout) -> str:
        for x, y in zip(a.path, b.path):
            if x != y:
                return f"{self.item(x)} over {self.item(y)}"
        if a.keystone != b.keystone:
            return f"{self.perk(a.keystone)} over {self.perk(b.keystone)}"
        return f"{self.page_label(a.page)}"

    def explain(self, best: Loadout, c_raw: Sequence[float], ranked: list[Loadout], limit: int = 3,
                known: int = 5) -> list[str]:
        """The two or three enemy traits that moved the pick.

        Compared against the most popular loadout and against the best loadout
        that starts with a different item, so both the item and the rune choice
        get a reason when the comp drove them.
        """
        pop = self.bundle.get("popular", {})
        baselines = [
            next((l for l in ranked if l.path_i == pop.get("path") and l.page_i == pop.get("page")), None),
            next((l for l in ranked if l.path[0] != best.path[0]), None),
        ]
        reasons: list[tuple[float, str]] = []
        seen = set()
        for base in baselines:
            if base is None or base.key == best.key:
                continue
            label = self._diff_label(best, base)
            for trait, contrib in self.contributions(best, base, c_raw)[:2]:
                if contrib < 0.02 or trait in seen:
                    continue
                seen.add(trait)
                reasons.append((contrib, f"{describe(trait, c_raw[TRAITS.index(trait)], self.levels, known)} → {label}"))
        reasons.sort(key=lambda r: -r[0])
        out = [text for _, text in reasons[: limit - 1]]
        heal = c_raw[TRAITS.index("healing")]
        if trait_level("healing", heal, self.levels) == "high":
            item = self.situational_pick("anti-heal", c_raw, exclude=set(best.path))
            if item:
                out.append(f"{describe('healing', heal, self.levels)} → {self.item(item)} in situational")
        if not out:
            out.append("No enemy trait moves the pick much; this is the strongest loadout overall.")
        return out[:limit]

    def change_reason(self, old: Recommendation, new: Recommendation) -> str:
        """Why the top loadout changed, e.g. 'Titanic Hydra → The Collector: enemy now has 0 tanks'."""
        a, b = new.best, old.best
        if a.path != b.path:
            head = next(f"{self.item(y)} → {self.item(x)}" for x, y in zip(a.path, b.path) if x != y)
        else:
            head = f"{self.perk(b.keystone)} → {self.perk(a.keystone)}"
        # The trait that changed since the last pick and now pushes hardest toward the new loadout.
        changed = {t for t, x, y in zip(TRAITS, new.traits, old.traits) if abs(x - y) > 1e-6}
        best_t = next((t for t, contrib in self.contributions(a, b, new.traits) if t in changed and contrib > 0), None)
        if best_t is None:
            return f"{head}: updated for the new enemy pick"
        return f"{head}: enemy now has {describe(best_t, new.traits[TRAITS.index(best_t)], self.levels, new.known)}"

    # ---- items around the core path -----------------------------------------
    def _pick_from_stats(self, stats: dict, exclude: set[int], min_games: int = 30) -> int | None:
        best, best_lo = None, -1.0
        for item, s in stats.items():
            item = int(item)
            if item in exclude or s["n"] < min_games or s["w"] <= 0:
                continue
            p = s["wins"] / s["w"]
            lo, _ = wilson_interval(p, s["w"] ** 2 / s["w2"])
            if lo > best_lo:
                best, best_lo = item, lo
        return best

    def situational_pick(self, category: str, c_raw: Sequence[float], exclude: set[int] = frozenset()) -> int | None:
        stats = self.bundle.get("situational", {}).get(category, {})
        trait, _ = CATEGORY_TRAIT[category]
        level = trait_level(trait, c_raw[TRAITS.index(trait)], self.levels)
        return self._pick_from_stats(stats.get(level, {}), set(exclude)) or \
            self._pick_from_stats(stats.get("all", {}), set(exclude), min_games=10)

    def situational_reason(self, category: str, c_raw: Sequence[float]) -> str | None:
        trait, direction = CATEGORY_TRAIT[category]
        value = c_raw[TRAITS.index(trait)]
        if trait_level(trait, value, self.levels) != direction:
            return None
        if trait == "ap_share" and direction == "low":
            return f"vs {round((1 - value) * 100)}% AD damage"
        return f"vs {describe(trait, value, self.levels)}"

    def boots_pick(self, c_raw: Sequence[float]) -> int | None:
        stats = self.bundle.get("boots", {})
        level = trait_level("ap_share", c_raw[TRAITS.index("ap_share")], self.levels)
        return self._pick_from_stats(stats.get(level, {}), set()) or self._pick_from_stats(stats.get("all", {}), set(), 1)

    def boots_slot(self, path: Sequence[int]) -> int:
        for d in self.bundle.get("path_info", []):
            if tuple(d["path"]) == tuple(path):
                return d["boots_slot"]
        return 1

    def late_game(self, exclude: set[int], count: int = 3) -> list[int]:
        return [d["id"] for d in self.bundle.get("late_items", []) if d["id"] not in exclude][:count]

    def prefix_score(self, prefix: Sequence[int], enemies: Sequence[int]) -> Loadout | None:
        """Model score for a shorter path (e.g. Collector → BC): best page, weighted over its 3-item extensions."""
        ranked = self.score(enemies)
        page = ranked[0].page_i
        opts = [l for l in ranked if l.page_i == page and l.path[:len(prefix)] == tuple(prefix)]
        if not opts:
            return None
        w = np.array([max(l.games, 1) for l in opts], float)
        p = float(np.sum(w * [l.win_rate for l in opts]) / w.sum())
        n = float(sum(l.n_eff for l in opts))
        lo, hi = wilson_interval(p, n)
        return Loadout(-1, page, tuple(prefix), opts[0].page, p, lo, hi, n, int(sum(l.games for l in opts)),
                       confidence_label(sum(l.games for l in opts), lo, hi))


def format_recommendation(scorer: Scorer, rec: Recommendation) -> str:
    b = rec.best
    lines = [
        f"{scorer.bundle['champion_name']} {scorer.bundle['role'].lower()} · {rec.data_version}",
        f"Runes : {scorer.page_label(b.page)} · shards {', '.join(scorer.perk(s) for s in rec.shards)}",
        f"Build : {scorer.path_label(b.path)}",
        f"Win   : {b.win_rate:.1%} (95% {b.lo:.1%}–{b.hi:.1%}) · confidence {b.confidence}",
        "Why   :",
        *(f"  - {w}" for w in rec.why),
    ]
    if rec.runner_up:
        r = rec.runner_up
        lines.append(f"Runner-up ({scorer.perk(r.keystone)}): {scorer.path_label(r.path)} · {r.win_rate:.1%} "
                     f"({(r.win_rate - b.win_rate) * 100:+.1f} pts)")
    if rec.alt_path:
        lines.append(f"Alt path: {scorer.path_label(rec.alt_path.path)} · {rec.alt_path.win_rate:.1%}")
    return "\n".join(lines)
