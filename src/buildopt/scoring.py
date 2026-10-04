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
    min_games: int = 0  # builds below this weren't eligible to be the recommendation

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
        """Pull the coefficient blocks out once so every build x page can be scored with array maths."""
        m = self.model
        lay = m.layout()
        beta, T = m.beta, m.T
        P, R = len(m.paths), len(m.pages)

        def block(name):
            a, b = lay[name]
            return beta[a:b]

        D = np.zeros((P, T))  # how each build's value shifts with each trait (0 for rare builds)
        pt = block("path_trait")
        for ti, pi in enumerate(m.trait_paths):
            D[pi] = pt[ti * T:(ti + 1) * T]
        PR = np.zeros((P, R))
        pr = block("pair")
        for idx, (pi, ri) in enumerate(m.pairs):
            PR[pi, ri] = pr[idx]
        games = {tuple(d["path"]): d["games"] for d in self.bundle.get("path_info", [])}
        # Real builds to rank: those with their own model term, plus every rare build on top of its group.
        entries = [(tuple(p), i, 0.0, 0.0, games.get(tuple(p), 0)) for i, p in enumerate(m.paths) if 0 not in p]
        entries += [(tuple(r["path"]), r["group"], r["delta"], r["var"], r["games"])
                    for r in self.bundle.get("rare_paths", [])]
        return {
            "entries": entries,
            "e_term": np.array([e[1] for e in entries], int),
            "e_delta": np.array([e[2] for e in entries], float),
            "b0": beta[lay["intercept"][0]], "bp": block("path"), "br": block("page"), "g": block("trait"),
            "D": D, "K": block("keystone_trait").reshape(len(m.keystones), T),
            "key_of_page": np.array([m.keystones.index(p[2]) for p in m.pages], int), "PR": PR,
            "enemy": block("enemy"),
        }

    def traits(self, enemies: Sequence[int]) -> list[float]:
        return trait_vector(enemies, self.profiles)

    def score(self, enemies: Sequence[int], enemy_roles: dict[int, str] | None = None,
              rank_by: str = "win_rate", detail: int = 120) -> list[Loadout]:
        """Score every build players finished x every rune page for this comp.

        rank_by="win_rate": highest adjusted win rate first, every build eligible.
        rank_by="lower_bound": safest first (Wilson lower bound), which favours builds with lots of games.
        Returns the best page for every build plus the top combinations overall, with intervals.
        """
        m, b = self.model, self._base
        cz = m.standardise_traits(self.traits(enemies))
        enemy_set = {(int(c), r) for c, r in (enemy_roles or {}).items()}
        enemy_shift = sum(b["enemy"][j] for j, t in enumerate(m.enemy_terms) if t in enemy_set)
        eta = (b["b0"] + b["bp"][:, None] + b["br"][None, :] + float(b["g"] @ cz) + (b["D"] @ cz)[:, None]
               + (b["K"] @ cz)[b["key_of_page"]][None, :] + b["PR"] + enemy_shift)
        eta = eta[b["e_term"]] + b["e_delta"][:, None]  # one row per real build
        prob, slope = m.marginal(eta)
        E, R = prob.shape
        best_page = prob.argmax(axis=1)
        chosen = {(e, int(best_page[e])) for e in range(E)}
        flat = np.argsort(-prob, axis=None)[: (detail if rank_by == "win_rate" else max(detail, 3 * E))]
        chosen |= {(int(i // R), int(i % R)) for i in flat}
        chosen = sorted(chosen)

        # Model variance depends only on (model term, page): compute each distinct one once.
        terms = sorted({(int(b["e_term"][e]), ri) for e, ri in chosen})
        X = np.vstack([m.row(t, ri, cz, None, enemy_set) for t, ri in terms])
        var = dict(zip(terms, np.einsum("ij,jk,ik->i", X, m.cov, X)))
        out = []
        for e, ri in chosen:
            path, term, _, extra_var, games = b["entries"][e]
            p, d = float(prob[e, ri]), float(slope[e, ri])
            # Effective sample size implied by the model's uncertainty (delta method on the marginal rate).
            n_eff = p * (1 - p) / max(d * d * (var[(term, ri)] + extra_var), 1e-12)
            lo, hi = wilson_interval(p, n_eff)
            out.append(Loadout(term, ri, path, m.pages[ri], p, lo, hi, n_eff, games,
                               confidence_label(min(games, n_eff), lo, hi)))
        key = (lambda l: -l.lo) if rank_by == "lower_bound" else (lambda l: -l.win_rate)
        out.sort(key=key)
        return out

    @staticmethod
    def top_builds(ranked: list[Loadout], n: int = 5) -> list[Loadout]:
        """The n best distinct build paths for this comp, each on its best rune page."""
        out, seen = [], set()
        for l in ranked:
            if l.path not in seen:
                seen.add(l.path)
                out.append(l)
            if len(out) == n:
                break
        return out

    def recommend(self, enemies: Sequence[int], enemy_roles: dict[int, str] | None = None,
                  rank_by: str = "win_rate", min_games: int = 20) -> Recommendation:
        """Best loadout for this comp. Every build is scored; the one recommended (and imported) must have
        at least ``min_games`` games so a couple of lucky games can't decide your build. ``ranked`` keeps all."""
        all_ranked = self.score(enemies, enemy_roles, rank_by=rank_by)
        ranked = [l for l in all_ranked if l.games >= min_games] or all_ranked
        best = ranked[0]
        runner = next((l for l in ranked if l.keystone != best.keystone), None)
        alt = next((l for l in ranked if l.page_i == best.page_i and l.path[:2] != best.path[:2]), None) or \
            next((l for l in ranked if l.page_i == best.page_i and l.path != best.path), None)
        c_raw = self.traits(enemies)
        return Recommendation(
            best=best, runner_up=runner, alt_path=alt, why=self.explain(best, c_raw, ranked),
            traits=c_raw, data_version=self.data_version, shards=tuple(self.bundle["page_shards"][best.page_i]),
            ranked=all_ranked, known=len([e for e in enemies if e]), min_games=min_games,
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
            k0 = lay["keystone_trait"][0] + k * T
            out = m.beta[k0:k0 + T].copy()
            ti = m.trait_index.get(l.path_i)
            if ti is not None:
                p0 = lay["path_trait"][0] + ti * T
                out += m.beta[p0:p0 + T]
            return out

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
        elif a.keystone != b.keystone:
            head = f"{self.perk(b.keystone)} → {self.perk(a.keystone)}"
        else:  # same keystone, different minor runes
            gone = [p for p in b.page[3:] if p not in a.page[3:]]
            new_ = [p for p in a.page[3:] if p not in b.page[3:]]
            head = f"{', '.join(self.perk(p) for p in gone)} → {', '.join(self.perk(p) for p in new_)}"
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
        opts = [l for l in self.top_builds(ranked, len(ranked)) if l.path[:len(prefix)] == tuple(prefix)]
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
