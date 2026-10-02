"""Joint rune x build-path logistic model, one per champion and role.

    logit P(win) = b_path + b_rune + g'c + d_path'c + b_path x rune + k_keystone'c + t's  (+ per-enemy terms)

c  = standardised enemy comp traits
d  = how each path's value shifts with each trait (answers "when does it win")
s  = game-state controls at first-item completion (controls, not recommendations)

Every coefficient is ridge-penalised: main effects shrink toward the champion
average, interaction terms shrink harder toward zero unless the data supports
them. Per-enemy-champion terms are only added where a matchup has 300+ games.
"""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from buildopt.analysis.dataset import Game
from buildopt.analysis.traits import TRAITS

CONTROLS = ("first_lane_gd", "first_team_gd", "first_min")
TRAIT_CLIP = 2.5
N_STATE_OFFSETS = 41


def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, penalty: np.ndarray,
                 max_iter: int = 100, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Weighted ridge logistic regression by Newton-Raphson.

    Returns (beta, covariance), with covariance = inverse penalised Hessian.
    """
    n, k = X.shape
    beta = np.zeros(k)
    pen = np.asarray(penalty, dtype=float)
    H = np.eye(k)
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        g = X.T @ (w * (y - p)) - pen * beta
        W = w * p * (1 - p)
        H = (X * W[:, None]).T @ X + np.diag(pen)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    H = (X * (w * p * (1 - p))[:, None]).T @ X + np.diag(pen)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    return beta, cov


@dataclass
class Penalties:
    main: float = 2.0  # path and rune page main effects (shrink toward champion average)
    trait: float = 1.0
    interaction: float = 15.0  # path x trait, keystone x trait
    pair: float = 25.0  # path x rune page
    control: float = 0.01
    champion: float = 25.0  # per-enemy-champion terms

    def scaled(self, k: float) -> "Penalties":
        """Scale every shrinkage term (not the game-state controls) by k."""
        return Penalties(self.main * k, self.trait * k, self.interaction * k, self.pair * k, self.control, self.champion * k)

    def with_group(self, group: str, k: float) -> "Penalties":
        """Scale one group of terms by k."""
        p = Penalties(**self.__dict__)
        for name in PENALTY_GROUPS[group]:
            setattr(p, name, getattr(p, name) * k)
        return p


# Groups tuned separately, in this order: the trait interactions that answer
# "when does a path win" first, then path x page pairs, matchup terms, main effects.
PENALTY_GROUPS = {"interaction": ("interaction",), "pair": ("pair",), "champion": ("champion",), "main": ("main", "trait")}
SCALE_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def tune_penalties(games: list, paths: list[tuple], pages: list[tuple], base: Penalties | None = None,
                   folds: int = 3, grid=SCALE_GRID) -> tuple[Penalties, dict]:
    """Choose shrinkage per term group by k-fold cross-validated log-likelihood.

    One pass of coordinate search over the groups. Scored the way the model is
    used (pre-game, game state marginalised), so the penalties are tuned for
    recommendation quality rather than fit with game state. The feature space
    is built once on all games and folds slice its rows.
    """
    import zlib

    pen = base or Penalties()
    spec = build_model(games, paths, pages)
    use = spec.usable(games)
    X, y, w = spec.design(use)
    fold = np.array([zlib.crc32(g.row["match_id"].encode()) % folds for g in use])
    a, b = spec.layout()["control"]

    def cv(p: Penalties) -> float:
        vec = spec.penalty_vector(p)
        total = 0.0
        for f in range(folds):
            tr, te = fold != f, fold == f
            beta, _ = fit_logistic(X[tr], y[tr], w[tr], vec)
            offsets = X[tr][:, a:b] @ beta[a:b]
            spec.state_offsets = np.quantile(offsets, np.linspace(0.0125, 0.9875, N_STATE_OFFSETS)).tolist()
            Xte = X[te].copy()
            Xte[:, a:b] = 0.0
            prob = np.clip(spec.marginal(Xte @ beta)[0], 1e-6, 1 - 1e-6)
            total += float(np.sum(w[te] * (y[te] * np.log(prob) + (1 - y[te]) * np.log(1 - prob))))
        return total / float(np.sum(w))

    chosen, best_score = {}, None
    for group in PENALTY_GROUPS:
        scores = {k: cv(pen.with_group(group, k)) for k in grid}
        k = max(scores, key=scores.get)
        chosen[group], best_score = k, scores[k]
        pen = pen.with_group(group, k)
    return pen, {"scales": chosen, "cv_log_likelihood": best_score}


@dataclass
class LoadoutModel:
    paths: list[tuple[int, ...]]
    pages: list[tuple]
    keystones: list[int]
    trait_mean: list[float]
    trait_std: list[float]
    control_mean: list[float]
    control_std: list[float]
    pairs: list[tuple[int, int]]  # (path index, page index) with their own interaction term
    enemy_terms: list[tuple[int, str]]  # (enemy champion id, enemy role)
    beta: np.ndarray | None = None
    cov: np.ndarray | None = None
    baseline_rate: float = 0.5
    state_offsets: list[float] = field(default_factory=lambda: [0.0])  # quantiles of t's over training games
    meta: dict = field(default_factory=dict)

    # ---- layout --------------------------------------------------------
    @property
    def T(self) -> int:
        return len(TRAITS)

    def layout(self) -> dict[str, tuple[int, int]]:
        P, R, K, T = len(self.paths), len(self.pages), len(self.keystones), self.T
        sizes = [("intercept", 1), ("path", P), ("page", R), ("trait", T), ("path_trait", P * T),
                 ("keystone_trait", K * T), ("pair", len(self.pairs)), ("control", len(CONTROLS)),
                 ("enemy", len(self.enemy_terms))]
        out, start = {}, 0
        for name, size in sizes:
            out[name] = (start, start + size)
            start += size
        return out

    @property
    def n_features(self) -> int:
        return self.layout()["enemy"][1]

    def penalty_vector(self, pen: Penalties) -> np.ndarray:
        lay = self.layout()
        v = np.zeros(self.n_features)
        v[lay["intercept"][0]] = 1e-6
        for name, lam in (("path", pen.main), ("page", pen.main), ("trait", pen.trait), ("path_trait", pen.interaction),
                          ("keystone_trait", pen.interaction), ("pair", pen.pair), ("control", pen.control),
                          ("enemy", pen.champion)):
            a, b = lay[name]
            v[a:b] = lam
        return v

    # ---- features ------------------------------------------------------
    def standardise_traits(self, c_raw) -> np.ndarray:
        # Clip to the range seen in training so unusual comps don't extrapolate wildly.
        z = (np.asarray(c_raw, float) - np.asarray(self.trait_mean)) / np.asarray(self.trait_std)
        return np.clip(z, -TRAIT_CLIP, TRAIT_CLIP)

    def row(self, path_i: int, page_i: int, cz: np.ndarray, sz: np.ndarray | None = None,
            enemies: set[tuple[int, str]] | None = None) -> np.ndarray:
        lay = self.layout()
        T = self.T
        x = np.zeros(self.n_features)
        x[lay["intercept"][0]] = 1.0
        x[lay["path"][0] + path_i] = 1.0
        x[lay["page"][0] + page_i] = 1.0
        a = lay["trait"][0]
        x[a:a + T] = cz
        a = lay["path_trait"][0] + path_i * T
        x[a:a + T] = cz
        k = self.keystones.index(self.pages[page_i][2])
        a = lay["keystone_trait"][0] + k * T
        x[a:a + T] = cz
        pair_index = self._pair_index.get((path_i, page_i))
        if pair_index is not None:
            x[lay["pair"][0] + pair_index] = 1.0
        if sz is not None:
            a = lay["control"][0]
            x[a:a + len(CONTROLS)] = sz
        if enemies:
            a = lay["enemy"][0]
            for j, term in enumerate(self.enemy_terms):
                if term in enemies:
                    x[a + j] = 1.0
        return x

    @property
    def _pair_index(self) -> dict:
        cache = self.__dict__.get("_pair_cache")
        if cache is None or len(cache) != len(self.pairs):
            cache = {tuple(p): i for i, p in enumerate(self.pairs)}
            self.__dict__["_pair_cache"] = cache
        return cache

    def controls(self, g: Game) -> np.ndarray:
        raw = np.array([g.first_lane_gd, g.first_team_gd, g.first_min], float)
        return (raw - np.asarray(self.control_mean)) / np.asarray(self.control_std)

    def design(self, games: list[Game]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path_idx = {p: i for i, p in enumerate(self.paths)}
        page_idx = {p: i for i, p in enumerate(self.pages)}
        X, y, w = [], [], []
        for g in games:
            pi = path_idx.get(g.path[:3])
            ri = page_idx.get(g.page)
            if pi is None or ri is None:
                continue
            enemies = {(e["champion_id"], e["role"]) for e in g.enemies}
            X.append(self.row(pi, ri, self.standardise_traits(g.c_raw), self.controls(g), enemies))
            y.append(g.win)
            w.append(g.weight)
        if not X:
            return np.zeros((0, self.n_features)), np.zeros(0), np.zeros(0)
        return np.vstack(X), np.array(y, float), np.array(w, float)

    def usable(self, games: list[Game]) -> list[Game]:
        paths, pages = set(self.paths), set(self.pages)
        return [g for g in games if g.path[:3] in paths and g.page in pages]

    # ---- fit / predict -------------------------------------------------
    def fit(self, games: list[Game], pen: Penalties | None = None) -> "LoadoutModel":
        X, y, w = self.design(games)
        if len(y) == 0:
            raise ValueError("No usable games for the model (no candidate path/page coverage).")
        self.beta, self.cov = fit_logistic(X, y, w, self.penalty_vector(pen or Penalties()))
        self.baseline_rate = float(np.sum(w * y) / np.sum(w))
        a, b = self.layout()["control"]
        offsets = X[:, a:b] @ self.beta[a:b]
        self.state_offsets = np.quantile(offsets, np.linspace(0.0125, 0.9875, N_STATE_OFFSETS)).tolist()
        self.meta["train_games"] = int(len(y))
        return self

    def marginal(self, eta) -> tuple[np.ndarray, np.ndarray]:
        """Pre-game win probability: average over the game states seen in training.

        Game-state controls are unknown in champ select, and plugging in the
        average state would overstate how far a loadout moves the win rate.
        Returns (probability, d probability / d eta).
        """
        eta = np.asarray(eta, float)
        o = np.asarray(self.state_offsets)
        p = 1.0 / (1.0 + np.exp(-(eta[..., None] + o)))
        return p.mean(axis=-1), (p * (1 - p)).mean(axis=-1)

    def linear(self, x: np.ndarray) -> tuple[float, float]:
        eta = float(x @ self.beta)
        var = float(x @ self.cov @ x)
        return eta, max(var, 1e-12)

    def predict_games(self, games: list[Game]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, y, w = self.design(games)
        p = 1.0 / (1.0 + np.exp(-(X @ self.beta)))
        return p, y, w

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "paths": [list(p) for p in self.paths],
            "pages": [list(p) for p in self.pages],
            "keystones": self.keystones,
            "traits": list(TRAITS),
            "trait_mean": self.trait_mean,
            "trait_std": self.trait_std,
            "controls": list(CONTROLS),
            "control_mean": self.control_mean,
            "control_std": self.control_std,
            "pairs": [list(p) for p in self.pairs],
            "enemy_terms": [list(t) for t in self.enemy_terms],
            "beta": self.beta.tolist(),
            "cov_f32": base64.b64encode(self.cov.astype(np.float32).tobytes()).decode("ascii"),
            "baseline_rate": self.baseline_rate,
            "state_offsets": self.state_offsets,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoadoutModel":
        m = cls(
            paths=[tuple(p) for p in d["paths"]],
            pages=[tuple(p) for p in d["pages"]],
            keystones=list(d["keystones"]),
            trait_mean=d["trait_mean"],
            trait_std=d["trait_std"],
            control_mean=d["control_mean"],
            control_std=d["control_std"],
            pairs=[tuple(p) for p in d["pairs"]],
            enemy_terms=[(int(c), r) for c, r in d["enemy_terms"]],
            baseline_rate=d.get("baseline_rate", 0.5),
            state_offsets=d.get("state_offsets", [0.0]),
            meta=d.get("meta", {}),
        )
        m.beta = np.array(d["beta"], float)
        k = len(m.beta)
        m.cov = np.frombuffer(base64.b64decode(d["cov_f32"]), dtype=np.float32).astype(float).reshape(k, k)
        return m


def build_model(games: list[Game], paths: list[tuple], pages: list[tuple], min_pair_games: int = 30,
                min_enemy_games: int = 300) -> LoadoutModel:
    """Set up the feature space (standardisation, interaction pairs, enemy terms) from training games."""
    path_set, page_set = set(paths), set(pages)
    use = [g for g in games if g.path[:3] in path_set and g.page in page_set]
    if not use:
        raise ValueError("No games match the candidate paths and pages.")
    C = np.array([g.c_raw for g in use], float)
    S = np.array([[g.first_lane_gd, g.first_team_gd, g.first_min] for g in use], float)
    tstd = C.std(axis=0)
    sstd = S.std(axis=0)
    path_i = {p: i for i, p in enumerate(paths)}
    page_i = {p: i for i, p in enumerate(pages)}
    pair_counts = Counter((path_i[g.path[:3]], page_i[g.page]) for g in use)
    pairs = sorted(p for p, n in pair_counts.items() if n >= min_pair_games)
    # Per-champion terms only for the exact matchup (the enemy in the same role, e.g. the enemy jungler).
    enemy_counts = Counter((e["champion_id"], e["role"]) for g in use for e in g.enemies if e["role"] == g.row["role"])
    enemy_terms = sorted(t for t, n in enemy_counts.items() if n >= min_enemy_games)
    return LoadoutModel(
        paths=list(paths),
        pages=list(pages),
        keystones=sorted({p[2] for p in pages}),
        trait_mean=C.mean(axis=0).tolist(),
        trait_std=np.where(tstd > 1e-9, tstd, 1.0).tolist(),
        control_mean=S.mean(axis=0).tolist(),
        control_std=np.where(sstd > 1e-9, sstd, 1.0).tolist(),
        pairs=pairs,
        enemy_terms=enemy_terms,
    )
