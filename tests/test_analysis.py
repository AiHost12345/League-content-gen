"""Model and path analysis recover the effects planted in the synthetic data."""

import numpy as np

from buildopt.analysis.candidates import _swap_kind, candidate_pages, candidate_paths
from buildopt.analysis.evaluate import evaluate
from buildopt.analysis.model import fit_logistic
from buildopt.analysis.paths import compare_paths, pair_synergy, prefix_tree, stability
from buildopt.analysis.traits import TRAITS, tercile_levels
from buildopt.pipeline.synthetic import BC, COLLECTOR, TITANIC


def test_fit_logistic_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = np.hstack([np.ones((20000, 1)), rng.normal(size=(20000, 2))])
    true = np.array([0.2, 0.8, -0.5])
    y = (rng.random(20000) < 1 / (1 + np.exp(-X @ true))).astype(float)
    beta, cov = fit_logistic(X, y, np.ones(len(y)), np.full(3, 1e-6))
    assert np.allclose(beta, true, atol=0.06)
    assert np.all(np.diag(cov) > 0)


def test_candidates(games):
    paths = candidate_paths(games)
    assert (COLLECTOR, BC, 6333) in paths and (TITANIC, BC, 6333) in paths
    pages = candidate_pages(games)
    assert 4 <= len(pages) <= 16
    conq = (8000, 8100, 8010, 9111, 9104, 8014, 8139, 8135)
    assert _swap_kind(conq, (8000, 8100, 8008, 9111, 9104, 8014, 8139, 8135)) == "keystone"
    assert _swap_kind(conq, (8000, 8100, 8010, 9111, 9104, 8017, 8139, 8135)) == "minor"
    assert _swap_kind(conq, (8000, 8400, 8010, 9111, 9104, 8014, 8473, 8242)) == "secondary"


def test_patch_weighting(games):
    patches = {g.row["patch"] for g in games}
    assert patches == {"16.19"}  # enough current-patch games, so the previous patch is dropped


def test_prefix_tree_and_pairs(games):
    tree = prefix_tree(games, max_depth=2)
    firsts = {n["items"][0] for n in tree}
    assert {COLLECTOR, TITANIC} <= firsts
    for n in tree:
        assert n["games"] >= 50 and all(c["items"][:1] == n["items"] for c in n["children"])
    pairs = pair_synergy(games)
    assert pairs and all(p["games"] >= 50 for p in pairs)


def test_ipw_removes_selection_bias(games):
    """Collector is bought more when ahead: raw comparison flatters it, IPW should shrink that edge."""
    levels = tercile_levels([g.c_raw for g in games])
    a, b = (COLLECTOR, BC), (TITANIC, BC)
    raw = compare_paths(games, a, b, levels, adjust=False)["conditions"][0]["difference"]
    adj = compare_paths(games, a, b, levels, adjust=True)["conditions"][0]["difference"]
    assert raw > 0.03
    assert adj < raw - 0.02


def test_comparison_finds_the_tank_flip(games):
    levels = tercile_levels([g.c_raw for g in games])
    res = compare_paths(games, (COLLECTOR, BC), (TITANIC, BC), levels)
    by = {c["condition"]: c for c in res["conditions"]}
    assert by["0–1 tanks"]["difference"] > by["2+ tanks"]["difference"]
    assert by["all games"]["games_a"] > 0 and res["headline"]["leader"] in ("a", "b")
    st = stability(games, (COLLECTOR, BC), (TITANIC, BC), levels)
    assert st["conditions_compared"] > 0


def test_model_learns_tank_interaction(bundle):
    from buildopt.bundle import model_of

    m = model_of(bundle)
    lay = m.layout()
    T = len(TRAITS)
    ti = TRAITS.index("tanks")

    def delta(path):
        i = m.paths.index(path)
        return m.beta[lay["path_trait"][0] + i * T + ti]

    assert delta((COLLECTOR, BC, 6333)) < delta((TITANIC, BC, 6333))


def test_evaluation_gate(games):
    res = evaluate(games)
    ll = res["log_likelihood"]
    assert res["test_games"] > 500
    assert ll["model_with_game_state"] > ll["champion_average"]
    assert res["gate_passed"]
