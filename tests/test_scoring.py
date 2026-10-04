import json
import time

from buildopt.bundle import load_bundle, save_bundle
from buildopt.itemset import build_item_set
from buildopt.lcu.champselect import assign_roles
from buildopt.pipeline.synthetic import BORK, COLLECTOR, TITANIC
from buildopt.scoring import Scorer, format_recommendation

TANKY = ["Ornn", "Sejuani", "Galio", "Ashe", "Braum"]
SQUISHY = ["Teemo", "Kha'Zix", "Zed", "Jinx", "Lulu"]


def ids(static, names):
    return [static.champion_id(n) for n in names]


def test_bundle_round_trip(bundle, tmp_path):
    path = save_bundle(bundle, tmp_path)
    again = load_bundle(path)
    assert again["data_version"]["id"] == bundle["data_version"]["id"]
    assert json.loads((tmp_path / "index.json").read_text())["bundles"]["233_JUNGLE"]["file"] == path.name
    a, b = Scorer(bundle), Scorer(again)
    assert a.score([])[0].win_rate == b.score([])[0].win_rate


def test_recommendation_moves_with_tanks(bundle, static):
    sc = Scorer(bundle)
    tanky = sc.recommend(ids(static, TANKY))
    squishy = sc.recommend(ids(static, SQUISHY))
    assert tanky.best.path[0] in (TITANIC, BORK)
    assert squishy.best.path[0] == COLLECTOR
    assert tanky.why and squishy.why
    for rec in (tanky, squishy):
        assert rec.runner_up is None or rec.runner_up.keystone != rec.best.keystone
        assert rec.best.lo <= rec.best.win_rate <= rec.best.hi
        assert 0.3 < rec.best.win_rate < 0.7
        assert rec.data_version.startswith("data 16.19-")
    text = format_recommendation(sc, tanky)
    assert "Titanic Hydra" in text and "data 16.19" in text


def test_ranking_modes(bundle, static):
    sc = Scorer(bundle)
    by_rate = [l.win_rate for l in sc.score(ids(static, TANKY))]
    assert by_rate == sorted(by_rate, reverse=True)  # default: highest win rate first
    by_lo = [l.lo for l in sc.score(ids(static, TANKY), rank_by="lower_bound")]
    assert by_lo == sorted(by_lo, reverse=True)


def test_every_build_is_scored(bundle, games):
    """No shortlist: every 3-item build that appears in the data is in the model and gets a score."""
    seen = {g.path[:3] for g in games if len(g.path) >= 3}
    model_paths = {tuple(p) for p in bundle["model"]["paths"]}
    assert seen == model_paths
    ranked = Scorer(bundle).score([])
    assert {l.path for l in ranked} == model_paths


def test_scoring_is_fast(bundle, static):
    sc = Scorer(bundle)
    enemies = ids(static, TANKY)
    t = time.perf_counter()
    for k in range(1, 6):
        sc.recommend(enemies[:k], assign_roles(enemies[:k], sc.profiles))
    assert (time.perf_counter() - t) / 5 < 0.2  # target: under 200 ms per enemy lock


def test_change_reason(bundle, static):
    sc = Scorer(bundle)
    old = sc.recommend(ids(static, TANKY))
    new = sc.recommend(ids(static, SQUISHY))
    reason = sc.change_reason(old, new)
    assert reason.split(":")[0] in ("Titanic Hydra → The Collector", "Blade of The Ruined King → The Collector")


def test_item_set_blocks(bundle, static):
    sc = Scorer(bundle)
    rec = sc.recommend(ids(static, ["Aatrox", "Warwick", "Vladimir", "Samira", "Soraka"]))
    s = build_item_set(sc, rec)
    titles = [b["type"] for b in s["blocks"]]
    assert titles[0] == "Start" and titles[1].startswith("Core")
    assert any(t.startswith("Situational: anti-heal (vs heavy healing)") for t in titles)
    assert "Late game" in titles and any(t.startswith("Alt path") for t in titles)
    core = [int(i["id"]) for i in s["blocks"][1]["items"]]
    assert [i for i in core if i in rec.best.path] == list(rec.best.path)
    assert any(static.is_boots(i) for i in core)
    assert s["title"].startswith("BO: Briar 16.19") and s["associatedChampions"] == [233] and s["associatedMaps"] == [11]


def test_enemy_role_assignment(bundle, static):
    roles = assign_roles(ids(static, ["Jinx", "Thresh", "Lee Sin", "Ahri", "Darius"]), bundle["profiles"])
    by_name = {static.champion_name(c): r for c, r in roles.items()}
    assert by_name == {"Jinx": "BOTTOM", "Thresh": "UTILITY", "Lee Sin": "JUNGLE", "Ahri": "MIDDLE", "Darius": "TOP"}


def test_top_builds_include_bork_into_tanks(bundle, static):
    sc = Scorer(bundle)
    rec = sc.recommend(ids(static, TANKY))
    firsts = [l.path[0] for l in sc.top_builds(rec.ranked, 5)]
    assert len(set(l.path for l in sc.top_builds(rec.ranked, 5))) == 5
    assert BORK in firsts


def test_first_item_stats(bundle):
    firsts = {d["id"]: d for d in bundle["first_items"]}
    assert {COLLECTOR, TITANIC, BORK} <= set(firsts)
    bork = firsts[BORK]
    assert bork["vs_2plus_tanks"]["games"] + bork["vs_0_1_tanks"]["games"] == bork["games"]
