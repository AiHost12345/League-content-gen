from buildopt.analysis import jungle, wincon


def test_zone_mapping_is_side_symmetric():
    assert jungle.zone((3800, 7900), 100) == "own blue side"
    assert jungle.zone((14800 - 3800, 14800 - 7900), 200) == "own blue side"
    assert jungle.zone((7800, 4000), 100) == "own red side"
    assert jungle.zone((11000, 6900), 100) == "enemy blue side"


def test_route_stats_from_synthetic_games(bundle):
    stats = bundle["jungle"]
    assert stats["routes"]
    best = jungle.best_routes(stats, None)
    assert best and best[0]["lo"] >= best[-1]["lo"]


def test_gank_priority(bundle, static):
    ids = static.champion_id
    allies = {"TOP": ids("Ornn"), "MIDDLE": ids("Orianna"), "BOTTOM": ids("Ashe"), "UTILITY": ids("Leona")}
    enemies = {"TOP": ids("Fiora"), "MIDDLE": ids("Xerath"), "BOTTOM": ids("Jinx"), "UTILITY": ids("Soraka")}
    ranked = jungle.gank_priority(allies, enemies, bundle["profiles"])
    assert [r["lane"] for r in ranked][0] == "BOTTOM"


def test_wincon_analysis(bundle, static):
    table = bundle["wincon"]
    ids = [static.champion_id(n) for n in ("Malphite", "Sejuani", "Orianna", "Miss Fortune", "Leona")]
    them = [static.champion_id(n) for n in ("Fiora", "Kha'Zix", "Zed", "Caitlyn", "Lulu")]
    res = wincon.analyze(ids, them, table)
    assert res["your_archetype"] in wincon.ARCHETYPES
    assert res["fatal_flaw"] in wincon.SIGNALS
    assert res["gaps"]["frontline"] > 0


def test_fit_signal_weights(synth_store, bundle):
    fitted = wincon.fit_signal_weights(synth_store.iter_rows(), bundle["wincon"])
    assert len(fitted["weights"]) == len(wincon.SIGNALS)
