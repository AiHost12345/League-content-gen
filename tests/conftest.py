import pytest

from buildopt import ddragon
from buildopt.analysis.dataset import load_games
from buildopt.analysis.profiles import build_profiles
from buildopt.bundle import build_bundle
from buildopt.pipeline.store import Store
from buildopt.pipeline.synthetic import generate


@pytest.fixture(scope="session")
def static():
    return ddragon.load(offline=True)


@pytest.fixture(scope="session")
def synth_store(static, tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("db") / "syn.sqlite")
    generate(store, static, n_matches=20000, seed=11, target_frac=1.0)
    return store


@pytest.fixture(scope="session")
def profiles(synth_store, static):
    return build_profiles(synth_store.iter_rows(patches=synth_store.patches()[-2:]), static)


@pytest.fixture(scope="session")
def games(synth_store, static, profiles):
    games, _ = load_games(synth_store, static.champion_id("Briar"), "JUNGLE", profiles)
    return games


@pytest.fixture(scope="session")
def bundle(synth_store, static, profiles):
    return build_bundle(synth_store, static, static.champion_id("Briar"), "JUNGLE", profiles=profiles)
