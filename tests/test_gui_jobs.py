"""The desktop app's background jobs (UI-free parts)."""

import json

import pytest

from buildopt.app.settings import Settings
from buildopt.gui import jobs


def test_current_patches(static):
    major, minor = static.version.split(".")[:2]
    patches = jobs.current_patches(static)
    assert patches[0] == f"{major}.{minor}"
    assert len(patches) == 2


def test_test_key_needs_a_key():
    assert jobs.test_key("  ", "na1") == "Paste your key first."


def test_build_recommendations_without_data(tmp_path, monkeypatch):
    monkeypatch.setenv("BUILDOPT_HOME", str(tmp_path))
    s = Settings.load()
    assert jobs.games_collected(s) == 0
    with pytest.raises(ValueError, match="Not enough games yet"):
        jobs.build_recommendations(s, demo=True)


def test_settings_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("BUILDOPT_HOME", str(tmp_path))
    s = Settings.load()
    s.api_key, s.regions, s.champion = "RGAPI-x", ["kr"], "Lee Sin"
    s.save()
    data = json.loads((tmp_path / "settings.json").read_text())
    assert data["regions"] == ["kr"] and data["champion"] == "Lee Sin"
    again = Settings.load()
    assert again.api_key == "RGAPI-x" and again.db_path == tmp_path / "games.sqlite"
