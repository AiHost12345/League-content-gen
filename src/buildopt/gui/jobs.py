"""Long-running work behind the desktop app's buttons, kept free of UI code.

Every function here runs on a worker thread and reports through callbacks.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

from buildopt import ddragon
from buildopt.app.settings import Settings, app_dir
from buildopt.bundle import build_bundle, data_version_label, save_bundle
from buildopt.pipeline.store import Store
from buildopt.riot.api import RiotApiError, RiotClient

log = logging.getLogger("buildopt.gui")

REGIONS = {  # platform id -> label
    "na1": "North America", "euw1": "EU West", "eun1": "EU Nordic & East", "kr": "Korea", "br1": "Brazil",
    "la1": "LAN", "la2": "LAS", "oc1": "Oceania", "tr1": "Turkey", "ru": "Russia", "jp1": "Japan",
    "me1": "Middle East", "sg2": "Singapore", "tw2": "Taiwan", "vn2": "Vietnam",
}
ROLE_LABELS = {"TOP": "Top", "JUNGLE": "Jungle", "MIDDLE": "Mid", "BOTTOM": "Bot (ADC)", "UTILITY": "Support"}


def static_data() -> ddragon.StaticData:
    """Current Data Dragon (cached on disk), or the built-in snapshot when offline."""
    return ddragon.load(cache_dir=app_dir() / "ddragon")


def current_patches(static: ddragon.StaticData) -> list[str]:
    major, minor = (int(x) for x in static.version.split(".")[:2])
    patches = [f"{major}.{minor}"]
    if minor > 1:
        patches.append(f"{major}.{minor - 1}")
    return patches


def test_key(api_key: str, region: str) -> str:
    """Return a human message about whether the key works."""
    if not api_key.strip():
        return "Paste your key first."
    client = RiotClient(api_key.strip(), max_retries=1)
    try:
        client.get(region, "/lol/status/v4/platform-data")
    except RiotApiError as e:
        if e.status in (401, 403):
            return "That key was rejected. Personal keys expire every 24 hours, so get a fresh one."
        return f"Riot's servers answered with an error ({e.status}). Try again in a minute."
    except Exception as e:  # network problems
        return f"Couldn't reach Riot's servers: {e}"
    return "OK"


@dataclass
class CollectJob:
    settings: Settings
    on_done: Callable[[str], None]
    stop_event: threading.Event | None = None

    def run(self) -> None:
        from buildopt.pipeline.crawler import CrawlConfig, Crawler

        s = self.settings
        try:
            static = static_data()
            champ = static.champion_id(s.champion)
            patches = current_patches(static)
            log.info("Collecting %s %s games on patch %s from %s", static.champion_name(champ),
                     s.role.lower(), " + ".join(patches), ", ".join(s.regions))
            cfg = CrawlConfig(platforms=list(s.regions), patches=patches, targets={(champ, s.role)},
                              max_target_rows=s.target_games)
            self.crawler = Crawler(RiotClient(s.api_key.strip()), Store(s.db_path), static, cfg)
            if self.stop_event is not None:
                self.crawler.stop_event = self.stop_event
            self.crawler.run()
            self.on_done("stopped" if self.stop_event is not None and self.stop_event.is_set() else "finished")
        except Exception as e:
            log.exception("collection failed")
            self.on_done(f"error: {e}")


def games_collected(settings: Settings) -> int:
    if not settings.db_path.exists():
        return 0
    try:
        static = ddragon.load(offline=True)
        store = Store(settings.db_path)
        n = store.count_rows(static.champion_id(settings.champion), settings.role)
        store.close()
        return n
    except Exception:
        return 0


def build_recommendations(settings: Settings, demo: bool = False) -> str:
    """Fit the model for the chosen champion/role and save the bundle the app reads. Returns a summary."""
    static = ddragon.load(offline=True) if demo else static_data()
    store = Store(settings.demo_db_path if demo else settings.db_path)
    champ = static.champion_id("Briar" if demo else settings.champion)
    role = "JUNGLE" if demo else settings.role
    try:
        bundle = build_bundle(store, static, champ, role)
    except ValueError as e:
        have = store.count_rows(champ, role)
        raise ValueError(f"Not enough games yet ({have:,} collected). Keep collecting and try again. ({e})") from e
    if demo:
        bundle["demo"] = True
        bundle["data_version"]["id"] = "DEMO-" + bundle["data_version"]["id"]
    save_bundle(bundle, settings.bundle_dir)
    return data_version_label(bundle)


def make_demo_data(settings: Settings, progress: Callable[[str], None]) -> str:
    from buildopt.pipeline.synthetic import generate

    path = settings.demo_db_path
    if path.exists():
        path.unlink()
    progress("Making 12,000 fake Briar games…")
    generate(Store(path), ddragon.load(offline=True), n_matches=12000, seed=7, target_frac=1.0)
    progress("Fitting the model (about a minute)…")
    return build_recommendations(settings, demo=True)
