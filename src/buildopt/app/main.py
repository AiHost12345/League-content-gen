"""Desktop companion entry point: connect to the client, subscribe, score, import."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from buildopt.app.bundles import BundleCache, sync_bundles
from buildopt.app.companion import Companion
from buildopt.app.settings import Settings
from buildopt.app.window import ConsoleUI, make_ui
from buildopt.lcu.connection import LcuClient, LcuEvents, read_lockfile
from buildopt.lcu.importer import Importer

log = logging.getLogger(__name__)


def _sync(settings: Settings, ui) -> None:
    if not settings.bundle_source:
        return
    try:
        updated = sync_bundles(settings.bundle_source, settings.bundle_dir)
        if updated:
            ui.status(f"Downloaded {len(updated)} updated bundle(s)")
    except Exception as e:  # network trouble must never block champ select
        log.warning("bundle sync failed: %s", e)
        ui.status(f"Bundle sync failed ({e}); using cached bundles")


def connect_loop(settings: Settings, ui, stop: threading.Event, on_companion=None) -> None:
    cache = BundleCache(settings.bundle_dir)
    current = None
    events = None
    ui.status("Waiting for the League client (set league_path in settings if it's not in the default folder)")
    while not stop.is_set():
        lock = read_lockfile(settings.league_path)
        if lock is None:
            if current is not None:
                ui.status("League client closed; waiting for it to start")
                if events:
                    events.stop()
                current = events = None
            time.sleep(3)
            continue
        if current is not None and (lock.port, lock.password) == (current.port, current.password):
            time.sleep(3)
            continue
        # New client process (password changes every restart): reconnect.
        if events:
            events.stop()
        current = lock
        lcu = LcuClient(lock)
        try:
            failures = lcu.smoke_test()
        except Exception as e:
            failures = [str(e)]
        read_only = bool(failures)
        if read_only:
            ui.status("Client API changed or unavailable; read-only mode: " + "; ".join(failures)[:200])
        cache.clear()
        importer = Importer(lcu, settings, ask_page=ui.ask_page)
        companion = Companion(cache, ui, settings, importer=importer, read_only=read_only)
        if on_companion:
            on_companion(companion)
        events = LcuEvents(lock, companion.on_session)
        events.start()
        ui.status("Connected to the League client" + (" (auto-import off)" if not settings.auto_import else ""))
        try:
            session = lcu.champ_select_session()
            if session:
                companion.on_session("Update", session)
        except Exception as e:
            log.debug("no initial session: %s", e)
        time.sleep(3)


def run(settings_path: str | None = None, headless: bool = False) -> None:
    settings = Settings.load(settings_path)
    if not Path(settings.path).exists():
        settings.save()  # write defaults so users can find and edit them
    ui = make_ui(headless)
    threading.Thread(target=_sync, args=(settings, ui), daemon=True).start()
    stop = threading.Event()
    worker = threading.Thread(target=connect_loop, args=(settings, ui, stop), daemon=True)
    worker.start()
    try:
        if isinstance(ui, ConsoleUI):
            while worker.is_alive():
                worker.join(1.0)
        else:
            ui.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
