"""Connection to the League client's local API (LCU), per Hextech Docs.

* Read the lockfile: ``name:pid:port:password:protocol``. The password changes
  every client restart.
* HTTPS to 127.0.0.1:<port> with Basic auth (user ``riot``).
* The client uses a self-signed certificate. We trust Riot's root certificate
  (shipped as riotgames.pem) instead of turning verification off. The
  certificate is not issued for 127.0.0.1, so only the hostname check is
  disabled; the chain is still verified.
* A websocket on the same port delivers champ select updates as events.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import platform
import ssl
import threading
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Callable

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger(__name__)

CHAMP_SELECT_EVENT = "OnJsonApiEvent_lol-champ-select_v1_session"
DEFAULT_INSTALL_PATHS = {
    "Windows": [r"C:\Riot Games\League of Legends", r"D:\Riot Games\League of Legends"],
    "Darwin": ["/Applications/League of Legends.app/Contents/LoL"],
    "Linux": [os.path.expanduser("~/Games/league-of-legends/drive_c/Riot Games/League of Legends")],
}


class LcuError(RuntimeError):
    def __init__(self, status: int, method: str, path: str, body: str = ""):
        super().__init__(f"LCU {method} {path} -> {status}: {body[:200]}")
        self.status = status


@dataclass
class Lockfile:
    name: str
    pid: int
    port: int
    password: str
    protocol: str

    @classmethod
    def parse(cls, text: str) -> "Lockfile":
        name, pid, port, password, protocol = text.strip().split(":")
        return cls(name, int(pid), int(port), password, protocol)


def find_lockfile(install_path: str | None = None) -> Path | None:
    candidates = []
    if install_path:
        candidates.append(install_path)
    if os.environ.get("LEAGUE_PATH"):
        candidates.append(os.environ["LEAGUE_PATH"])
    candidates += DEFAULT_INSTALL_PATHS.get(platform.system(), [])
    for c in candidates:
        p = Path(c) / "lockfile"
        if p.exists():
            return p
    return None


def read_lockfile(install_path: str | None = None) -> Lockfile | None:
    p = find_lockfile(install_path)
    if p is None:
        return None
    try:
        return Lockfile.parse(p.read_text())
    except (OSError, ValueError):
        return None


def riot_pem_path() -> str:
    return str(resources.files("buildopt.lcu").joinpath("riotgames.pem"))


def riot_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.load_verify_locations(cafile=riot_pem_path())
    ctx.check_hostname = False  # cert is not issued for 127.0.0.1; chain is still verified
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


class _RiotAdapter(HTTPAdapter):
    def __init__(self, ctx: ssl.SSLContext, **kw):
        self._ctx = ctx
        super().__init__(**kw)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ctx
        kwargs["assert_hostname"] = False
        return super().init_poolmanager(*args, **kwargs)


class LcuClient:
    """Thin wrapper over the LCU REST API. Keep endpoint knowledge in one place."""

    def __init__(self, lock: Lockfile, session: requests.Session | None = None):
        self.lock = lock
        self.base = f"https://127.0.0.1:{lock.port}"
        self.auth = ("riot", lock.password)
        if session is None:
            session = requests.Session()
            session.mount("https://127.0.0.1", _RiotAdapter(riot_ssl_context()))
        self.session = session

    def request(self, method: str, path: str, body=None, ok_404: bool = False):
        resp = self.session.request(method, self.base + path, auth=self.auth, json=body, timeout=10)
        if resp.status_code == 404 and ok_404:
            return None
        if resp.status_code >= 400:
            raise LcuError(resp.status_code, method, path, resp.text)
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    def get(self, path: str, ok_404: bool = False):
        return self.request("GET", path, ok_404=ok_404)

    def post(self, path: str, body=None):
        return self.request("POST", path, body)

    def put(self, path: str, body=None):
        return self.request("PUT", path, body)

    def delete(self, path: str):
        return self.request("DELETE", path)

    # ---- endpoints used by the app -----------------------------------------
    def gameflow_phase(self) -> str:
        return self.get("/lol-gameflow/v1/gameflow-phase") or "None"

    def champ_select_session(self) -> dict | None:
        return self.get("/lol-champ-select/v1/session", ok_404=True)

    def current_summoner(self) -> dict:
        return self.get("/lol-summoner/v1/current-summoner")

    def smoke_test(self) -> list[str]:
        """Check every endpoint the app relies on. Returns the failures (empty = OK)."""
        failures = []
        checks = ["/lol-gameflow/v1/gameflow-phase", "/lol-perks/v1/pages", "/lol-perks/v1/currentpage",
                  "/lol-perks/v1/inventory", "/lol-summoner/v1/current-summoner"]
        for path in checks:
            try:
                self.get(path, ok_404=path.endswith("currentpage"))
            except (LcuError, requests.RequestException) as e:
                failures.append(f"{path}: {e}")
        try:
            sid = self.current_summoner()["summonerId"]
            self.get(f"/lol-item-sets/v1/item-sets/{sid}/sets")
        except (LcuError, requests.RequestException, KeyError, TypeError) as e:
            failures.append(f"item sets: {e}")
        return failures


class LcuEvents:
    """Websocket subscription to champ select session events (no polling)."""

    def __init__(self, lock: Lockfile, on_session: Callable[[str, dict | None], None],
                 on_close: Callable[[], None] | None = None):
        self.lock = lock
        self.on_session = on_session
        self.on_close = on_close
        self._ws = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _url_and_headers(self):
        token = base64.b64encode(f"riot:{self.lock.password}".encode()).decode()
        return f"wss://127.0.0.1:{self.lock.port}/", [f"Authorization: Basic {token}"]

    def handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            return
        # WAMP event: [8, topic, payload]
        if isinstance(msg, list) and len(msg) == 3 and msg[0] == 8 and msg[1] == CHAMP_SELECT_EVENT:
            payload = msg[2] or {}
            self.on_session(payload.get("eventType", "Update"), payload.get("data"))

    def start(self) -> None:
        import websocket  # websocket-client

        url, headers = self._url_and_headers()
        sslopt = {"cert_reqs": ssl.CERT_REQUIRED, "ca_certs": riot_pem_path(), "check_hostname": False}

        def on_open(ws):
            ws.send(json.dumps([5, CHAMP_SELECT_EVENT]))

        def run():
            while not self._stop.is_set():
                self._ws = websocket.WebSocketApp(
                    url, header=headers, on_open=on_open, on_message=lambda ws, m: self.handle_message(m),
                    on_error=lambda ws, e: log.debug("ws error: %s", e),
                )
                self._ws.run_forever(sslopt=sslopt)
                if self._stop.is_set():
                    break
                if self.on_close:
                    self.on_close()
                time.sleep(2)

        self._thread = threading.Thread(target=run, name="lcu-events", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            self._ws.close()
