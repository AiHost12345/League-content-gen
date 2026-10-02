"""Riot Games API client with per-region rate limiting.

Personal keys allow 20 requests / 1 s and 100 requests / 2 min per routing
value. Limits are read from the X-App-Rate-Limit header when present, and a
429 honours Retry-After.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

import requests

log = logging.getLogger(__name__)

PLATFORM_TO_REGION = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe", "me1": "europe",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "sg2": "sea", "tw2": "sea", "vn2": "sea",
}

RANKED_SOLO = "RANKED_SOLO_5x5"
RANKED_SOLO_QUEUE_ID = 420
DEFAULT_LIMITS = ((20, 1.0), (100, 120.0))


class RiotApiError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"Riot API {status} for {url}: {body[:200]}")
        self.status = status


class RateLimiter:
    """Sliding-window limiter supporting several (count, seconds) windows."""

    def __init__(self, limits=DEFAULT_LIMITS, clock: Callable[[], float] = time.monotonic, sleep=time.sleep):
        self.limits = list(limits)
        self.clock = clock
        self.sleep = sleep
        self.calls: deque[float] = deque()
        self.blocked_until = 0.0
        self.lock = threading.Lock()

    def set_limits(self, header: str) -> None:
        """Parse an X-App-Rate-Limit header like '20:1,100:120'."""
        try:
            parsed = [(int(c), float(s)) for c, s in (part.split(":") for part in header.split(","))]
        except ValueError:
            return
        if parsed:
            self.limits = parsed

    def block(self, seconds: float) -> None:
        with self.lock:
            self.blocked_until = max(self.blocked_until, self.clock() + seconds)

    def wait_time(self) -> float:
        now = self.clock()
        longest = max(s for _, s in self.limits)
        while self.calls and now - self.calls[0] > longest:
            self.calls.popleft()
        wait = max(0.0, self.blocked_until - now)
        for count, seconds in self.limits:
            recent = [t for t in self.calls if now - t < seconds]
            if len(recent) >= count:
                wait = max(wait, seconds - (now - recent[-count]) + 0.01)
        return wait

    def acquire(self) -> None:
        while True:
            with self.lock:
                wait = self.wait_time()
                if wait <= 0:
                    self.calls.append(self.clock())
                    return
            self.sleep(wait)


class RiotClient:
    def __init__(self, api_key: str, session: requests.Session | None = None, max_retries: int = 5, sleep=time.sleep):
        if not api_key:
            raise ValueError("A Riot API key is required (set RIOT_API_KEY).")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.sleep = sleep
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def limiter(self, host: str) -> RateLimiter:
        with self._lock:
            if host not in self._limiters:
                self._limiters[host] = RateLimiter(sleep=self.sleep)
            return self._limiters[host]

    def get(self, host: str, path: str, params: dict | None = None):
        url = f"https://{host}.api.riotgames.com{path}"
        limiter = self.limiter(host)
        for attempt in range(self.max_retries + 1):
            limiter.acquire()
            try:
                resp = self.session.get(url, params=params, headers={"X-Riot-Token": self.api_key}, timeout=30)
            except requests.RequestException as e:
                log.warning("network error %s (attempt %d)", e, attempt)
                self.sleep(2 ** attempt)
                continue
            if "X-App-Rate-Limit" in resp.headers:
                limiter.set_limits(resp.headers["X-App-Rate-Limit"])
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                retry = float(resp.headers.get("Retry-After", 2 ** attempt))
                log.info("429 on %s, sleeping %.0fs", host, retry)
                limiter.block(retry)
                continue
            if resp.status_code in (500, 502, 503, 504):
                self.sleep(2 ** attempt)
                continue
            raise RiotApiError(resp.status_code, url, resp.text)
        raise RiotApiError(-1, url, "retries exhausted")

    # ---- league-v4 -----------------------------------------------------
    def apex_league(self, platform: str, tier: str) -> list[dict]:
        name = {"CHALLENGER": "challengerleagues", "GRANDMASTER": "grandmasterleagues", "MASTER": "masterleagues"}[tier]
        data = self.get(platform, f"/lol/league/v4/{name}/by-queue/{RANKED_SOLO}")
        return data.get("entries", [])

    def league_entries(self, platform: str, tier: str, division: str, page: int) -> list[dict]:
        return self.get(platform, f"/lol/league/v4/entries/{RANKED_SOLO}/{tier}/{division}", {"page": page})

    # ---- match-v5 ------------------------------------------------------
    def match_ids(self, platform: str, puuid: str, start: int = 0, count: int = 100, start_time: int | None = None) -> list[str]:
        params = {"queue": RANKED_SOLO_QUEUE_ID, "type": "ranked", "start": start, "count": count}
        if start_time:
            params["startTime"] = start_time
        return self.get(PLATFORM_TO_REGION[platform], f"/lol/match/v5/matches/by-puuid/{puuid}/ids", params)

    def match(self, platform: str, match_id: str) -> dict:
        return self.get(PLATFORM_TO_REGION[platform], f"/lol/match/v5/matches/{match_id}")

    def timeline(self, platform: str, match_id: str) -> dict:
        return self.get(PLATFORM_TO_REGION[platform], f"/lol/match/v5/matches/{match_id}/timeline")
