import pytest

from buildopt.pipeline.crawler import CrawlConfig, Crawler
from buildopt.pipeline.store import Store
from buildopt.riot.api import RateLimiter, RiotApiError

from test_parse import _match, _timeline


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_rate_limiter_respects_both_windows():
    clock = FakeClock()
    rl = RateLimiter([(20, 1.0), (100, 120.0)], clock=clock, sleep=clock.sleep)
    for _ in range(100):
        rl.acquire()
    # 100 calls need at least 4 full seconds under 20/s
    assert clock.t >= 4.0
    rl.acquire()  # 101st call must wait for the 2-minute window
    assert clock.t >= 120.0


def test_rate_limiter_parses_header():
    rl = RateLimiter()
    rl.set_limits("50:10,3000:600")
    assert rl.limits == [(50, 10.0), (3000, 600.0)]


class FakeRiot:
    def __init__(self):
        self.calls = []

    def apex_league(self, platform, tier):
        self.calls.append(("apex", tier))
        return [{"puuid": f"{tier}-1"}] if tier == "CHALLENGER" else []

    def league_entries(self, platform, tier, division, page):
        self.calls.append(("entries", tier, division, page))
        return []

    def match_ids(self, platform, puuid, start=0, count=100, start_time=None):
        self.calls.append(("ids", puuid, start))
        return ["M_NEW", "M_OLD", "M_NEVER"] if start == 0 else []

    def match(self, platform, match_id):
        self.calls.append(("match", match_id))
        m = _match()
        m["metadata"]["matchId"] = match_id
        if match_id == "M_OLD":
            m["info"]["gameVersion"] = "16.17.1"
        return m

    def timeline(self, platform, match_id):
        self.calls.append(("timeline", match_id))
        return _timeline([])


def test_crawler_cascade_and_patch_cutoff(static, tmp_path):
    store = Store(tmp_path / "c.sqlite")
    riot = FakeRiot()
    cfg = CrawlConfig(platforms=["euw1"], patches=["16.19", "16.18"], targets={(233, "JUNGLE")}, lowest=("DIAMOND", "I"))
    crawler = Crawler(riot, store, static, cfg)
    crawler._run_platform("euw1")
    apex = [c[1] for c in riot.calls if c[0] == "apex"]
    assert apex == ["CHALLENGER", "GRANDMASTER", "MASTER"]
    assert ("entries", "DIAMOND", "I", 1) in riot.calls
    assert not any(c[0] == "entries" and c[2] == "II" for c in riot.calls)  # stopped at the lowest step
    fetched = [c[1] for c in riot.calls if c[0] == "match"]
    assert fetched == ["M_NEW", "M_OLD"]  # older patch ends the player's history
    assert [c[1] for c in riot.calls if c[0] == "timeline"] == ["M_NEW"]
    assert store.count_rows(233, "JUNGLE") == 1
    assert next(store.iter_rows(233))["tier"] == "CHALLENGER"
    # resumable: a second run fetches nothing new
    riot.calls.clear()
    crawler._run_platform("euw1")
    assert not any(c[0] == "match" for c in riot.calls)


def test_riot_client_retries_on_429(monkeypatch):
    from buildopt.riot.api import RiotClient

    class Resp:
        def __init__(self, status, body=None, headers=None):
            self.status_code, self._body, self.headers, self.text = status, body, headers or {}, ""

        def json(self):
            return self._body

    responses = [Resp(429, headers={"Retry-After": "0"}), Resp(200, ["X"])]

    class Session:
        def get(self, *a, **kw):
            return responses.pop(0)

    client = RiotClient("key", session=Session(), sleep=lambda s: None)
    assert client.get("euw1", "/x") == ["X"]
    with pytest.raises(RiotApiError):
        responses.append(Resp(403))
        client.get("euw1", "/x")
