"""Client integration against a fake LCU: safety rules and the champ select flow."""

import json

import pytest

from buildopt.app.bundles import BundleCache, sync_bundles
from buildopt.app.companion import Companion
from buildopt.app.settings import Settings
from buildopt.bundle import save_bundle
from buildopt.lcu.champselect import parse_session
from buildopt.lcu.connection import CHAMP_SELECT_EVENT, LcuError, LcuEvents, Lockfile
from buildopt.lcu.importer import ImportBlocked, Importer


class FakeLcu:
    def __init__(self, pages=None, owned=5, phase="ChampSelect"):
        self.phase = phase
        self.owned = owned
        self.pages = pages if pages is not None else [{"id": 1, "name": "My page", "isEditable": True}]
        self.current = self.pages[0]["id"] if self.pages else None
        self.sets = {"accountId": 9, "itemSets": [{"title": "My Briar set", "associatedChampions": [233]}], "timestamp": 1}
        self.log = []
        self.next_id = 100

    def gameflow_phase(self):
        return self.phase

    def current_summoner(self):
        return {"summonerId": 42, "accountId": 9}

    def get(self, path, ok_404=False):
        if path == "/lol-perks/v1/pages":
            return [dict(p) for p in self.pages]
        if path == "/lol-perks/v1/currentpage":
            return next((dict(p) for p in self.pages if p["id"] == self.current), None)
        if path == "/lol-perks/v1/inventory":
            return {"ownedPageCount": self.owned}
        if path == "/lol-item-sets/v1/item-sets/42/sets":
            return json.loads(json.dumps(self.sets))
        raise LcuError(404, "GET", path)

    def post(self, path, body):
        self.log.append(("POST", path))
        if len([p for p in self.pages if p.get("isEditable", True)]) >= self.owned:
            raise LcuError(400, "POST", path, "Max pages reached")
        page = {**body, "id": self.next_id, "isEditable": True}
        self.next_id += 1
        self.pages.append(page)
        self.current = page["id"]
        return page

    def put(self, path, body):
        self.log.append(("PUT", path))
        if path.startswith("/lol-perks/v1/pages/"):
            pid = int(path.rsplit("/", 1)[1])
            for p in self.pages:
                if p["id"] == pid:
                    p.update(body)
            self.current = pid
        else:
            self.sets = body
        return None

    def delete(self, path):
        self.log.append(("DELETE", path))
        pid = int(path.rsplit("/", 1)[1])
        self.pages = [p for p in self.pages if p["id"] != pid]


def settings(tmp_path, **kw):
    s = Settings(path=str(tmp_path / "settings.json"), **kw)
    return s


PAGE = (8000, 8100, [8010, 9111, 9104, 8014, 8139, 8135], [5008, 5008, 5011])


def test_rune_page_never_touches_user_pages(tmp_path):
    lcu = FakeLcu()
    imp = Importer(lcu, settings(tmp_path))
    imp.write_rune_page("Briar Conqueror", *PAGE)
    imp.write_rune_page("Briar Lethal Tempo", 8000, 8100, [8008, 9111, 9104, 8014, 8139, 8135], [5008, 5008, 5011])
    names = [p["name"] for p in lcu.pages]
    assert names == ["My page", "BO: Briar Lethal Tempo"]
    assert ("DELETE", "/lol-perks/v1/pages/1") not in lcu.log
    created = lcu.pages[-1]
    assert created["selectedPerkIds"] == [8008, 9111, 9104, 8014, 8139, 8135, 5008, 5008, 5011]
    assert created["primaryStyleId"] == 8000 and created["subStyleId"] == 8100 and created["current"]


def test_page_limit_asks_once_and_reuses_slot(tmp_path):
    pages = [{"id": i, "name": f"User {i}", "isEditable": True} for i in (1, 2)]
    lcu = FakeLcu(pages=pages, owned=2)
    asked = []
    s = settings(tmp_path)
    imp = Importer(lcu, s, ask_page=lambda ps: asked.append(ps) or 2)
    imp.write_rune_page("Briar Conqueror", *PAGE)
    assert len(asked) == 1 and s.owned_page_id == 2
    assert [p["name"] for p in lcu.pages] == ["User 1", "BO: Briar Conqueror"]
    assert json.loads((tmp_path / "settings.json").read_text())["owned_page_id"] == 2
    imp.write_rune_page("Briar Lethal Tempo", *PAGE)  # now ours by prefix: delete + recreate, no new question
    assert len(asked) == 1
    assert [p["name"] for p in lcu.pages] == ["User 1", "BO: Briar Lethal Tempo"]


def test_page_limit_declined(tmp_path):
    lcu = FakeLcu(pages=[{"id": 1, "name": "User", "isEditable": True}], owned=1)
    imp = Importer(lcu, settings(tmp_path), ask_page=lambda ps: None)
    with pytest.raises(ImportBlocked):
        imp.write_rune_page("x", *PAGE)
    assert lcu.pages == [{"id": 1, "name": "User", "isEditable": True}]


def test_item_set_replaces_only_own_set(tmp_path):
    lcu = FakeLcu()
    imp = Importer(lcu, settings(tmp_path))
    for title in ("BO: Briar 16.19 #aaaaaa", "BO: Briar 16.19 #bbbbbb"):
        imp.write_item_set({"title": title, "associatedChampions": [233], "blocks": []})
    titles = [s["title"] for s in lcu.sets["itemSets"]]
    assert titles == ["My Briar set", "BO: Briar 16.19 #bbbbbb"]
    assert lcu.sets["accountId"] == 9


def test_writes_blocked_outside_champ_select(tmp_path):
    lcu = FakeLcu(phase="InProgress")
    imp = Importer(lcu, settings(tmp_path))
    with pytest.raises(ImportBlocked):
        imp.write_rune_page("x", *PAGE)
    with pytest.raises(ImportBlocked):
        imp.write_item_set({"title": "BO: x", "associatedChampions": [233]})
    assert lcu.log == []


def test_lockfile_and_events():
    lock = Lockfile.parse("LeagueClient:1234:54321:s3cret:https")
    assert lock.port == 54321 and lock.password == "s3cret"
    seen = []
    ev = LcuEvents(lock, lambda t, d: seen.append((t, d)))
    ev.handle_message(json.dumps([8, CHAMP_SELECT_EVENT, {"eventType": "Update", "data": {"x": 1}}]))
    ev.handle_message(json.dumps([8, "OnJsonApiEvent_other", {"data": {}}]))
    ev.handle_message("not json")
    assert seen == [("Update", {"x": 1})]


# ---- champ select flow --------------------------------------------------------

def session(my_champ, locked, enemies, phase="BAN_PICK", time_left=30000):
    return {
        "localPlayerCellId": 1,
        "myTeam": [{"cellId": 1, "championId": my_champ if locked else 0, "championPickIntent": my_champ,
                    "assignedPosition": "jungle"},
                   {"cellId": 2, "championId": 498, "assignedPosition": "bottom"}],
        "theirTeam": [{"cellId": 5 + i, "championId": c} for i, c in enumerate(enemies)],
        "actions": [[{"actorCellId": 1, "type": "pick", "championId": my_champ, "completed": locked}]],
        "timer": {"phase": phase, "adjustedTimeLeftInPhase": time_left},
    }


class RecUI:
    def __init__(self):
        self.views, self.statuses = [], []

    def show(self, view):
        self.views.append(view)

    def status(self, msg):
        self.statuses.append(msg)

    def ask_page(self, pages):
        return None


@pytest.fixture
def cache(bundle, tmp_path):
    src = tmp_path / "server"
    save_bundle(bundle, src)
    dest = tmp_path / "client"
    assert sync_bundles(str(src), dest) == ["233_JUNGLE"]
    assert sync_bundles(str(src), dest) == []  # already current
    return BundleCache(dest)


def test_parse_session():
    st = parse_session(session(233, False, [516, 113]))
    assert st.my_champion == 233 and not st.my_locked and st.my_role == "JUNGLE"
    assert st.enemies == [516, 113] and st.allies == {"BOTTOM": 498}


def test_companion_flow(cache, static, tmp_path):
    lcu = FakeLcu()
    ui = RecUI()
    s = settings(tmp_path, final_write_lead_ms=10**9)
    comp = Companion(cache, ui, s, importer=Importer(lcu, s))
    tanky = [static.champion_id(n) for n in ("Ornn", "Sejuani", "Galio", "Ashe", "Braum")]
    squishy = [static.champion_id(n) for n in ("Teemo", "Kha'Zix", "Zed", "Jinx", "Lulu")]

    comp.on_session("Update", session(233, False, tanky[:2]))  # hovering: score, but don't write
    assert ui.views and lcu.log == []
    assert "data 16.19" in ui.views[-1]["header"]

    comp.on_session("Update", session(233, True, tanky))  # locked: write both
    assert ("POST", "/lol-perks/v1/pages") in lcu.log
    assert any(t["title"].startswith("BO: Briar") for t in lcu.sets["itemSets"])
    assert ui.views[-1]["build"].split(" → ")[0] in ("Titanic Hydra", "Blade of The Ruined King")
    writes = len(lcu.log)

    comp.on_session("Update", session(233, True, tanky))  # nothing changed: no rewrite
    assert len(lcu.log) == writes

    comp.on_session("Update", session(233, True, squishy, phase="FINALIZATION", time_left=5000))  # top loadout flips
    assert len(lcu.log) > writes
    assert ui.views[-1]["change"].split(":")[0].endswith("→ The Collector")
    assert len([p for p in lcu.pages if p["name"].startswith("BO:")]) == 1
    assert len([t for t in lcu.sets["itemSets"] if t["title"].startswith("BO:")]) == 1
    assert comp.final_timer is not None
    comp.final_timer.cancel()

    comp.on_session("Delete", None)
    assert comp.rec is None and ui.statuses[-1] == "Waiting for champ select"


def test_companion_read_only(cache, static, tmp_path):
    lcu = FakeLcu()
    ui = RecUI()
    s = settings(tmp_path, auto_import=False)
    comp = Companion(cache, ui, s, importer=Importer(lcu, s))
    comp.on_session("Update", session(233, True, [static.champion_id("Ornn")]))
    assert ui.views and ui.views[-1]["read_only"] and lcu.log == []


def test_companion_without_data(cache, tmp_path):
    ui = RecUI()
    comp = Companion(cache, ui, settings(tmp_path))
    comp.on_session("Update", session(99, True, []))
    assert ui.statuses[-1].startswith("No recommendations for Lux jungle yet")


def test_item_set_written_even_when_rune_page_fails(cache, static, tmp_path):
    """Regression: a rune page failure used to skip the item set silently."""
    lcu = FakeLcu(pages=[{"id": 1, "name": "User", "isEditable": True}], owned=1)  # rune pages full
    ui = RecUI()
    results = []
    ui.import_result = results.append
    s = settings(tmp_path)
    comp = Companion(cache, ui, s, importer=Importer(lcu, s, ask_page=lambda pages: None))  # user declines
    comp.on_session("Update", session(233, True, [static.champion_id("Ornn")]))
    assert any(t["title"].startswith("BO: Briar") for t in lcu.sets["itemSets"])
    assert results[-1]["items"][0] and not results[-1]["runes"][0]
    assert "rune page limit" in results[-1]["runes"][1]
    assert ui.statuses[-1].startswith("Runes FAILED, item set written")


def test_item_set_not_saved_is_reported(cache, static, tmp_path):
    class DroppingLcu(FakeLcu):
        def put(self, path, body):
            if "item-sets" in path:
                self.log.append(("PUT", path))
                return None  # accepted, but nothing stored
            return super().put(path, body)

    lcu = DroppingLcu()
    ui = RecUI()
    results = []
    ui.import_result = results.append
    s = settings(tmp_path)
    comp = Companion(cache, ui, s, importer=Importer(lcu, s))
    comp.on_session("Update", session(233, True, [static.champion_id("Ornn")]))
    ok, msg = results[-1]["items"]
    assert not ok and "wasn't there when read back" in msg
    assert results[-1]["runes"][0]
