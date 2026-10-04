"""Write one rune page and one item set into the client.

Safety rules:
  * writes only happen during champ select;
  * only pages and sets the app created are touched (name/title prefix), or the
    one page slot the user explicitly gave the app when the page limit was hit;
  * the user's own pages are never deleted.
"""

from __future__ import annotations

import logging
from typing import Callable

from buildopt.lcu.connection import LcuClient, LcuError

log = logging.getLogger(__name__)


class ImportBlocked(RuntimeError):
    pass


class Importer:
    def __init__(self, lcu: LcuClient, settings, ask_page: Callable[[list[dict]], int | None] | None = None):
        """``settings`` needs ``page_prefix``, ``owned_page_id`` and a ``save()`` method."""
        self.lcu = lcu
        self.settings = settings
        self.ask_page = ask_page

    @property
    def prefix(self) -> str:
        return self.settings.page_prefix

    def ensure_champ_select(self) -> None:
        phase = self.lcu.gameflow_phase()
        if phase != "ChampSelect":
            raise ImportBlocked(f"not in champ select (phase {phase})")

    # ---- rune page ---------------------------------------------------------
    def is_ours(self, page: dict) -> bool:
        return page.get("name", "").startswith(self.prefix) or page.get("id") == self.settings.owned_page_id

    def write_rune_page(self, name: str, primary: int, sub: int, perks: list[int], shards: list[int]) -> dict:
        self.ensure_champ_select()
        log.info("writing rune page '%s %s'", self.prefix, name)
        body = {
            "name": f"{self.prefix} {name}"[:40],
            "primaryStyleId": primary,
            "subStyleId": sub,
            "selectedPerkIds": list(perks) + list(shards),
            "current": True,
        }
        pages = self.lcu.get("/lol-perks/v1/pages") or []
        ours = [p for p in pages if p.get("isEditable", True) and self.is_ours(p)]
        current = self.lcu.get("/lol-perks/v1/currentpage", ok_404=True)
        # Prefer the active page if it is ours, otherwise any page of ours.
        target = next((p for p in ours if current and p.get("id") == current.get("id")), ours[0] if ours else None)
        if target is not None:
            if target.get("id") == self.settings.owned_page_id and not target.get("name", "").startswith(self.prefix):
                # The user's granted slot: overwrite in place rather than deleting it.
                return self.lcu.put(f"/lol-perks/v1/pages/{target['id']}", {**body, "id": target["id"]}) or body
            self.lcu.delete(f"/lol-perks/v1/pages/{target['id']}")
        try:
            return self.lcu.post("/lol-perks/v1/pages", body)
        except LcuError as e:
            if target is not None or not self._page_limit_reached(pages):
                raise
            log.info("rune page limit reached (%s); asking which page the app may use", e)
        slot = self._granted_slot(pages)
        if slot is None:
            raise ImportBlocked("rune page limit reached and no page was granted to the app")
        return self.lcu.put(f"/lol-perks/v1/pages/{slot}", {**body, "id": slot}) or body

    def _page_limit_reached(self, pages: list[dict]) -> bool:
        try:
            inv = self.lcu.get("/lol-perks/v1/inventory")
            owned = inv.get("ownedPageCount")
        except LcuError:
            return True
        editable = sum(1 for p in pages if p.get("isEditable", True))
        return owned is None or editable >= owned

    def _granted_slot(self, pages: list[dict]) -> int | None:
        """Ask once which page the app may own; remember it from then on."""
        editable = [p for p in pages if p.get("isEditable", True)]
        if self.settings.owned_page_id and any(p["id"] == self.settings.owned_page_id for p in editable):
            return self.settings.owned_page_id
        if self.settings.asked_for_page or self.ask_page is None:
            return None
        self.settings.asked_for_page = True
        choice = self.ask_page(editable)
        self.settings.owned_page_id = choice
        self.settings.save()
        return choice

    # ---- item set ----------------------------------------------------------
    def write_item_set(self, item_set: dict) -> None:
        """Replace the app's own set for this champion, then read back to confirm the client saved it."""
        self.ensure_champ_select()
        summoner = self.lcu.current_summoner() or {}
        sid = summoner.get("summonerId") or summoner.get("accountId")
        if not sid:
            raise ImportFailed(f"the client didn't report a summoner id ({sorted(summoner)})")
        path = f"/lol-item-sets/v1/item-sets/{sid}/sets"
        data = self.lcu.get(path) or {}
        sets = data.get("itemSets", [])
        champs = set(item_set["associatedChampions"])
        keep = [s for s in sets if not (s.get("title", "").startswith(self.prefix) and set(s.get("associatedChampions", [])) == champs)]
        body = {**data, "accountId": data.get("accountId", summoner.get("accountId")), "itemSets": keep + [item_set]}
        log.info("writing item set '%s' (%d blocks) to %s", item_set["title"], len(item_set["blocks"]), path)
        self.lcu.put(path, body)
        saved = self.lcu.get(path) or {}
        if not any(s.get("title") == item_set["title"] for s in saved.get("itemSets", [])):
            raise ImportFailed("the client accepted the item set but it wasn't there when read back")
        log.info("item set saved")


class ImportFailed(RuntimeError):
    pass
