from buildopt.pipeline.parse import parse_match, patch_of, resolve_items

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def _participant(pid, champ, role, team, win):
    return {
        "participantId": pid, "puuid": f"puuid{pid}", "championId": champ, "teamPosition": role, "teamId": team,
        "win": win, **{f"item{i}": 0 for i in range(7)},
        "perks": {
            "statPerks": {"offense": 5008, "flex": 5008, "defense": 5011},
            "styles": [
                {"description": "primaryStyle", "style": 8000,
                 "selections": [{"perk": 8010}, {"perk": 9111}, {"perk": 9104}, {"perk": 8014}]},
                {"description": "subStyle", "style": 8100, "selections": [{"perk": 8139}, {"perk": 8135}]},
            ],
        },
        "totalDamageDealtToChampions": 20000, "magicDamageDealtToChampions": 5000,
        "physicalDamageDealtToChampions": 15000, "trueDamageDealtToChampions": 0, "totalHeal": 8000,
        "timeCCingOthers": 30, "totalDamageTaken": 25000, "damageSelfMitigated": 20000,
    }


def _match():
    parts = [_participant(i + 1, 233 if i == 1 else 100 + i, ROLES[i % 5], 100 if i < 5 else 200, i < 5) for i in range(10)]
    return {"metadata": {"matchId": "EUW1_1"}, "info": {
        "gameVersion": "16.19.705.1234", "gameDuration": 1800, "gameStartTimestamp": 0, "queueId": 420,
        "participants": parts}}


def _ev(t, typ, pid=2, **kw):
    return {"timestamp": t, "type": typ, "participantId": pid, **kw}


def _timeline(events):
    frames = []
    for m in range(31):
        pf = {str(p): {"totalGold": 500 + m * (400 if p == 2 else 350), "position": {"x": 3800, "y": 7900}} for p in range(1, 11)}
        frames.append({"timestamp": m * 60000, "participantFrames": pf, "events": events if m == 0 else []})
    return {"info": {"frames": frames}}


def test_patch_of():
    assert patch_of("16.19.705.1234") == "16.19"


def test_resolve_items_handles_undo_sell_and_components(static):
    events = [
        _ev(5000, "ITEM_PURCHASED", itemId=1101),  # pet start
        _ev(6000, "ITEM_PURCHASED", itemId=2003),
        _ev(400000, "ITEM_PURCHASED", itemId=1036),  # component, ignored
        _ev(420000, "ITEM_PURCHASED", itemId=3748),  # Titanic...
        _ev(425000, "ITEM_UNDO", beforeId=3748, afterId=0),  # ...undone
        _ev(430000, "ITEM_PURCHASED", itemId=6676),  # Collector
        _ev(500000, "ITEM_PURCHASED", itemId=3047),  # Steelcaps
        _ev(800000, "ITEM_PURCHASED", itemId=3071),  # Black Cleaver
        _ev(900000, "ITEM_PURCHASED", itemId=6333),  # Death's Dance, flipped quickly
        _ev(960000, "ITEM_SOLD", itemId=6333),
        _ev(1000000, "ITEM_PURCHASED", itemId=3053),  # Sterak's
        _ev(1000000, "ITEM_PURCHASED", pid=3, itemId=3031),  # someone else
    ]
    seq, start = resolve_items(events, 2, static)
    assert [c["id"] for c in seq] == [6676, 3047, 3071, 3053]
    assert start == [1101, 2003]


def test_parse_match_builds_rows(static):
    events = [_ev(430000, "ITEM_PURCHASED", itemId=6676), _ev(800000, "ITEM_PURCHASED", itemId=3071)]
    rows = parse_match(_match(), _timeline(events), static, "MASTER", "euw1")
    assert len(rows) == 10
    briar = next(r for r in rows if r["champion_id"] == 233)
    assert briar["role"] == "JUNGLE" and briar["win"] and briar["tier"] == "MASTER"
    assert briar["keystone"] == 8010 and briar["perks"] == [8010, 9111, 9104, 8014, 8139, 8135]
    assert briar["shards"] == [5008, 5008, 5011]
    assert [i["id"] for i in briar["items"]] == [6676, 3071]
    # Briar gains 400/min vs the enemy jungler's 350/min: 50 gold/min lead at ~7.2 min.
    assert briar["items"][0]["lane_gd"] == round(50 * 7.1667)
    assert briar["items"][0]["min"] == 7.17
    assert len(briar["enemies"]) == 5 and {e["role"] for e in briar["enemies"]} == set(ROLES)
    assert len(briar["positions"]) == 6


def test_parse_match_without_timeline(static):
    rows = parse_match(_match(), None, static, "DIAMOND", "euw1")
    assert len(rows) == 10 and not rows[0]["has_timeline"] and rows[0]["items"] == []
