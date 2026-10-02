"""Command line interface: ``buildopt <command>``."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

from buildopt import ROLES


def _static(args):
    from buildopt import ddragon

    return ddragon.load(cache_dir=Path(args.db).parent / "ddragon" if getattr(args, "db", None) else None,
                        offline=getattr(args, "offline", False))


def _dataset_cfg(args):
    from buildopt.analysis.dataset import DatasetConfig

    return DatasetConfig(
        patch=args.patch,
        prev_weight=args.prev_weight,
        min_current_games=args.min_current_games,
        changed_items={int(x) for x in args.changed_items.split(",") if x} if args.changed_items else set(),
        changed_runes={int(x) for x in args.changed_runes.split(",") if x} if args.changed_runes else set(),
        min_tier=args.min_tier,
    )


def _games(args, static):
    from buildopt.analysis.dataset import load_games
    from buildopt.analysis.profiles import build_profiles
    from buildopt.pipeline.store import Store

    store = Store(args.db)
    champ = static.champion_id(args.champion)
    profiles = build_profiles(store.iter_rows(patches=store.patches()[-2:]), static)
    games, info = load_games(store, champ, args.role, profiles, _dataset_cfg(args))
    if not games:
        sys.exit(f"No games for {args.champion} {args.role} in {args.db}")
    return games, info, champ


def _path(arg: str, static) -> tuple[int, ...]:
    return tuple(static.item_id(x.strip()) for x in arg.replace("→", ",").replace(">", ",").split(",") if x.strip())


def _names(path, static) -> str:
    return " → ".join(static.item_name(i) for i in path)


def _version_line(info: dict) -> str:
    prev = f" + {info['previous_patch']}×{info['previous_weight']:g}" if info.get("previous_patch") else ""
    return f"data: patch {info['patch']}{prev} · {info['games_used']:,} games"


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ---- commands ----------------------------------------------------------------

def cmd_crawl(args):
    from buildopt.pipeline.crawler import CrawlConfig, Crawler
    from buildopt.pipeline.store import Store
    from buildopt.riot.api import RiotClient

    static = _static(args)
    key = args.api_key or os.environ.get("RIOT_API_KEY")
    targets = set()
    for spec in args.target or []:
        name, _, role = spec.partition(":")
        targets.add((static.champion_id(name), role.upper()))
    since = int(dt.datetime.fromisoformat(args.since).replace(tzinfo=dt.timezone.utc).timestamp()) if args.since else None
    cfg = CrawlConfig(platforms=args.platform, patches=args.patch, since=since, targets=targets,
                      max_target_rows=args.max_games, max_pages_per_division=args.max_pages,
                      lowest=tuple(args.lowest.split(":")))
    crawler = Crawler(RiotClient(key), Store(args.db), static, cfg)
    crawler.run()
    print(f"target rows stored: {crawler.target_rows():,}")


def cmd_synth(args):
    from buildopt.pipeline.store import Store
    from buildopt.pipeline.synthetic import generate

    static = _static(args)
    n = generate(Store(args.db), static, n_matches=args.matches, seed=args.seed)
    print(f"wrote {n:,} synthetic rows to {args.db}")


def cmd_status(args):
    from buildopt.pipeline.store import Store

    store = Store(args.db)
    static = _static(args)
    print("patches:", ", ".join(store.patches()) or "none")
    for champ, role, n in store.champion_roles(min_games=args.min_games)[:30]:
        print(f"  {static.champion_name(champ):<14} {role:<8} {n:>8,} games with timeline")


def cmd_bundle(args):
    from buildopt.bundle import build_bundle, data_version_label, save_bundle
    from buildopt.pipeline.store import Store

    static = _static(args)
    store = Store(args.db)
    champ = static.champion_id(args.champion)
    bundle = build_bundle(store, static, champ, args.role, _dataset_cfg(args), min_games=args.min_games)
    path = save_bundle(bundle, args.out)
    print(f"wrote {path} · {data_version_label(bundle)}")
    print(f"  {len(bundle['model']['paths'])} candidate paths × {len(bundle['model']['pages'])} rune pages, "
          f"{len(bundle['model']['pairs'])} path×page terms, {len(bundle['model']['enemy_terms'])} per-enemy terms")


def cmd_tree(args):
    from buildopt.analysis.paths import prefix_tree

    static = _static(args)
    games, info, _ = _games(args, static)
    tree = prefix_tree(games, args.depth, args.min_games)
    if args.json:
        return _print_json({"data_version": info, "tree": tree})
    print(_version_line(info))

    def show(node, depth):
        name = static.item_name(node["items"][-1])
        print(f"{'  ' * depth}{name:<28} {node['win_rate']:.1%}  [{node['lo']:.1%}–{node['hi']:.1%}]  "
              f"{node['games']:>6} games  {node['confidence']}")
        for ch in node["children"]:
            show(ch, depth + 1)

    for root in tree:
        show(root, 0)


def cmd_pairs(args):
    from buildopt.analysis.paths import pair_synergy

    static = _static(args)
    games, info, _ = _games(args, static)
    pairs = pair_synergy(games, args.within, args.min_games)
    if args.json:
        return _print_json({"data_version": info, "pairs": pairs})
    print(_version_line(info))
    print(f"{'pair':<48} {'lift':>7} {'games':>7}")
    for p in pairs[: args.top]:
        label = f"{static.item_name(p['a'])} + {static.item_name(p['b'])}"
        print(f"{label:<48} {p['lift']:>+7.3f} {p['games']:>7}")


def cmd_compare(args):
    from buildopt.analysis.paths import compare_paths, stability
    from buildopt.analysis.traits import tercile_levels

    static = _static(args)
    games, info, _ = _games(args, static)
    a, b = _path(args.a, static), _path(args.b, static)
    levels = tercile_levels([g.c_raw for g in games])
    res = compare_paths(games, a, b, levels, args.min_games, adjust=not args.no_adjust)
    out = {"data_version": info, **res}
    if args.stability:
        st = stability(games, a, b, levels, args.min_games)
        out["stability"] = {k: v for k, v in st.items() if k != "runs"}
    if args.bundle and args.enemies:
        from buildopt.bundle import load_bundle
        from buildopt.scoring import Scorer

        sc = Scorer(load_bundle(args.bundle))
        enemies = [static.champion_id(e.strip()) for e in args.enemies.split(",")]
        la, lb = sc.prefix_score(a, enemies), sc.prefix_score(b, enemies)
        out["this_comp"] = {"a": la.to_dict() if la else None, "b": lb.to_dict() if lb else None}
    if args.json:
        return _print_json(out)

    na, nb = _names(a, static), _names(b, static)
    print(_version_line(info))
    print(f"A = {na}   ({res['games_a']} games at the decision point)")
    print(f"B = {nb}   ({res['games_b']} games)")
    print("win rates " + ("adjusted for gold lead at first item (IPW)" if res["adjusted"] else "unadjusted"))
    print(f"\n{'condition':<30} {'A win %':>16} {'B win %':>16} {'diff':>7} {'games A/B':>11}  confidence")
    for c in res["conditions"]:
        def cell(x):
            return f"{x['win_rate']:.1%} ±{(x['hi'] - x['lo']) / 2:.1%}" if x else "hidden (<50)"
        diff = f"{c['difference'] * 100:+.1f}" if c["difference"] is not None else "—"
        print(f"{c['condition']:<30} {cell(c['a']):>16} {cell(c['b']):>16} {diff:>7} "
              f"{c['games_a']:>5}/{c['games_b']:<5}  {c['confidence'] or ''}")
    h = res["headline"]
    if h["leader"]:
        lead, trail = (na, nb) if h["leader"] == "a" else (nb, na)
        line = f"\nOverall, {lead} is ahead by {h['difference'] * 100:.1f} points"
        if h["flips"]:
            conds = ", ".join(f["condition"] for f in h["flips"][:3])
            line += f"; {trail} only wins when: {conds}."
        else:
            line += f" in every condition with enough games."
        print(line)
    if "this_comp" in out:
        tc = out["this_comp"]
        if tc["a"] and tc["b"]:
            d = (tc["a"]["win_rate"] - tc["b"]["win_rate"]) * 100
            lead = na if d >= 0 else nb
            print(f"Against this comp ({args.enemies}): {lead} is ahead by {abs(d):.1f} points "
                  f"(A {tc['a']['win_rate']:.1%}, B {tc['b']['win_rate']:.1%}).")
    if "stability" in out:
        s = out["stability"]
        if s["sign_agreement"] is not None:
            print(f"Stability across two halves: winner agrees in {s['sign_agreement']:.0%} of "
                  f"{s['conditions_compared']} conditions.")


def cmd_recommend(args):
    from buildopt.bundle import load_bundle
    from buildopt.itemset import build_item_set
    from buildopt.lcu.champselect import assign_roles
    from buildopt.scoring import Scorer, format_recommendation

    from buildopt import ddragon

    static = ddragon.load(offline=True)
    sc = Scorer(load_bundle(args.bundle))
    enemies = [static.champion_id(e.strip()) for e in args.enemies.split(",") if e.strip()]
    roles = assign_roles(enemies, sc.profiles)
    rec = sc.recommend(enemies, roles)
    if args.json:
        out = rec.to_dict()
        if args.itemset:
            out["item_set"] = build_item_set(sc, rec)
        return _print_json(out)
    print(format_recommendation(sc, rec))
    print("Enemy roles: " + ", ".join(f"{sc.champion(c)} {r.lower()}" for c, r in roles.items()))
    if args.itemset:
        print("\nItem set:")
        for block in build_item_set(sc, rec)["blocks"]:
            print(f"  {block['type']}: " + ", ".join(sc.item(int(i['id'])) for i in block["items"]))


def cmd_evaluate(args):
    from buildopt.analysis.evaluate import evaluate

    static = _static(args)
    games, info, _ = _games(args, static)
    res = evaluate(games, args.min_games)
    if args.json:
        return _print_json({"data_version": info, **res})
    print(_version_line(info))
    print(f"held-out games: {res['test_games']:,} (trained on {res['train_games']:,})")
    for k, v in res["log_likelihood"].items():
        print(f"  mean log-likelihood {k:<24} {v:.4f}")
    print("gate (comp-only model beats both baselines):", "PASS" if res["gate_passed"] else "FAIL")


def cmd_wincon(args):
    from buildopt.analysis import wincon
    from buildopt.bundle import load_bundle

    from buildopt import ddragon

    static = ddragon.load(offline=True)
    b = load_bundle(args.bundle)
    allies = [static.champion_id(x.strip()) for x in args.allies.split(",")]
    enemies = [static.champion_id(x.strip()) for x in args.enemies.split(",")]
    _print_json(wincon.analyze(allies, enemies, b["wincon"]))


def cmd_sync(args):
    from buildopt.app.bundles import sync_bundles

    updated = sync_bundles(args.source, args.dest)
    print(f"updated {len(updated)} bundle(s): {', '.join(updated) or '-'}")


def cmd_companion(args):
    from buildopt.app import main
    from buildopt.app.settings import Settings

    if args.read_only or args.bundle_dir or args.source:
        s = Settings.load(args.settings)
        if args.read_only:
            s.auto_import = False
        if args.bundle_dir:
            s.bundle_dir = args.bundle_dir
        if args.source:
            s.bundle_source = args.source
        s.save()
    main.run(args.settings, headless=args.headless)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="buildopt", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def db(sp):
        sp.add_argument("--db", default="data/games.sqlite")
        sp.add_argument("--offline", action="store_true", help="use the bundled Data Dragon snapshot")

    def data(sp):
        db(sp)
        sp.add_argument("--champion", default="Briar")
        sp.add_argument("--role", default="JUNGLE", choices=ROLES)
        sp.add_argument("--patch", help="current patch (default: newest in the database)")
        sp.add_argument("--prev-weight", type=float, default=0.3)
        sp.add_argument("--min-current-games", type=int, default=5000)
        sp.add_argument("--changed-items", help="comma-separated item ids changed this patch")
        sp.add_argument("--changed-runes", help="comma-separated rune ids changed this patch")
        sp.add_argument("--min-tier", default="EMERALD")
        sp.add_argument("--min-games", type=int, default=50)
        sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("crawl", help="collect ranked games from the Riot API (rank cascade)")
    db(sp)
    sp.add_argument("--api-key")
    sp.add_argument("--platform", action="append", required=True, help="e.g. euw1, na1, kr (repeatable)")
    sp.add_argument("--patch", action="append", required=True, help="patch to keep, e.g. 16.19 (repeatable)")
    sp.add_argument("--target", action="append", help="Champion:ROLE to fetch timelines for, e.g. Briar:JUNGLE")
    sp.add_argument("--since", help="ISO date; only list matches after it (patch release date)")
    sp.add_argument("--max-games", type=int, help="stop after this many target rows")
    sp.add_argument("--max-pages", type=int, default=50, help="league-entries pages per division")
    sp.add_argument("--lowest", default="EMERALD:IV", help="lowest tier:division to crawl")
    sp.set_defaults(fn=cmd_crawl)

    sp = sub.add_parser("synth", help="generate synthetic games with known effects (no API key needed)")
    db(sp)
    sp.add_argument("--matches", type=int, default=20000)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(fn=cmd_synth)

    sp = sub.add_parser("status", help="games per champion and role")
    db(sp)
    sp.add_argument("--min-games", type=int, default=1)
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("bundle", help="fit the model and write the stats bundle for one champion/role")
    data(sp)
    sp.add_argument("--out", default="bundles")
    sp.set_defaults(fn=cmd_bundle)

    sp = sub.add_parser("tree", help="win rate for every ordered prefix of completed items")
    data(sp)
    sp.add_argument("--depth", type=int, default=3)
    sp.set_defaults(fn=cmd_tree)

    sp = sub.add_parser("pairs", help="item pair synergy (log-odds lift)")
    data(sp)
    sp.add_argument("--within", type=int, default=3)
    sp.add_argument("--top", type=int, default=20)
    sp.set_defaults(fn=cmd_pairs)

    sp = sub.add_parser("compare", help="compare two build paths at the decision point, by condition")
    data(sp)
    sp.add_argument("--a", required=True, help='e.g. "Collector, Black Cleaver"')
    sp.add_argument("--b", required=True, help='e.g. "Titanic Hydra, Black Cleaver"')
    sp.add_argument("--no-adjust", action="store_true", help="skip gold-lead weighting")
    sp.add_argument("--stability", action="store_true", help="milestone 2 gate: agreement across two halves")
    sp.add_argument("--bundle", help="bundle to score a specific comp with")
    sp.add_argument("--enemies", help="comma-separated enemy champions for --bundle")
    sp.set_defaults(fn=cmd_compare)

    sp = sub.add_parser("recommend", help="score loadouts from a bundle against an enemy comp")
    sp.add_argument("--bundle", required=True)
    sp.add_argument("--enemies", required=True, help="comma-separated enemy champions (any order)")
    sp.add_argument("--itemset", action="store_true", help="also show the item set blocks")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_recommend)

    sp = sub.add_parser("evaluate", help="milestone 3 gate: held-out log-likelihood vs baselines")
    data(sp)
    sp.set_defaults(fn=cmd_evaluate)

    sp = sub.add_parser("wincon", help="win-condition analysis for two teams")
    sp.add_argument("--bundle", required=True)
    sp.add_argument("--allies", required=True)
    sp.add_argument("--enemies", required=True)
    sp.set_defaults(fn=cmd_wincon)

    sp = sub.add_parser("sync", help="download updated bundles from a URL or directory")
    sp.add_argument("--source", required=True)
    sp.add_argument("--dest", default=str(Path.home() / ".buildopt" / "bundles"))
    sp.set_defaults(fn=cmd_sync)

    sp = sub.add_parser("companion", help="run the champ select companion")
    sp.add_argument("--settings")
    sp.add_argument("--headless", action="store_true", help="console output instead of a window")
    sp.add_argument("--read-only", action="store_true", help="turn auto-import off")
    sp.add_argument("--bundle-dir")
    sp.add_argument("--source", help="bundle source URL or directory")
    sp.set_defaults(fn=cmd_companion)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        args.fn(args)
    except (KeyError, ValueError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
