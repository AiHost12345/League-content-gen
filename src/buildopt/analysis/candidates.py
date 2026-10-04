"""Candidate rune pages and build paths for the joint scorer."""

from __future__ import annotations

from collections import Counter, defaultdict

from buildopt.analysis.dataset import Game
from buildopt.stats import MIN_GAMES


def candidate_paths(games: list[Game], min_games: int = 1, max_paths: int | None = None) -> list[tuple[int, ...]]:
    """Every first-three-item build players finished (by default, no shortlist)."""
    counts = Counter(g.path[:3] for g in games if len(g.path) >= 3)
    return [p for p, n in counts.most_common(max_paths) if n >= min_games]


ANY = 0  # placeholder item in a group key, e.g. (BoRK, BC, ANY) = "BoRK -> BC -> anything rarer"


def model_paths(games: list[Game], min_games: int = 30) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Split builds into the joint model's path terms and the rare builds scored on top of them.

    Builds with ``min_games``+ games get their own term. Rarer builds are grouped by their first two
    items, then by their first item, then into one catch-all group, so the model stays small. Each
    rare build is still scored individually afterwards (see ``LoadoutModel.rare_effects``).
    Returns (model paths including groups, rare builds).
    """
    counts = Counter(g.path[:3] for g in games if len(g.path) >= 3)
    common = [p for p, n in counts.most_common() if n >= min_games]
    rare = {p: n for p, n in counts.items() if n < min_games}
    two = Counter()
    for p, n in rare.items():
        two[(p[0], p[1], ANY)] += n
    groups2 = sorted(k for k, n in two.items() if n >= min_games)
    one = Counter()
    for p, n in rare.items():
        if (p[0], p[1], ANY) not in groups2:
            one[(p[0], ANY, ANY)] += n
    groups1 = sorted(k for k, n in one.items() if n >= min_games)
    leftover = any((p[0], p[1], ANY) not in groups2 and (p[0], ANY, ANY) not in groups1 for p in rare)
    paths = common + groups2 + groups1 + ([(ANY, ANY, ANY)] if leftover else [])
    return paths, sorted(rare)


def _swap_kind(a: tuple, b: tuple) -> str | None:
    """Return the kind of single swap turning page a into page b, if any."""
    if a == b:
        return None
    a_prim, a_sub, a_perks = a[0], a[1], a[2:]
    b_prim, b_sub, b_perks = b[0], b[1], b[2:]
    if a_prim != b_prim:
        return None
    if a_sub == b_sub and a_perks[1:] == b_perks[1:] and a_perks[0] != b_perks[0]:
        return "keystone"
    if a_perks[:4] == b_perks[:4] and (a_sub != b_sub or a_perks[4:] != b_perks[4:]):
        return "secondary"
    if a_sub == b_sub and a_perks[0] == b_perks[0]:
        diff = sum(1 for x, y in zip(a_perks[1:], b_perks[1:]) if x != y)
        if diff == 1 or (diff == 2 and set(a_perks[4:]) == set(b_perks[4:])):
            return "minor"
    return None


def candidate_pages(games: list[Game], min_games: int = MIN_GAMES, top: int = 8, max_pages: int = 16) -> list[tuple]:
    """The most played pages plus single swaps (keystone, secondary tree, one minor) that reach the threshold."""
    counts = Counter(g.page for g in games)
    popular = [p for p, n in counts.most_common(top) if n >= min_games]
    out = list(popular)
    for page, n in counts.most_common():
        if len(out) >= max_pages or n < min_games:
            break
        if page in out:
            continue
        if any(_swap_kind(base, page) for base in popular):
            out.append(page)
    return out


def page_shards(games: list[Game], pages: list[tuple]) -> dict[tuple, tuple]:
    by_page: dict[tuple, Counter] = defaultdict(Counter)
    for g in games:
        by_page[g.page][g.shards] += 1
    return {p: by_page[p].most_common(1)[0][0] for p in pages if by_page[p]}
