"""Candidate rune pages and build paths for the joint scorer."""

from __future__ import annotations

from collections import Counter, defaultdict

from buildopt.analysis.dataset import Game
from buildopt.stats import MIN_GAMES


def candidate_paths(games: list[Game], min_games: int = MIN_GAMES, max_paths: int = 30) -> list[tuple[int, ...]]:
    """All first-three-item prefixes above the sample threshold."""
    counts = Counter(g.path[:3] for g in games if len(g.path) >= 3)
    return [p for p, n in counts.most_common(max_paths) if n >= min_games]


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
