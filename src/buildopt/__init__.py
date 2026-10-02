"""League Build Optimizer.

Server side: crawl ranked games, reduce them to per-player rows, fit a joint
rune x build-path model per champion and role, and publish a stats bundle.

Desktop side: watch champ select through the League client API, score
loadouts locally against the enemy comp, and write the rune page and item set.
"""

__version__ = "0.1.0"

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
