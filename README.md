# League Build Optimizer

Recommends a **rune page + build path** scored against the five enemy champions in your lobby, then puts both into the League client during champ select so you never tab out.

## How to use it (no programming needed)

### 1. Download

Go to **[Releases → latest](https://github.com/AiHost12345/League-content-gen/releases/latest)** and download:

* **Windows:** `LeagueBuildOptimizer.exe`
* **Mac:** `LeagueBuildOptimizer-Mac.zip` (double-click to unzip)

There's nothing to install. Double-click the file to open it.

> **"Windows protected your PC"?** The app isn't code-signed (signing costs money), so Windows warns about every new app. Click **More info → Run anyway**.
> **Mac says it can't be opened?** Right-click the app → **Open** → **Open**.

### 2. See it working (1 minute)

Open the **Get Data** tab and click **No key? Try demo data**. This makes *fake* Briar games and shows what a recommendation looks like in **Try a Matchup**. Demo advice isn't real, and it's labelled "(demo)".

### 3. Get real data

1. Sign in at **[developer.riotgames.com](https://developer.riotgames.com/)** with your League account and copy the **Development API Key**.
2. In **Get Data**, paste the key and click **Test key**.
3. Pick your champion, role and regions, then click **Start collecting**.
   * Leave the app open. A personal key is slow: expect roughly a day for 20,000 games.
   * You can stop and start whenever you like. Progress is saved.
   * The key **expires every 24 hours**. When collecting stops with a key error, get a new key, paste it, and press Start again.
4. Click **Build recommendations**. It works with fewer games too, but the confidence will be lower.

### 4. Play

Keep the app open and start League. In champ select, the **Champ Select** tab shows your recommended runes, build, other strong builds and the reasons. When you lock in, the rune page and an item set are added automatically, both named `BO: …`. In game, pick the `BO:` set from the shop's item set list. The **Last import** box shows whether each one worked; if something fails, click **Open log file** and send it over. Your own rune pages are never deleted.

Untick **"Put runes and item set into my client automatically"** if you only want to look.

If the top-right corner keeps saying **League client: not found** while League is open, go to **Settings** and pick your League folder (the one with `LeagueClient.exe`).

Your data lives in a `.buildopt` folder in your user folder (**Settings → Open data folder**).

---

## For developers

It follows YordleDiff's approach (Wilson-scored, per-matchup win rates) and fixes these gaps:

| Gap in YordleDiff | Here |
|---|---|
| Item slots scored one at a time | Ordered build paths scored as a unit (`analysis/paths.py`, `analysis/model.py`) |
| Five enemies treated independently | Enemies summarised as comp traits, plus per-champion terms for exact matchups with 300+ games (`analysis/traits.py`) |
| Items and runes scored separately | Joint item × rune logistic model with interaction terms |
| Survivorship / selection bias | Decision-point comparisons and inverse-probability weighting on gold lead |
| Website only | Desktop companion that writes to the client (`lcu/`, `app/`) |
| Stale patch labels | One data version stamped on every output |

## Install

```bash
pip install -e ".[dev]"      # Python 3.10+; numpy, requests, websocket-client
pytest                        # ~50 tests, ~25 s
buildopt-gui                  # the desktop app
python packaging/build.py     # build the one-file app into dist/ (needs pyinstaller)
```

## Try it without an API key

`synth` generates games with known planted effects (Collector → BC wins against 0–1 tanks, Titanic → BC against 2+, Collector bought more when already ahead, and so on). Every other command works on it unchanged.

```bash
buildopt synth   --db data/games.sqlite --matches 20000 --offline
buildopt bundle  --db data/games.sqlite --champion Briar --role JUNGLE --out bundles --offline
buildopt recommend --bundle bundles/233_JUNGLE.json --enemies "Ornn,Sejuani,Galio,Ashe,Braum" --itemset
buildopt compare --db data/games.sqlite --a "Collector, BC" --b "Titanic, BC" --stability --offline
buildopt tree    --db data/games.sqlite --depth 3 --offline
buildopt pairs   --db data/games.sqlite --offline
buildopt evaluate --db data/games.sqlite --offline
```

`--offline` uses the Data Dragon snapshot shipped in the package (16.19.1). Without it, the current Data Dragon is fetched and cached next to the database.

## Real data

```bash
export RIOT_API_KEY=RGAPI-...
buildopt crawl --db data/games.sqlite --platform euw1 --platform na1 --platform kr \
  --patch 16.19 --patch 16.18 --since 2026-09-24 --target Briar:JUNGLE --max-games 20000
```

* **Rank cascade:** Challenger → Grandmaster → Master → Diamond I–IV → Emerald I–IV (`--lowest` sets the floor). Each step is crawled until its players' games on the target patches run out, then the crawl moves down. Every row keeps its tier.
* **Resumable:** all crawl state lives in SQLite.
* **Rate limits:** each region has its own sliding-window limiter (default 20/1 s and 100/2 min, updated from `X-App-Rate-Limit`; 429s honour `Retry-After`). One thread per platform.
* **Cost control:** timelines are fetched only for matches that contain a `--target` champion and role. Other matches still store all 10 rows, which feed the champion trait profiles and role play-rates.
* A personal key expires every 24 h and is enough for one champion. Apply for a production key before going past Briar.

### The row table

`pipeline/parse.py` reduces each match and timeline to one row per player. A row holds the patch, region, tier, game length, win, the champion and role, all 9 other champions with their roles, the full rune page with shards, damage/heal/CC stats, and the **completed-item sequence**: legendaries and tier-2 boots only, with undos removed and items sold back within 3 minutes dropped. Each item records the minute it finished and the lane and team gold difference at that moment. Starting items and per-minute positions (for the jungle planner) are kept too.

## Analysis

* **Patch policy** (`analysis/dataset.py`): the current patch is primary. The previous patch is used at weight 0.3 until the current one has `--min-current-games`, then it is dropped. Previous-patch rows that touch `--changed-items` / `--changed-runes` get weight 0.
* **Comp traits** (`analysis/traits.py`, `analysis/profiles.py`): tanks, AP share, hard CC, healing, burst, ranged, poke. Per-champion scores are measured from match data as percentiles (damage taken and mitigated, magic damage share, CC time, healing) and blended with a Data Dragon prior for rarely played champions.
* **Path views** (`analysis/paths.py`):
  * the prefix tree;
  * pair synergy, `lift(A,B) = logit WR(A∩B) − logit WR(A) − logit WR(B) + logit WR(champ)`;
  * conditional comparison: games at the decision point only (first k items exactly A or B), weighted by inverse propensity on gold lead and minute at first item, then split by comp condition, game length and decision-item minute. Paths under 50 games are hidden and 50–200 games are labelled low confidence.
* **Joint model** (`analysis/model.py`): `logit P(win) = β_path + β_rune + γᵀc + δ_pathᵀc + κ_keystoneᵀc + β_path×rune + θᵀs + per-matchup terms`. It is ridge-penalised: main effects shrink toward the champion average, and interactions shrink toward zero unless the data supports them. Penalty strength per group (interactions, path×page, matchup, main) is picked by 3-fold cross-validation.
* **Builds — no shortlist** (`analysis/candidates.py`): every 3-item build players finished is scored. Builds with 30+ games get their own model term; rarer builds are fitted through a group sharing their first two items (or first item), then each gets its own win-rate adjustment from its own games, shrunk toward the group so a few lucky games can't fake a high win rate. Rune pages: the top 8 plus single swaps (keystone, secondary tree, one minor).
* **Scoring** (`scoring.py`): every build × page is scored for the current comp and ranked by **adjusted win rate** (or, as a setting, by the Wilson lower bound). The recommended/imported build needs at least 20 games by default (Settings; 1 = any build); every build is listed in **All builds for this comp**. The effective sample size comes from the model's uncertainty for that loadout. Win rates are averaged over the game states seen in training, because gold lead isn't known in champ select. Output: the best loadout, the runner-up with a different keystone, the alt path, and 2–3 "why" lines such as `0 tanks → The Collector over Titanic Hydra`.

## Stats bundle

`buildopt bundle` writes `bundles/<championId>_<ROLE>.json` and updates `bundles/index.json` (with sha256). A bundle holds:

* the fitted model and its covariance;
* candidate paths and pages, plus the shards per page;
* boots placement and boots choice by AP share;
* situational items per category (anti-heal / armor / MR / survive) by comp level;
* late-game items, start items, and champion profiles;
* jungle routes and win-condition signals;
* item, rune and champion names.

Every bundle carries one data version, e.g. `data 16.19-20261002-e65c89 · patch 16.19 · 12,007 games`. It is shown in every CLI output, in the window, and in the item set title.

Host the `bundles/` folder on any static file host (or a shared folder) and point the app at it.

## Champ select companion (command line)

The desktop app (`buildopt-gui`, `src/buildopt/gui/`) wraps everything below. The `Build desktop app` GitHub workflow tests, builds the Windows `.exe` and Mac app with PyInstaller, and publishes them to the `latest` release.


```bash
buildopt companion --source https://your-host/bundles        # Tk window
buildopt companion --headless --read-only                     # console, never writes
```

Settings live in `~/.buildopt/settings.json`: `auto_import`, `bundle_source`, `league_path`, and the page prefix (`BO:`).

* **Connection** (`lcu/connection.py`): the app reads the `lockfile` and talks HTTPS to `127.0.0.1:<port>` with Basic auth `riot:<password>`. It trusts **Riot's root certificate** (`lcu/riotgames.pem`) rather than disabling verification. Only the hostname check is off, because the cert isn't issued for 127.0.0.1. Updates arrive over a websocket subscription to `OnJsonApiEvent_lol-champ-select_v1_session`, with no polling. When the password changes on a client restart, the app reconnects.
* **Smoke test** on connect: every endpoint the app uses is checked first. If any fails, the app runs read-only.
* **Flow** (`app/companion.py`):
  * Hover or lock → load the bundle.
  * Each enemy lock → enemy roles are assigned from play-rates (exact assignment over ≤120 permutations), then the loadout is re-scored (~7 ms).
  * Your lock → the rune page and item set are written.
  * A later enemy lock that changes the top loadout → both are rewritten, and the window says why (e.g. `Titanic Hydra → The Collector: enemy now has 0 tanks`).
  * One last write runs 3 s before finalization ends.
* **Import** (`lcu/importer.py`):
  * Writes happen only in `ChampSelect`.
  * Only pages and sets whose name starts with `BO:` are touched; user pages are never deleted.
  * At the rune page limit the app asks once which page it may own, and remembers the answer.
  * The item set has these blocks: Start · Core (boots placed where the data puts them) · Situational anti-tank / anti-heal / armor / MR / survive (with the reason, e.g. `vs heavy healing`) · Late game · Alt path.
* **Scope:** only what champ select shows. No enemy names, ranks, histories or dodge advice.

## Later modules (first versions included)

* **Win conditions** (`analysis/wincon.py`, `buildopt wincon`):
  * seven-signal champion vectors, summed into team vectors;
  * one shared archetype list, matched by cosine;
  * the fatal flaw is the largest gap in your favour;
  * `fit_signal_weights` learns dimension weights from which team gaps actually predict wins.
* **Jungle planner v1** (`analysis/jungle.py`): first-clear routes come from minute-2–4 positions, normalised to the blue side, and are ranked by Wilson lower bound vs the enemy jungler. Gank priority by lane comes from ally CC vs enemy escape. **v2 (per-minute win-probability model) isn't built.**

## Milestone gates

| Gate | How to check |
|---|---|
| 1. 20,000+ Emerald+ games, sequences validated | `buildopt status`. Spot-check `items` in a few rows against op.gg (manual). |
| 2. Collector→BC vs Titanic→BC stable across two pulls | `buildopt compare … --stability` (sign agreement across two disjoint halves) |
| 3. Joint model beats most-popular loadout on held-out games | `buildopt evaluate`: held-out log-likelihood vs champion average and per-loadout win rate |
| 4. 20 real games written correctly, no user pages touched | Manual, in the real client. Fake-client tests cover the safety rules. |

## Status and known limits

* Everything above runs end to end on synthetic data and is covered by tests. On synthetic data, IPW turns a raw +6.5-point Collector edge into −2.0, matching the planted truth. Evaluation also passes gate 3.
* **Not yet exercised against the live Riot API or a live League client.** The crawler, LCU client, websocket and importer are tested against fakes that follow the Riot and Hextech docs. LCU endpoints are unofficial and can change between patches; the smoke test and read-only fallback exist for that reason.
* The Tk window isn't covered by tests (the console UI is). Register the app on the Riot developer portal before distributing it.
* Stat shards are taken as the most common set for the chosen page; they aren't scored against the comp yet.
