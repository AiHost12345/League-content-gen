"""League Build Optimizer desktop app.

Tabs:
  Champ Select  - live recommendation while you're in a lobby; imports runes + items.
  Get Data      - paste a Riot API key, collect games, build recommendations (or try demo data).
  Try a Matchup - pick five enemies and see what the tool would recommend.
  Settings      - League folder, auto-import, data folder.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk

from buildopt import ROLES, __version__, ddragon
from buildopt.app.bundles import BundleCache
from buildopt.app.companion import recommendation_view
from buildopt.app.settings import Settings, app_dir
from buildopt.gui import jobs
from buildopt.gui.jobs import REGIONS, ROLE_LABELS

# League-client-like palette
BG = "#0a1428"
PANEL = "#111e33"
FIELD = "#1a2940"
GOLD = "#c8aa6e"
TEXT = "#f0e6d2"
MUTED = "#a09b8c"
TEAL = "#0ac8b9"
RED = "#e05a5a"
FONT = "Segoe UI" if sys.platform == "win32" else "Helvetica"


def apply_theme(root: tk.Tk) -> None:
    root.configure(bg=BG)
    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure(".", background=BG, foreground=TEXT, fieldbackground=FIELD, bordercolor=PANEL,
                 lightcolor=PANEL, darkcolor=PANEL, font=(FONT, 10))
    st.configure("TFrame", background=BG)
    st.configure("Panel.TFrame", background=PANEL)
    st.configure("TLabel", background=BG, foreground=TEXT)
    st.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
    st.configure("Muted.TLabel", background=BG, foreground=MUTED)
    st.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED)
    st.configure("Title.TLabel", background=BG, foreground=GOLD, font=(FONT, 16, "bold"))
    st.configure("H2.TLabel", background=BG, foreground=GOLD, font=(FONT, 12, "bold"))
    st.configure("Big.TLabel", background=PANEL, foreground=TEXT, font=(FONT, 13, "bold"))
    st.configure("Win.TLabel", background=PANEL, foreground=TEAL, font=(FONT, 12, "bold"))
    st.configure("PanelGold.TLabel", background=PANEL, foreground=GOLD, font=(FONT, 10, "bold"))
    st.configure("TButton", background=FIELD, foreground=TEXT, padding=(12, 6), borderwidth=1)
    st.map("TButton", background=[("active", "#24395a"), ("disabled", PANEL)], foreground=[("disabled", MUTED)])
    st.configure("Accent.TButton", background=GOLD, foreground=BG, font=(FONT, 10, "bold"))
    st.map("Accent.TButton", background=[("active", "#e0c48a"), ("disabled", "#5c5240")])
    st.configure("TCheckbutton", background=BG, foreground=TEXT)
    st.map("TCheckbutton", background=[("active", BG)])
    st.configure("Panel.TCheckbutton", background=PANEL, foreground=TEXT)
    st.map("Panel.TCheckbutton", background=[("active", PANEL)])
    st.configure("TEntry", fieldbackground=FIELD, foreground=TEXT, insertcolor=TEXT)
    st.configure("TCombobox", fieldbackground=FIELD, background=FIELD, foreground=TEXT, arrowcolor=GOLD)
    st.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)])
    st.configure("TNotebook", background=BG, borderwidth=0)
    st.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(16, 8), font=(FONT, 10, "bold"))
    st.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", GOLD)])
    st.configure("Horizontal.TProgressbar", background=TEAL, troughcolor=FIELD)
    st.configure("Vertical.TScrollbar", background=FIELD, troughcolor=PANEL, arrowcolor=GOLD, bordercolor=PANEL,
                 lightcolor=FIELD, darkcolor=FIELD, gripcount=0)
    st.map("Vertical.TScrollbar", background=[("active", "#24395a"), ("disabled", PANEL)])
    st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=24)
    st.configure("Treeview.Heading", background=FIELD, foreground=GOLD)
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", GOLD)
    root.option_add("*TCombobox*Listbox.selectForeground", BG)


def open_path(path) -> None:
    path = str(path)
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - opening a folder for the user
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class SearchCombo(ttk.Combobox):
    """Combobox that filters its list as you type."""

    def __init__(self, master, values: list[str], **kw):
        super().__init__(master, values=values, **kw)
        self.all_values = values
        self.bind("<KeyRelease>", self._filter)

    def _filter(self, event):
        if event.keysym in ("Up", "Down", "Return", "Escape", "Tab"):
            return
        text = self.get().lower()
        self["values"] = [v for v in self.all_values if text in v.lower()] if text else self.all_values


class LoadoutCard(ttk.Frame):
    """Shows one recommendation (the same view the companion produces)."""

    def __init__(self, master, empty_text: str):
        super().__init__(master, style="Panel.TFrame")
        # Scrollable: a canvas holding the content frame.
        self.canvas = tk.Canvas(self, bg=PANEL, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas, style="Panel.TFrame", padding=16)
        win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))
        self.header = ttk.Label(self.inner, style="PanelMuted.TLabel", text=empty_text, wraplength=640, justify="left")
        self.header.pack(anchor="w")
        self.body = ttk.Frame(self.inner, style="Panel.TFrame")
        self.body.pack(fill="both", expand=True)

    def _wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120) or (-1 if event.delta > 0 else 1), "units")

    def clear(self, text: str) -> None:
        self.header.configure(text=text)
        for w in self.body.winfo_children():
            w.destroy()

    def show(self, view: dict, item_blocks: list[tuple[str, str]] | None = None) -> None:
        self.clear(view["header"] + ("   ·   read-only (auto-import off)" if view.get("read_only") else ""))
        self.canvas.yview_moveto(0)
        b = self.body

        def row(label, value, style="Big.TLabel"):
            f = ttk.Frame(b, style="Panel.TFrame")
            f.pack(fill="x", pady=(10, 0))
            ttk.Label(f, text=label, style="PanelGold.TLabel", width=12).pack(side="left", anchor="n")
            ttk.Label(f, text=value, style=style, wraplength=560, justify="left").pack(side="left", anchor="w")

        row("RUNES", view["runes"])
        row("BUILD", view["build"])
        row("WIN RATE", view["win"], "Win.TLabel")
        if view.get("why"):
            row("WHY", "\n".join(f"• {w}" for w in view["why"]), "Panel.TLabel")
        if view.get("runner_up"):
            row("RUNNER-UP", view["runner_up"], "Panel.TLabel")
        if view.get("change"):
            row("CHANGED", view["change"], "Win.TLabel")
        for line in view.get("extras", []):
            ttk.Label(b, text=line, style="PanelMuted.TLabel", wraplength=640, justify="left").pack(anchor="w", pady=(8, 0))
        if item_blocks:
            ttk.Label(b, text="ITEM SET (goes into the in-game shop)", style="PanelGold.TLabel").pack(anchor="w", pady=(14, 2))
            for title, items in item_blocks:
                ttk.Label(b, text=f"{title}:  {items}", style="Panel.TLabel", wraplength=660,
                          justify="left").pack(anchor="w", pady=1)


class App:
    def __init__(self, root: tk.Tk, settings: Settings | None = None, autostart: bool = True):
        self.root = root
        self.settings = settings or Settings.load()
        if not os.path.exists(self.settings.path):
            self.settings.save()
        self.queue: queue.Queue = queue.Queue()
        self.static = ddragon.load(offline=True)  # names for menus; real data is fetched when collecting
        self.champ_names = sorted(c["name"] for c in self.static.champions.values())
        self.collect_stop: threading.Event | None = None
        self.collect_started = 0.0
        self.collect_start_count = 0

        root.title(f"League Build Optimizer {__version__}")
        try:
            from importlib import resources

            self._icon = tk.PhotoImage(file=str(resources.files("buildopt").joinpath("data/icon.png")))
            root.iconphoto(True, self._icon)
        except Exception:
            pass
        root.geometry("820x700")
        root.minsize(700, 560)
        apply_theme(root)

        top = ttk.Frame(root, padding=(16, 12, 16, 0))
        top.pack(fill="x")
        ttk.Label(top, text="League Build Optimizer", style="Title.TLabel").pack(side="left")
        self.conn_label = ttk.Label(top, text="● League client: not found", style="Muted.TLabel")
        self.conn_label.pack(side="right")

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=12, pady=12)
        self.live_tab = self._build_live(self.nb)
        self.data_tab = self._build_data(self.nb)
        self.try_tab = self._build_try(self.nb)
        self.settings_tab = self._build_settings(self.nb)

        self._install_log_handler()
        self.root.after(100, self._drain)
        self.refresh_bundles()
        if not self.cache.available():
            self.nb.select(self.data_tab)  # first run: nothing to recommend yet
        if autostart:
            threading.Thread(target=self._companion_thread, daemon=True).start()

    # ---- threading helpers ---------------------------------------------------
    def post(self, fn, *args) -> None:
        """Run fn(*args) on the UI thread."""
        self.queue.put((fn, args))

    def _drain(self) -> None:
        while True:
            try:
                fn, args = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn(*args)
            except Exception:
                logging.getLogger(__name__).exception("UI update failed")
        self.root.after(100, self._drain)

    def run_in_background(self, work, on_done=None, on_error=None) -> None:
        def target():
            try:
                result = work()
            except Exception as e:
                if on_error:
                    self.post(on_error, e)
                return
            if on_done:
                self.post(on_done, result)

        threading.Thread(target=target, daemon=True).start()

    def _install_log_handler(self) -> None:
        app = self

        class Handler(logging.Handler):
            def emit(self, record):
                app.post(app.append_log, self.format(record))

        h = Handler()
        h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        h.setLevel(logging.INFO)
        logger = logging.getLogger("buildopt")
        logger.setLevel(logging.INFO)
        logger.addHandler(h)

    # ---- Champ Select tab ------------------------------------------------------
    def _build_live(self, nb) -> ttk.Frame:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="  Champ Select  ")
        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        self.auto_var = tk.BooleanVar(value=self.settings.auto_import)
        ttk.Checkbutton(bar, text="Put runes and item set into my client automatically", variable=self.auto_var,
                        command=self._toggle_auto).pack(side="left")
        self.ready_label = ttk.Label(tab, text="", style="Muted.TLabel", wraplength=760, justify="left")
        self.ready_label.pack(fill="x", pady=(8, 8))
        self.live_card = LoadoutCard(tab, "Open League and enter champ select. Your recommendation appears here "
                                          "as soon as you hover a champion you have data for.")
        self.live_card.pack(fill="both", expand=True)
        self.live_status = ttk.Label(tab, text="Starting…", style="Muted.TLabel", wraplength=760)
        self.live_status.pack(fill="x", pady=(8, 0))
        return tab

    def _toggle_auto(self) -> None:
        self.settings.auto_import = self.auto_var.get()
        self.settings.save()

    def _companion_thread(self) -> None:
        from buildopt.app import main as app_main

        app = self

        class UIAdapter:
            def show(self, view):
                app.post(app.live_card.show, view)
                app.post(app.nb.select, app.live_tab)

            def status(self, message):
                app.post(app._set_live_status, message)

            def ask_page(self, pages):
                done = threading.Event()
                result = [None]

                def ask():
                    result[0] = app.ask_rune_page(pages)
                    done.set()

                app.post(ask)
                done.wait()
                return result[0]

        app_main.connect_loop(self.settings, UIAdapter(), threading.Event())

    def _set_live_status(self, message: str) -> None:
        self.live_status.configure(text=message)
        if message.startswith("Connected"):
            self.conn_label.configure(text="● League client: connected", foreground=TEAL)
        elif message.startswith(("Waiting for the League", "League client closed")):
            self.conn_label.configure(text="● League client: not found", foreground=MUTED)
            if message.startswith("Waiting for the League"):
                message = ("Waiting for the League client. If League is open and this doesn't change, "
                           "set your League folder in Settings.")
                self.live_status.configure(text=message)
        elif "read-only" in message:
            self.conn_label.configure(text="● League client: read-only", foreground=RED)

    def ask_rune_page(self, pages: list[dict]) -> int | None:
        win = tk.Toplevel(self.root)
        win.title("Rune pages full")
        win.configure(bg=BG)
        win.transient(self.root)
        ttk.Label(win, text="All your rune pages are in use.\nWhich page may the app overwrite? (asked only once)",
                  padding=12).pack()
        lb = tk.Listbox(win, height=min(10, len(pages)), bg=FIELD, fg=TEXT, selectbackground=GOLD)
        for p in pages:
            lb.insert("end", p.get("name"))
        lb.pack(padx=12, fill="both")
        result = [None]

        def choose():
            sel = lb.curselection()
            result[0] = pages[sel[0]]["id"] if sel else None
            win.destroy()

        f = ttk.Frame(win, padding=12)
        f.pack(fill="x")
        ttk.Button(f, text="Use this page", style="Accent.TButton", command=choose).pack(side="left")
        ttk.Button(f, text="Don't import runes", command=win.destroy).pack(side="right")
        win.grab_set()
        self.root.wait_window(win)
        return result[0]

    # ---- Get Data tab ------------------------------------------------------------
    def _build_data(self, nb) -> ttk.Frame:
        tab = ttk.Frame(nb, padding=16)
        nb.add(tab, text="  Get Data  ")
        s = self.settings

        ttk.Label(tab, text="The app needs recent ranked games for your champion before it can recommend anything.",
                  style="Muted.TLabel", wraplength=760, justify="left").pack(anchor="w")

        # Step 1: key
        ttk.Label(tab, text="1.  Riot API key", style="H2.TLabel").pack(anchor="w", pady=(14, 4))
        ttk.Label(tab, text="Sign in at developer.riotgames.com with your League account and copy the "
                            "Development API Key. It expires every 24 hours, so paste a new one each day.",
                  style="Muted.TLabel", wraplength=760, justify="left").pack(anchor="w")
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=6)
        self.key_var = tk.StringVar(value=s.api_key)
        ttk.Entry(row, textvariable=self.key_var, width=52, show="•").pack(side="left")
        ttk.Button(row, text="Get a key", command=lambda: webbrowser.open("https://developer.riotgames.com/")).pack(side="left", padx=6)
        ttk.Button(row, text="Test key", command=self._test_key).pack(side="left")
        self.key_result = ttk.Label(tab, text="", style="Muted.TLabel")
        self.key_result.pack(anchor="w")

        # Step 2: what to collect
        ttk.Label(tab, text="2.  What to collect", style="H2.TLabel").pack(anchor="w", pady=(14, 4))
        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="Champion").pack(side="left")
        self.champ_var = tk.StringVar(value=s.champion)
        SearchCombo(row, self.champ_names, textvariable=self.champ_var, width=18).pack(side="left", padx=(6, 18))
        ttk.Label(row, text="Role").pack(side="left")
        self.role_var = tk.StringVar(value=ROLE_LABELS[s.role])
        ttk.Combobox(row, values=[ROLE_LABELS[r] for r in ROLES], textvariable=self.role_var, state="readonly",
                     width=12).pack(side="left", padx=(6, 18))
        ttk.Label(row, text="Games").pack(side="left")
        self.target_var = tk.StringVar(value=str(s.target_games))
        ttk.Entry(row, textvariable=self.target_var, width=8).pack(side="left", padx=6)
        regions = ttk.Frame(tab)
        regions.pack(fill="x", pady=(8, 0))
        ttk.Label(regions, text="Regions").grid(row=0, column=0, sticky="w")
        self.region_vars = {}
        for i, (pid, label) in enumerate(list(REGIONS.items())[:10]):
            v = tk.BooleanVar(value=pid in s.regions)
            self.region_vars[pid] = v
            ttk.Checkbutton(regions, text=label, variable=v).grid(row=i // 5, column=1 + i % 5, sticky="w", padx=6)

        # Step 3: collect
        ttk.Label(tab, text="3.  Collect games, then build recommendations", style="H2.TLabel").pack(anchor="w", pady=(14, 4))
        row = ttk.Frame(tab)
        row.pack(fill="x")
        self.collect_btn = ttk.Button(row, text="Start collecting", style="Accent.TButton", command=self._toggle_collect)
        self.collect_btn.pack(side="left")
        self.build_btn = ttk.Button(row, text="Build recommendations", command=self._build_real)
        self.build_btn.pack(side="left", padx=6)
        ttk.Button(row, text="No key? Try demo data", command=self._demo).pack(side="right")
        self.progress = ttk.Progressbar(tab, maximum=max(1, s.target_games))
        self.progress.pack(fill="x", pady=(10, 2))
        self.progress_label = ttk.Label(tab, text="", style="Muted.TLabel")
        self.progress_label.pack(anchor="w")
        self.log_text = tk.Text(tab, height=8, bg=PANEL, fg=MUTED, insertbackground=TEXT, relief="flat",
                                font=("Consolas" if sys.platform == "win32" else "Courier", 9), wrap="word")
        self.log_text.pack(fill="both", expand=True, pady=(8, 0))
        self.log_text.configure(state="disabled")
        self._update_progress()
        return tab

    def append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 500:
            self.log_text.delete("1.0", f"{lines - 500}.0")
        self.log_text.configure(state="disabled")

    def _save_data_form(self) -> bool:
        s = self.settings
        s.api_key = self.key_var.get().strip()
        try:
            self.static.champion_id(self.champ_var.get())
        except KeyError:
            messagebox.showerror("Champion", f"I don't recognise the champion '{self.champ_var.get()}'.")
            return False
        s.champion = self.champ_var.get()
        s.role = {v: k for k, v in ROLE_LABELS.items()}[self.role_var.get()]
        try:
            s.target_games = max(500, int(self.target_var.get().replace(",", "")))
        except ValueError:
            s.target_games = 20000
        s.regions = [pid for pid, v in self.region_vars.items() if v.get()] or ["na1"]
        s.save()
        self.progress.configure(maximum=s.target_games)
        return True

    def _test_key(self) -> None:
        if not self._save_data_form():
            return
        self.key_result.configure(text="Checking…", foreground=MUTED)
        region = self.settings.regions[0]

        def done(msg):
            ok = msg == "OK"
            self.key_result.configure(text="✓ Key works" if ok else msg, foreground=TEAL if ok else RED)

        self.run_in_background(lambda: jobs.test_key(self.settings.api_key, region), done)

    def _toggle_collect(self) -> None:
        if self.collect_stop is not None:
            self.collect_stop.set()
            self.collect_btn.configure(text="Stopping…", state="disabled")
            return
        if not self._save_data_form():
            return
        if not self.settings.api_key:
            messagebox.showinfo("API key needed", "Paste your Riot API key in step 1 first.")
            return
        self.collect_stop = threading.Event()
        self.collect_started = time.time()
        self.collect_start_count = jobs.games_collected(self.settings)
        self.collect_btn.configure(text="Stop collecting")

        def finished(how: str):
            self.collect_stop = None
            self.collect_btn.configure(text="Start collecting", state="normal")
            self._update_progress()
            if how.startswith("error"):
                messagebox.showerror("Collection stopped", how[7:] + "\n\nIf your key expired, paste a new one "
                                     "and press Start again. Progress is saved.")
            elif how == "finished":
                messagebox.showinfo("Done", "Collected enough games. Press 'Build recommendations'.")

        job = jobs.CollectJob(self.settings, lambda how: self.post(finished, how), self.collect_stop)
        threading.Thread(target=job.run, daemon=True).start()
        self.root.after(3000, self._poll_progress)

    def _poll_progress(self) -> None:
        self._update_progress()
        if self.collect_stop is not None:
            self.root.after(5000, self._poll_progress)

    def _update_progress(self) -> None:
        s = self.settings
        n = jobs.games_collected(s)
        self.progress.configure(value=min(n, s.target_games))
        text = f"{n:,} of {s.target_games:,} {s.champion} {ROLE_LABELS[s.role].lower()} games collected"
        if self.collect_stop is not None and time.time() - self.collect_started > 60:
            rate = (n - self.collect_start_count) / ((time.time() - self.collect_started) / 3600)
            if rate > 0:
                hours = (s.target_games - n) / rate
                text += f"   ·   ~{rate:,.0f} games/hour, about {hours:,.0f} h to go"
        elif n == 0:
            text += "   ·   you can stop any time; progress is saved"
        self.progress_label.configure(text=text)

    def _build_real(self) -> None:
        if not self._save_data_form():
            return
        self.build_btn.configure(state="disabled", text="Building… (a minute or two)")

        def done(summary):
            self.build_btn.configure(state="normal", text="Build recommendations")
            self.refresh_bundles()
            messagebox.showinfo("Ready", f"Recommendations built:\n{summary}\n\nThey'll show up in champ select.")

        def failed(e):
            self.build_btn.configure(state="normal", text="Build recommendations")
            messagebox.showerror("Not ready yet", str(e))

        self.run_in_background(lambda: jobs.build_recommendations(self.settings), done, failed)

    def _demo(self) -> None:
        if not messagebox.askokcancel(
                "Demo data", "This makes FAKE Briar jungle games so you can see how the app works. "
                             "Its recommendations are not real advice.\n\nIt takes about a minute. Continue?"):
            return
        self.append_log("Building demo data…")

        def done(summary):
            self.append_log(f"Demo ready: {summary}")
            self.refresh_bundles()
            demo_key = next((k for k in self.bundle_choices if k.startswith("Briar · Jungle")), None)
            if demo_key:
                self.try_champ_var.set(demo_key)
            for v, name in zip(self.enemy_vars, ["Ornn", "Sejuani", "Galio", "Ashe", "Braum"]):
                v.set(name)
            self.nb.select(self.try_tab)
            self._recommend()

        def failed(e):
            messagebox.showerror("Demo failed", str(e))

        self.run_in_background(lambda: jobs.make_demo_data(self.settings, lambda m: self.post(self.append_log, m)),
                               done, failed)

    # ---- Try a Matchup tab -------------------------------------------------------
    def _build_try(self, nb) -> ttk.Frame:
        tab = ttk.Frame(nb, padding=16)
        nb.add(tab, text="  Try a Matchup  ")
        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="You play").pack(side="left")
        self.try_champ_var = tk.StringVar()
        self.try_champ_box = ttk.Combobox(row, textvariable=self.try_champ_var, state="readonly", width=26)
        self.try_champ_box.pack(side="left", padx=8)
        ttk.Label(tab, text="Enemy team (any order — leave blanks for picks you haven't seen yet)",
                  style="Muted.TLabel").pack(anchor="w", pady=(12, 4))
        row = ttk.Frame(tab)
        row.pack(fill="x")
        self.enemy_vars = []
        for _ in range(5):
            v = tk.StringVar()
            SearchCombo(row, self.champ_names, textvariable=v, width=14).pack(side="left", padx=(0, 6))
            self.enemy_vars.append(v)
        btns = ttk.Frame(tab)
        btns.pack(fill="x", pady=10)
        ttk.Button(btns, text="Recommend", style="Accent.TButton", command=self._recommend).pack(side="left")
        ttk.Button(btns, text="Clear", command=lambda: [v.set("") for v in self.enemy_vars]).pack(side="left", padx=6)
        self.try_card = LoadoutCard(tab, "Pick the champion you play and up to five enemies, then press Recommend.")
        self.try_card.pack(fill="both", expand=True)
        return tab

    def refresh_bundles(self) -> None:
        self.cache = BundleCache(self.settings.bundle_dir)
        avail = self.cache.available()
        self.bundle_choices = {}
        for (cid, role), entry in sorted(avail.items(), key=lambda kv: kv[1]["champion_name"]):
            demo = " (demo)" if str(entry.get("data_version", "")).startswith("DEMO") else ""
            self.bundle_choices[f"{entry['champion_name']} · {ROLE_LABELS[role]}{demo}"] = (cid, role)
        self.try_champ_box["values"] = list(self.bundle_choices)
        if self.bundle_choices and self.try_champ_var.get() not in self.bundle_choices:
            self.try_champ_var.set(next(iter(self.bundle_choices)))
        if avail:
            names = ", ".join(self.bundle_choices)
            self.ready_label.configure(text=f"Recommendations ready for: {names}")
        else:
            self.ready_label.configure(text="No recommendations yet. Go to Get Data to collect games "
                                            "(or try the demo).")

    def _recommend(self) -> None:
        from buildopt.itemset import build_blocks
        from buildopt.lcu.champselect import assign_roles

        choice = self.bundle_choices.get(self.try_champ_var.get())
        if not choice:
            # tolerate "(demo)" suffix differences
            choice = next((v for k, v in self.bundle_choices.items() if k.startswith(self.try_champ_var.get())), None)
        if not choice:
            messagebox.showinfo("No data", "Build recommendations for a champion first (Get Data tab).")
            return
        enemies = []
        for v in self.enemy_vars:
            name = v.get().strip()
            if not name:
                continue
            try:
                enemies.append(self.static.champion_id(name))
            except KeyError:
                messagebox.showerror("Unknown champion", f"I don't recognise '{name}'.")
                return
        scorer = self.cache.scorer(*choice)
        roles = assign_roles(enemies, scorer.profiles)
        rec = scorer.recommend(enemies, roles)
        view = recommendation_view(scorer, rec, roles, enemies=enemies).as_dict()
        blocks = [(b["type"], ", ".join(scorer.item(int(i["id"])) for i in b["items"])) for b in build_blocks(scorer, rec)]
        self.try_card.show(view, blocks)

    # ---- Settings tab --------------------------------------------------------------
    def _build_settings(self, nb) -> ttk.Frame:
        tab = ttk.Frame(nb, padding=16)
        nb.add(tab, text="  Settings  ")
        s = self.settings
        ttk.Label(tab, text="League of Legends folder", style="H2.TLabel").pack(anchor="w")
        ttk.Label(tab, text="Only needed if League isn't installed in the default place "
                            r"(C:\Riot Games\League of Legends). Pick the folder that contains LeagueClient.exe.",
                  style="Muted.TLabel", wraplength=760, justify="left").pack(anchor="w")
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=6)
        self.league_var = tk.StringVar(value=s.league_path or "")
        ttk.Entry(row, textvariable=self.league_var, width=60).pack(side="left")
        ttk.Button(row, text="Browse…", command=self._browse_league).pack(side="left", padx=6)

        ttk.Label(tab, text="Importing", style="H2.TLabel").pack(anchor="w", pady=(16, 0))
        ttk.Label(tab, text=f"The app only creates and replaces rune pages and item sets whose name starts with "
                            f"'{s.page_prefix}'. Your own pages are never deleted. If all your rune pages are "
                            f"used, it asks once which one it may overwrite.",
                  style="Muted.TLabel", wraplength=760, justify="left").pack(anchor="w")
        ttk.Button(tab, text="Forget which rune page the app may use",
                   command=self._reset_page).pack(anchor="w", pady=6)

        ttk.Label(tab, text="Your data", style="H2.TLabel").pack(anchor="w", pady=(16, 0))
        ttk.Label(tab, text=f"Games, recommendations and settings are stored in {app_dir()}",
                  style="Muted.TLabel", wraplength=760).pack(anchor="w")
        ttk.Button(tab, text="Open data folder", command=lambda: open_path(app_dir())).pack(anchor="w", pady=6)

        ttk.Label(tab, text="About", style="H2.TLabel").pack(anchor="w", pady=(16, 0))
        ttk.Label(tab, text="Uses only what the champ select screen shows (champions, your role). No enemy names, "
                            "ranks or match histories. Not endorsed by Riot Games.",
                  style="Muted.TLabel", wraplength=760, justify="left").pack(anchor="w")
        return tab

    def _browse_league(self) -> None:
        path = filedialog.askdirectory(title="Pick your League of Legends folder")
        if path:
            self.league_var.set(path)
            self.settings.league_path = path
            self.settings.save()
            if not os.path.exists(os.path.join(path, "LeagueClient.exe")) and not os.path.exists(os.path.join(path, "lockfile")):
                messagebox.showwarning("Hmm", "That folder doesn't look like a League install "
                                              "(no LeagueClient.exe). It's saved anyway.")

    def _reset_page(self) -> None:
        self.settings.owned_page_id = None
        self.settings.asked_for_page = False
        self.settings.save()
        messagebox.showinfo("Done", "The app will ask again next time your rune pages are full.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    root = tk.Tk()
    try:
        App(root)
    except Exception as e:  # show startup errors instead of silently closing
        messagebox.showerror("League Build Optimizer", f"Couldn't start:\n{e}")
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
