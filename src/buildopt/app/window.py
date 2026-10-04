"""Compact window beside the client (Tk), with a console fallback."""

from __future__ import annotations

import queue
import sys
import threading


class ConsoleUI:
    def __init__(self, stream=None, interactive: bool = True):
        self.stream = stream or sys.stdout
        self.interactive = interactive

    def show(self, view: dict) -> None:
        w = self.stream.write
        w("\n" + "=" * 72 + "\n")
        w(view["header"] + ("  [read-only]" if view.get("read_only") else "") + "\n")
        w(f"Runes : {view['runes']}\nBuild : {view['build']}\nWin   : {view['win']}\n")
        for line in view["why"]:
            w(f"  why: {line}\n")
        if view.get("runner_up"):
            w(f"Runner-up: {view['runner_up']}\n")
        for line in view.get("other_builds", []):
            w(f"  also: {line}\n")
        if view.get("change"):
            w(f"Changed: {view['change']}\n")
        for line in view.get("extras", []):
            w(f"{line}\n")
        w(f"(scored in {view.get('score_ms', 0):.0f} ms)\n")
        self.stream.flush()

    def status(self, message: str) -> None:
        self.stream.write(f"[status] {message}\n")
        self.stream.flush()

    def ask_page(self, pages: list[dict]) -> int | None:
        if not self.interactive:
            return None
        self.stream.write("Rune page limit reached. Which page may the app own and overwrite?\n")
        for i, p in enumerate(pages):
            self.stream.write(f"  {i + 1}. {p.get('name')}\n")
        self.stream.write("  0. none (skip rune import)\n> ")
        self.stream.flush()
        try:
            choice = int(input().strip() or "0")
        except (ValueError, EOFError):
            return None
        return pages[choice - 1]["id"] if 1 <= choice <= len(pages) else None


class TkUI:
    """Small always-on-top window. Calls from other threads are marshalled through a queue."""

    def __init__(self, title: str = "Build Optimizer"):
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes("-topmost", True)
        self.root.geometry("460x330")
        self.queue: queue.Queue = queue.Queue()
        self.text = tk.Text(self.root, wrap="word", font=("Segoe UI", 10), borderwidth=0, padx=8, pady=8)
        self.text.pack(fill="both", expand=True)
        self.status_var = tk.StringVar(value="Waiting for the League client")
        tk.Label(self.root, textvariable=self.status_var, anchor="w", font=("Segoe UI", 9), fg="#555").pack(fill="x")
        self.root.after(100, self._drain)

    def _drain(self):
        while True:
            try:
                fn, args = self.queue.get_nowait()
            except queue.Empty:
                break
            fn(*args)
        self.root.after(100, self._drain)

    def show(self, view: dict) -> None:
        self.queue.put((self._show, (view,)))

    def _show(self, view: dict) -> None:
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.tag_configure("h", font=("Segoe UI", 9), foreground="#666")
        t.tag_configure("b", font=("Segoe UI", 11, "bold"))
        t.insert("end", view["header"] + ("  [read-only]" if view.get("read_only") else "") + "\n", "h")
        t.insert("end", view["runes"] + "\n", "b")
        t.insert("end", view["build"] + "\n", "b")
        t.insert("end", view["win"] + "\n\n")
        for line in view["why"]:
            t.insert("end", f"• {line}\n")
        if view.get("runner_up"):
            t.insert("end", f"\nRunner-up: {view['runner_up']}\n")
        if view.get("change"):
            t.insert("end", f"Changed: {view['change']}\n")
        for line in view.get("extras", []):
            t.insert("end", f"{line}\n", "h")
        t.configure(state="disabled")

    def status(self, message: str) -> None:
        self.queue.put((self.status_var.set, (message,)))

    def ask_page(self, pages: list[dict]) -> int | None:
        """Blocking question from a worker thread, answered in the Tk thread."""
        done = threading.Event()
        result: list = [None]

        def ask():
            win = self.tk.Toplevel(self.root)
            win.title("Rune page limit reached")
            self.tk.Label(win, text="Which page may the app own and overwrite?\n(asked once)").pack(padx=10, pady=6)
            lb = self.tk.Listbox(win, height=min(10, len(pages)))
            for p in pages:
                lb.insert("end", p.get("name"))
            lb.pack(padx=10, fill="both")

            def ok():
                sel = lb.curselection()
                result[0] = pages[sel[0]]["id"] if sel else None
                win.destroy()
                done.set()

            def cancel():
                win.destroy()
                done.set()

            self.tk.Button(win, text="Use this page", command=ok).pack(side="left", padx=10, pady=8)
            self.tk.Button(win, text="Don't import runes", command=cancel).pack(side="right", padx=10, pady=8)

        self.queue.put((ask, ()))
        done.wait()
        return result[0]

    def mainloop(self) -> None:
        self.root.mainloop()


def make_ui(headless: bool):
    if not headless:
        try:
            return TkUI()
        except Exception:  # no display or no tkinter
            pass
    return ConsoleUI()
