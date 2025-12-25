from pathlib import Path

# NOTE: keep these scripts as plain strings in one file so you never fight heredocs again.

PLOT_TVD = """<PASTE THE FULL plot_tvd.py CONTENT HERE>"""
PLOT_RUNTIME = """<PASTE THE FULL plot_runtime.py CONTENT HERE>"""

Path("scripts/plot_tvd.py").write_text(PLOT_TVD, encoding="utf-8")
Path("scripts/plot_runtime.py").write_text(PLOT_RUNTIME, encoding="utf-8")
print("Restored plot scripts.")
