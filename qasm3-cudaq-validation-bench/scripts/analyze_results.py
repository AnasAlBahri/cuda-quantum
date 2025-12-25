#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, re
from pathlib import Path
from statistics import median
import matplotlib.pyplot as plt

RE_GHZ = re.compile(r"^ghz_(\d+)$")
RE_CLIFF = re.compile(r"^clifford_n(\d+)_d(\d+)_s(\d+)$")

def parse_family_and_n(circuit: str):
    m = RE_GHZ.match(circuit)
    if m:
        return "ghz", int(m.group(1))
    m = RE_CLIFF.match(circuit)
    if m:
        return "clifford", int(m.group(1))
    return "other", None

def read_rows(csv_path: Path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status") != "OK":
                continue
            ts = row.get("time_sec")
            if ts in (None, "", "None"):
                continue
            fam, n = parse_family_and_n(row["circuit"])
            if n is None:
                continue
            row["_family"] = fam
            row["_n"] = n
            row["_backend"] = row["backend"]
            row["_time_sec"] = float(ts)
            tvd = row.get("tvd")
            row["_tvd"] = None if tvd in (None, "", "None") else float(tvd)
            rows.append(row)
    return rows

def median_by_backend_n(rows, family: str, metric_key: str):
    agg = {}
    for r in rows:
        if r["_family"] != family:
            continue
        if metric_key == "_tvd" and r["_tvd"] is None:
            continue
        key = (r["_backend"], r["_n"])
        agg.setdefault(key, []).append(r[metric_key])
    return {k: median(v) for k, v in agg.items()}

def plot_lines(points, ylabel: str, title: str, out_png: Path):
    backends = sorted(set(b for (b, _) in points.keys()))
    plt.figure()
    for b in backends:
        xs = sorted(n for (bb, n) in points.keys() if bb == b)
        ys = [points[(b, n)] for n in xs]
        plt.plot(xs, ys, marker="o", label=b)
    plt.xlabel("n (qubits)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="results/plots")
    args = ap.parse_args()

    rows = read_rows(Path(args.csv))
    outdir = Path(args.outdir)

    for fam in ("ghz", "clifford"):
        pts = median_by_backend_n(rows, fam, "_time_sec")
        if pts:
            plot_lines(pts, "median wall time (s)", f"Runtime scaling ({fam})", outdir / f"runtime_{fam}.png")

    pts = median_by_backend_n(rows, "ghz", "_tvd")
    if pts:
        plot_lines(pts, "median TVD", "TVD vs Qiskit (GHZ)", outdir / "tvd_ghz.png")

    print(f"Wrote plots to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
