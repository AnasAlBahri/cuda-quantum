#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--tvd-thresh", type=float, default=0.02)
    ap.add_argument("--only-ok", action="store_true")
    ap.add_argument("--max-circuits", type=int, default=None)
    ap.add_argument("--save-pdf", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    for col in ("tvd", "time_sec", "qubits", "clbits", "depth"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if args.only_ok and "status" in df.columns:
        df = df[df["status"] == "OK"].copy()

    df = df[df["tvd"].notna()].copy()
    if df.empty:
        print("No TVD data to plot (after filtering).")
        return 0

    g = df.groupby(["circuit", "backend"], as_index=False)["tvd"].min()

    if args.max_circuits is not None:
        worst = g.groupby("circuit", as_index=False)["tvd"].max().sort_values("tvd", ascending=False)
        keep = set(worst.head(args.max_circuits)["circuit"].tolist())
        g = g[g["circuit"].isin(keep)]

    piv = g.pivot(index="circuit", columns="backend", values="tvd").sort_index()

    fig = plt.figure(figsize=(max(10, 0.6 * len(piv.index) + 2), 6))
    ax = fig.add_subplot(1, 1, 1)
    piv.plot(kind="bar", ax=ax)
    ax.axhline(args.tvd_thresh, linestyle="--", linewidth=1)
    ax.set_title("TVD by Circuit and Backend")
    ax.set_xlabel("Circuit")
    ax.set_ylabel("Total Variation Distance (TVD)")
    ax.legend(title="Backend", loc="best")
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    out_png = outdir / "tvd_by_circuit_backend.png"
    fig.savefig(out_png, dpi=300)
    if args.save_pdf:
        fig.savefig(outdir / "tvd_by_circuit_backend.pdf")
    plt.close(fig)

    avg = g.groupby("backend", as_index=False)["tvd"].mean().sort_values("tvd", ascending=True)

    fig2 = plt.figure(figsize=(8, 4.5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.bar(avg["backend"], avg["tvd"])
    ax2.axhline(args.tvd_thresh, linestyle="--", linewidth=1)
    ax2.set_title("Mean TVD per Backend")
    ax2.set_xlabel("Backend")
    ax2.set_ylabel("Mean TVD")
    ax2.tick_params(axis="x", labelrotation=30)
    fig2.tight_layout()

    out_png2 = outdir / "tvd_mean_by_backend.png"
    fig2.savefig(out_png2, dpi=300)
    if args.save_pdf:
        fig2.savefig(outdir / "tvd_mean_by_backend.pdf")
    plt.close(fig2)

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_png2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
