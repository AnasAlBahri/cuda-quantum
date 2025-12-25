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
    ap.add_argument("--only-ok", action="store_true")
    ap.add_argument("--baseline", default="qpp-cpu")
    ap.add_argument("--max-circuits", type=int, default=None)
    ap.add_argument("--save-pdf", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    for col in ("time_sec", "tvd", "qubits", "depth"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if args.only_ok and "status" in df.columns:
        df = df[df["status"] == "OK"].copy()

    df = df[df["time_sec"].notna()].copy()
    if df.empty:
        print("No runtime data to plot (after filtering).")
        return 0

    g = df.groupby(["circuit", "backend"], as_index=False)["time_sec"].min()

    if args.max_circuits is not None:
        base = g[g["backend"] == args.baseline].sort_values("time_sec", ascending=False)
        keep = set(base.head(args.max_circuits)["circuit"].tolist())
        g = g[g["circuit"].isin(keep)]

    piv = g.pivot(index="circuit", columns="backend", values="time_sec").sort_index()

    fig = plt.figure(figsize=(max(10, 0.6 * len(piv.index) + 2), 6))
    ax = fig.add_subplot(1, 1, 1)
    piv.plot(kind="bar", ax=ax)
    ax.set_title("Runtime (median wall time) by Circuit and Backend")
    ax.set_xlabel("Circuit")
    ax.set_ylabel("Time (sec)")
    ax.legend(title="Backend", loc="best")
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    out_png = outdir / "runtime_by_circuit_backend.png"
    fig.savefig(out_png, dpi=300)
    if args.save_pdf:
        fig.savefig(outdir / "runtime_by_circuit_backend.pdf")
    plt.close(fig)

    if args.baseline in piv.columns:
        base = piv[args.baseline]
        speed = piv.apply(lambda col: base / col)
        if args.baseline in speed.columns:
            speed = speed.drop(columns=[args.baseline])

        if speed.shape[1] > 0:
            fig2 = plt.figure(figsize=(max(10, 0.6 * len(speed.index) + 2), 6))
            ax2 = fig2.add_subplot(1, 1, 1)
            speed.plot(kind="bar", ax=ax2)
            ax2.axhline(1.0, linestyle="--", linewidth=1)
            ax2.set_title(f"Speedup vs {args.baseline} (baseline_time / backend_time)")
            ax2.set_xlabel("Circuit")
            ax2.set_ylabel("Speedup (x)")
            ax2.legend(title="Backend", loc="best")
            ax2.tick_params(axis="x", labelrotation=45, labelsize=9)
            fig2.tight_layout()

            out_png2 = outdir / f"speedup_vs_{args.baseline}.png"
            fig2.savefig(out_png2, dpi=300)
            if args.save_pdf:
                fig2.savefig(outdir / f"speedup_vs_{args.baseline}.pdf")
            plt.close(fig2)
            print(f"Wrote: {out_png2}")
        else:
            print("Speedup plot skipped (no non-baseline backends present).")
    else:
        print(f"Speedup plot skipped (baseline '{args.baseline}' not present).")

    print(f"Wrote: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
