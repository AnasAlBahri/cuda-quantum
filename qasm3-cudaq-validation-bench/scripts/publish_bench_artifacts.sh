#!/usr/bin/env bash
set -euo pipefail

DATE_TAG="${1:-$(date +%F)}"  # e.g., 2026-01-04
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTDIR="$ROOT/docs/benchmarks/$DATE_TAG"
mkdir -p "$OUTDIR"

cp "$ROOT/results/plots/"*.png "$OUTDIR/"
cp "$ROOT/results/benchmarks.csv" "$OUTDIR/" || true

echo "Published artifacts to: $OUTDIR"
