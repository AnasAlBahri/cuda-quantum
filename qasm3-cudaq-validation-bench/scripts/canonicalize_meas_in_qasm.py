#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

QREG = re.compile(r'^\s*qubit\s*\[\s*(\d+)\s*\]\s*([A-Za-z_]\w*)\s*;\s*$')
CREG = re.compile(r'^\s*bit\s*\[\s*(\d+)\s*\]\s*([A-Za-z_]\w*)\s*;\s*$')

# Matches BOTH:
#   measure q[0] -> c[0];
#   c[0] = measure q[0];
MEAS_ANY = re.compile(
    r'^\s*(?:'
    r'measure\b.*?;'
    r'|'
    r'[A-Za-z_]\w*\s*\[\s*\d+\s*\]\s*=\s*measure\b.*?;'
    r')\s*(//.*)?\s*$'
)

def canonicalize(text: str) -> str:
    qname = None
    qn = None
    cname = None
    cn = None

    kept: list[str] = []
    for ln in text.splitlines(True):
        mq = QREG.match(ln)
        if mq:
            qn = int(mq.group(1))
            qname = mq.group(2)

        mc = CREG.match(ln)
        if mc:
            cn = int(mc.group(1))
            cname = mc.group(2)

        # drop all existing measurement statements (both syntaxes)
        if MEAS_ANY.match(ln.strip("\n")):
            continue

        kept.append(ln)

    if qname is None or cname is None or qn is None or cn is None:
        return text

    n = min(qn, cn)
    if kept and not kept[-1].endswith("\n"):
        kept[-1] += "\n"

    kept.append("\n// Canonicalized measurements: c[i] = measure q[i]\n")
    for i in range(n):
        kept.append(f"{cname}[{i}] = measure {qname}[{i}];\n")

    return "".join(kept)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.qasm"))
    if not files:
        raise SystemExit(f"No .qasm files found in {in_dir}")

    changed = 0
    for f in files:
        txt = f.read_text(encoding="utf-8")
        new = canonicalize(txt)
        if new != txt:
            changed += 1
        (out_dir / f.name).write_text(new, encoding="utf-8")

    print(f"Wrote {len(files)} files to {out_dir} (measurement-canonicalized in {changed} files).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
