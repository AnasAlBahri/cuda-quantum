#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

SWAP_LINE = re.compile(r"^(\s*)swap\s+([^,;]+)\s*,\s*([^;]+)\s*;\s*(//.*)?\s*$")

def expand_swap(text: str) -> str:
    out_lines = []
    for line in text.splitlines(True):
        m = SWAP_LINE.match(line.rstrip("\n"))
        if not m:
            out_lines.append(line)
            continue

        indent, a, b, comment = m.group(1), m.group(2).strip(), m.group(3).strip(), m.group(4) or ""
        cmt = f" {comment}" if comment else ""
        out_lines.append(f"{indent}cx {a}, {b};{cmt}\n")
        out_lines.append(f"{indent}cx {b}, {a};\n")
        out_lines.append(f"{indent}cx {a}, {b};\n")
    return "".join(out_lines)

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
        new = expand_swap(txt)
        if new != txt:
            changed += 1
        (out_dir / f.name).write_text(new, encoding="utf-8")

    print(f"Wrote {len(files)} files to {out_dir} (swap-expanded in {changed} files).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
