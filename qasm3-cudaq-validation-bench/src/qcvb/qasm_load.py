from __future__ import annotations
from pathlib import Path

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def list_qasm_files(qasm_dir: Path) -> list[Path]:
    exts = {".qasm", ".qasm3", ".oq3"}
    files = [p for p in sorted(qasm_dir.rglob("*")) if p.suffix.lower() in exts]
    return [p for p in files if p.is_file()]
