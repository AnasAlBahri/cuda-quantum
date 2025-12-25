#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from metrics import avg_abs_marginal_diff, hamming_weight_tvd


# ----------------------------
# Small utilities
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def stable_seed(base: int, circuit: str, backend: str, salt: str = "") -> int:
    """Deterministic 32-bit seed from (base, circuit, backend)."""
    msg = f"{base}|{circuit}|{backend}|{salt}".encode("utf-8")
    h = hashlib.sha256(msg).digest()
    return int.from_bytes(h[:4], "little")


# ----------------------------
# Counts + TVD (self-contained)
# ----------------------------

def _extract_counts_dict(sample_res: Any) -> dict[Any, int]:
    """Extract {bitstring -> count} from CUDA-Q SampleResult and normalize to Qiskit bit order."""
    def norm_key(k: Any) -> str:
        s = safe_str(k).replace(" ", "")
        if s.startswith("0b"):
            s = s[2:]
        # Reverse bit order to match Qiskit convention (c[n-1]...c[0]).
        if s and set(s) <= {"0", "1"}:
            s = s[::-1]
        return s

    def to_items(obj):
        """Return iterable of (k,v) pairs from dict-like / iterable / CUDA-Q counts objects."""
        if obj is None:
            return None

        # If it's already a dict
        if isinstance(obj, dict):
            return obj.items()

        # Common helper
        for attr in ("to_dict", "toDict", "as_dict", "asDict"):
            if hasattr(obj, attr) and callable(getattr(obj, attr)):
                try:
                    d = getattr(obj, attr)()
                    if isinstance(d, dict):
                        return d.items()
                except Exception:
                    pass

        # If it has .items()
        if hasattr(obj, "items") and callable(getattr(obj, "items")):
            try:
                return list(obj.items())
            except Exception:
                pass

        # Try dict(obj) (works for many iterable mappings)
        try:
            d = dict(obj)
            if isinstance(d, dict):
                return d.items()
        except Exception:
            pass

        # Try list(obj) if it yields pairs
        try:
            it = list(obj)
            if it and isinstance(it[0], (tuple, list)) and len(it[0]) == 2:
                return it
        except Exception:
            pass

        return None

    # 1) Handle SampleResult: counts may be method or attribute
    if hasattr(sample_res, "counts"):
        try:
            c = sample_res.counts() if callable(getattr(sample_res, "counts")) else sample_res.counts
            items = to_items(c)
            if items is not None:
                d = {norm_key(k): int(v) for k, v in items}
                # Pad keys to consistent width (helps if leading zeros are dropped)
                if d:
                    w = max(len(k) for k in d.keys())
                    d = {k.zfill(w): v for k, v in d.items()}
                return d
        except Exception:
            pass

    # 2) Other possible accessors
    for attr in ("get_counts", "getCounts", "counts_dict", "to_dict", "toDict"):
        if hasattr(sample_res, attr):
            try:
                obj = getattr(sample_res, attr)
                m = obj() if callable(obj) else obj
                items = to_items(m)
                if items is not None:
                    d = {norm_key(k): int(v) for k, v in items}
                    if d:
                        w = max(len(k) for k in d.keys())
                        d = {k.zfill(w): v for k, v in d.items()}
                    return d
            except Exception:
                continue

    # 3) If the result itself is iterable / dict-like
    items = to_items(sample_res)
    if items is not None:
        d = {norm_key(k): int(v) for k, v in items}
        if d:
            w = max(len(k) for k in d.keys())
            d = {k.zfill(w): v for k, v in d.items()}
        return d

    raise TypeError(f"Don't know how to extract counts from sample result type: {type(sample_res)}")
def _infer_num_qubits_from_counts(raw_counts: Mapping[Any, Any]) -> int:
    max_len = 0
    max_int = 0
    for k in raw_counts.keys():
        if isinstance(k, int):
            max_int = max(max_int, k)
            continue
        s = str(k).strip().replace(" ", "")
        if s.startswith("0b"):
            s = s[2:]
        if s and set(s) <= {"0", "1"}:
            max_len = max(max_len, len(s))
    if max_len > 0:
        return max_len
    return max(1, max_int.bit_length())

def normalize_counts_local(raw_counts: Mapping[Any, Any], width_hint: Optional[int] = None) -> dict[str, int]:
    """Normalize to bitstring keys, left padded to width_hint if given."""
    out: dict[str, int] = {}
    w = int(width_hint) if width_hint is not None else None
    for k, v in raw_counts.items():
        if isinstance(k, int):
            if w is None:
                # Delay width decision; we'll pad later after we infer.
                s = format(k, "b")
            else:
                s = format(k, f"0{w}b")
        else:
            s = str(k).strip().replace(" ", "")
            if s.startswith("0b"):
                s = s[2:]
            if w is not None:
                s = s.zfill(w)
        out[s] = out.get(s, 0) + int(v)
    if w is not None:
        out = {k.zfill(w): int(v) for k, v in out.items()}
    return out

def tvd_from_counts(a: Mapping[str, int], b: Mapping[str, int]) -> float:
    sa = sum(a.values())
    sb = sum(b.values())
    if sa <= 0 or sb <= 0:
        raise ValueError(f"Invalid count totals: sa={sa} sb={sb}")
    keys = set(a.keys()) | set(b.keys())
    tvd = 0.0
    for k in keys:
        pa = a.get(k, 0) / sa
        pb = b.get(k, 0) / sb
        tvd += abs(pa - pb)
    return 0.5 * tvd


# ----------------------------
# QASM discovery
# ----------------------------

def iter_qasm_files(qasm_dirs: List[Path], recursive: bool = True) -> List[Path]:
    files: List[Path] = []
    for d in qasm_dirs:
        if not d.exists():
            continue
        it = d.rglob("*.qasm") if recursive else d.glob("*.qasm")
        for p in it:
            if p.is_file():
                files.append(p.resolve())
    files = sorted(set(files))
    return files


# ----------------------------
# Qiskit helpers
# ----------------------------

def get_qiskit_circuit(qasm_path: Path):
    from qiskit import qasm3
    return qasm3.load(str(qasm_path))

def qiskit_reference_counts(qasm_path: Path, shots: int, seed: int) -> Tuple[dict[str, int], int, int, int]:
    """
    Returns (counts, qubits, clbits, depth) using Statevector sampling (no Aer).
    """
    import numpy as np
    from qiskit.quantum_info import Statevector

    qc = get_qiskit_circuit(qasm_path)
    n = int(qc.num_qubits)
    c = int(getattr(qc, "num_clbits", 0))
    d = int(qc.depth())

    qc_u = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_u)
    probs = np.asarray(sv.probabilities(), dtype=float)

    rng = np.random.default_rng(seed)
    samples = rng.choice(len(probs), size=shots, p=probs)

    counts: dict[str, int] = {}
    for s in samples:
        b = format(int(s), f"0{n}b")
        counts[b] = counts.get(b, 0) + 1

    return counts, n, c, d


# ----------------------------
# CUDA-Q backend handling
# ----------------------------

def list_cudaq_version() -> str:
    try:
        import cudaq
        return getattr(cudaq, "__version__", "unknown")
    except Exception as e:
        return f"cudaq-import-failed: {e}"

def has_cudaq_target(name: str) -> bool:
    import cudaq
    try:
        if hasattr(cudaq, "has_target"):
            return bool(cudaq.has_target(name))
    except Exception:
        pass
    # Fallback: try set_target to test
    try:
        cudaq.set_target(name)
        return True
    except Exception:
        return False

def backend_target_candidates(backend_id: str) -> List[str]:
    low = backend_id.strip().lower()
    if low in ("qpp", "qpp-cpu", "qpp_cpu"):
        return ["qpp-cpu", "qpp"]
    if low in ("stim", "stim-sampler", "stim_sampler", "stim-cpu"):
        # your build exposes "stim"
        return ["stim", "stim_sampler"]
    if low in ("cuquantum", "cuquantum-gpu", "cuquantum_gpu", "gpu", "nvidia"):
        # your build likely doesn't have this, but keep candidates for portability
        return ["cuquantum", "nvidia", "cuquantum-gpu", "cuquantum_gpu", "nvidia-mgpu", "cuquantum-mgpu"]
    return [backend_id]

def is_stim_compatible(qasm_path: Path) -> Tuple[bool, str]:
    """
    Conservative compatibility:
    - Remove final measurements, then require only a small supported set of gate names.
    - Also require Clifford-ness.
    """
    try:
        qc = get_qiskit_circuit(qasm_path)
    except Exception as e:
        return False, f"Qiskit parse failed: {e}"

    qc_u = qc.remove_final_measurements(inplace=False)

    supported = {
        "id", "x", "y", "z", "h", "s", "sdg",
        "cx", "cz", "swap",
        "barrier",
    }
    bad = sorted({inst.operation.name for inst in qc_u.data if inst.operation.name not in supported})
    if bad:
        return False, f"Stim target unsupported gate(s): {bad}"

    try:
        from qiskit.quantum_info import Clifford  # type: ignore
        Clifford(qc_u)
        return True, ""
    except Exception as e:
        return False, f"Non-Clifford / not stabilizer-compatible: {e}"

def transpile_to_cudaq_kernel(qasm_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Project-specific transpiler entrypoint:
      qcvb.transpile_adapter.qasm3_to_cudaq_kernel(Path) -> kernel
    """
    from qcvb import transpile_adapter as ta

    kernel = ta.qasm3_to_cudaq_kernel(qasm_path)
    return kernel, {}

def cudaq_sample_once(kernel: Any, target: str, shots: int, width_hint: Optional[int] = None, seed: Optional[int] = None) -> Tuple[dict[str, int], int, float, float]:
    """
    Returns (counts, inferred_qubits, sample_time_sec, wall_time_sec)
    """
    import cudaq

    # ensure correct target for this run
    cudaq.set_target(target)

    # Deterministic sampling (set_target() may reset backend state)
    if seed is not None:
        cudaq.set_random_seed(int(seed))

    t0 = time.perf_counter()
    tS0 = time.perf_counter()
    try:
        try:
            sample_res = cudaq.sample(kernel, shots_count=shots)
        except TypeError:
            sample_res = cudaq.sample(kernel, shots=shots)
    finally:
        tS1 = time.perf_counter()
    t1 = time.perf_counter()

    raw = _extract_counts_dict(sample_res)
    n = int(width_hint) if width_hint is not None else _infer_num_qubits_from_counts(raw)
    counts = normalize_counts_local(raw, width_hint=n)

    total = sum(counts.values())
    if total != shots:
        raise RuntimeError(f"CUDA-Q counts sum to {total}, expected {shots}")

    sample_sec = tS1 - tS0
    wall_sec = t1 - t0
    return counts, n, sample_sec, wall_sec


# ----------------------------
# Result schema
# ----------------------------

@dataclass
class BenchRow:
    timestamp: str
    circuit: str
    backend: str
    target_used: str
    qasm_path: str
    shots: int
    warmup: int
    repeats: int
    qubits: int
    clbits: int
    depth: int
    time_sec: Optional[float]
    time_mean: Optional[float]
    time_std: Optional[float]
    sample_mean: Optional[float]
    tvd_ref: str
    tvd: Optional[float]
    tvd_thresh: float
    passed: Optional[bool]
    status: str
    reason: str


def write_jsonl(path: Path, rec: Dict[str, Any], append: bool) -> None:
    ensure_dir(path.parent)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def write_csv_header_if_needed(path: Path, fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

def append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)



def _infer_nbits_from_qasm(qasm_path: Path) -> int:
    import re
    txt = qasm_path.read_text(encoding="utf-8")
    m = re.search(r"^\s*bit\s*\[\s*(\d+)\s*\]\s*[A-Za-z_]\w*\s*;\s*$", txt, re.M)
    if not m:
        raise ValueError(f"Could not infer classical bit width from {qasm_path}")
    return int(m.group(1))


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm-dir", action="append", required=True, help="QASM directory (repeatable).")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--backends", nargs="+", required=True)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tvd-ref", choices=["qiskit", "qpp"], default="qiskit")
    ap.add_argument("--tvd-thresh", type=float, default=0.02)
    ap.add_argument("--marg-thresh", type=float, default=0.05)
    ap.add_argument("--wt-thresh", type=float, default=0.05)
    ap.add_argument("--results-csv", default="results/benchmarks.csv")
    ap.add_argument("--results-jsonl", default="results/benchmarks.jsonl")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--dump-counts", action="store_true")
    args = ap.parse_args()

    qasm_dirs = [Path(d).expanduser().resolve() for d in args.qasm_dir]
    qasm_files = iter_qasm_files(qasm_dirs, recursive=not args.no_recursive)

    # Discovery sanity print (this is what you need right now)
    print("Discovered QASM files: total=%d ghz=%d clifford=%d" % (
        len(qasm_files),
        sum(1 for f in qasm_files if f.stem.startswith("ghz_")),
        sum(1 for f in qasm_files if f.stem.startswith("clifford_")),
    ))

    results_csv = Path(args.results_csv)
    results_jsonl = Path(args.results_jsonl)
    ensure_dir(results_csv.parent)
    ensure_dir(results_jsonl.parent)

    fieldnames = [
        "timestamp","circuit","backend","target_used","qasm_path",
        "shots","warmup","repeats","qubits","clbits","depth",
        "time_sec","time_mean","time_std","sample_mean",
        "tvd_ref","tvd","tvd_thresh","passed","status","reason"
    ]
    if not args.append:
        if results_csv.exists():
            results_csv.unlink()
        if results_jsonl.exists():
            results_jsonl.unlink()
    write_csv_header_if_needed(results_csv, fieldnames)

    # Write meta header
    meta = {
        "type": "meta",
        "timestamp": utc_now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "env": {
            "QCVB_TRANSPILER_MODE": os.environ.get("QCVB_TRANSPILER_MODE", ""),
        },
        "cudaq_version": list_cudaq_version(),
    }
    write_jsonl(results_jsonl, meta, append=True)

    # Precompute qiskit reference once per file (counts + metadata).
    qiskit_cache: Dict[str, Tuple[dict[str,int], int, int, int]] = {}

    # If tvd_ref=qpp, we need qpp counts too (per circuit).
    qpp_cache: Dict[str, dict[str,int]] = {}

    for qasm_path in qasm_files:
        circuit_id = qasm_path.stem  # IMPORTANT: full stem (keeps clifford_* unique)
        family = "ghz" if circuit_id.startswith("ghz_") else ("clifford" if circuit_id.startswith("clifford_") else "other")
        nbits = _infer_nbits_from_qasm(qasm_path)

        # Qiskit reference + circuit metadata
        ref_counts: Optional[dict[str,int]] = None
        qubits = clbits = depth = 0
        ref_err = ""

        try:
            if circuit_id not in qiskit_cache:
                cts, n, c, d = qiskit_reference_counts(qasm_path, shots=args.shots, seed=args.seed)
                qiskit_cache[circuit_id] = (cts, n, c, d)
            ref_counts, qubits, clbits, depth = qiskit_cache[circuit_id]
        except Exception as e:
            ref_err = f"Qiskit ref failed: {e}"

        # Transpile once per circuit
        kernel = None
        meta_dict: Dict[str, Any] = {}
        try:
            kernel, meta_dict = transpile_to_cudaq_kernel(qasm_path)
        except Exception as e:
            tb = traceback.format_exc()
            for backend_id in args.backends:
                row = BenchRow(
                    timestamp=utc_now_iso(),
                    circuit=circuit_id,
                    backend=backend_id,
                    target_used="",
                    qasm_path=str(qasm_path),
                    shots=args.shots,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    qubits=qubits,
                    clbits=clbits,
                    depth=depth,
                    time_sec=None,
                    time_mean=None,
                    time_std=None,
                    sample_mean=None,
                    tvd_ref=args.tvd_ref,
                    tvd=None,
                    tvd_thresh=float(args.tvd_thresh),
                    passed=None,
                    status="ERR",
                    reason=f"Transpile failed: {e}",
                )
                append_csv_row(results_csv, fieldnames, asdict(row))
                write_jsonl(results_jsonl, {
                    "type": "row",
                    **asdict(row),
                    "traceback": tb,
                    "transpile_meta": meta_dict,
                }, append=True)
                print(f"{circuit_id:<30} {backend_id:<9} ERR   wall_med= tvd= reason=Transpile failed: {e}")
            continue

        # For each backend
        for backend_id in args.backends:
            status = "OK"
            reason = ""
            target_used = ""
            tvd_val: Optional[float] = None
            passed: Optional[bool] = None
            med_wall: Optional[float] = None
            mean_wall: Optional[float] = None
            std_wall: Optional[float] = None
            mean_sample: Optional[float] = None

            # Stim compatibility check
            if backend_id.strip().lower().startswith("stim"):
                ok, why = is_stim_compatible(qasm_path)
                if not ok:
                    status = "SKIP"
                    reason = why

            # Determine actual CUDA-Q target name
            last_exc: Optional[Exception] = None
            if status == "OK":
                chosen = None
                for cand in backend_target_candidates(backend_id):
                    try:
                        import cudaq
                        if hasattr(cudaq, "has_target") and not cudaq.has_target(cand):
                            last_exc = RuntimeError(f"Target not available: {cand}")
                            continue
                    except Exception:
                        pass
                    if has_cudaq_target(cand):
                        chosen = cand
                        break
                if chosen is None:
                    status = "SKIP"
                    reason = safe_str(last_exc) if last_exc else f"No available target for backend {backend_id}"
                else:
                    target_used = chosen

            # If OK, run warmup + repeats


            # Deterministic bitstring width (prevents dropped leading zeros)
            width_hint = clbits or qubits
            if not width_hint:
                try:
                    width_hint = _infer_nbits_from_qasm(qasm_path)
                except Exception:
                    width_hint = None

            counts_dut: Optional[dict[str,int]] = None
            if status == "OK":
                try:
                    # warmup (ignored)
                    for _ in range(max(0, int(args.warmup))):
                        cudaq_sample_once(kernel, target_used, args.shots, width_hint=width_hint, seed=stable_seed(args.seed, circuit_id, backend_id, salt=f'warm{_}'))

                    wall_times: List[float] = []
                    sample_times: List[float] = []
                    last_counts: Optional[dict[str,int]] = None
                    last_n: int = 0

                    for _ in range(max(1, int(args.repeats))):
                        cts, n_infer, sample_sec, wall_sec = cudaq_sample_once(kernel, target_used, args.shots, width_hint=width_hint, seed=stable_seed(args.seed, circuit_id, backend_id, salt=f'rep{_}'))
                        wall_times.append(wall_sec)
                        sample_times.append(sample_sec)
                        last_counts = cts
                        last_n = n_infer

                    counts_dut = last_counts if last_counts is not None else {}
                    if qubits == 0:
                        qubits = last_n

                    med_wall = statistics.median(wall_times)
                    mean_wall = statistics.mean(wall_times)
                    std_wall = statistics.pstdev(wall_times) if len(wall_times) > 1 else 0.0
                    mean_sample = statistics.mean(sample_times)

                except Exception as e:
                    status = "ERR"
                    reason = f"{e}"
                    tb = traceback.format_exc()

                    row = BenchRow(
                        timestamp=utc_now_iso(),
                        circuit=circuit_id,
                        backend=backend_id,
                        target_used=target_used,
                        qasm_path=str(qasm_path),
                        shots=args.shots,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        qubits=qubits,
                        clbits=clbits,
                        depth=depth,
                        time_sec=None,
                        time_mean=None,
                        time_std=None,
                        sample_mean=None,
                        tvd_ref=args.tvd_ref,
                        tvd=None,
                        tvd_thresh=float(args.tvd_thresh),
                        passed=None,
                        status=status,
                        reason=reason,
                    )
                    append_csv_row(results_csv, fieldnames, asdict(row))
                    write_jsonl(results_jsonl, {
                        "type": "row",
                        **asdict(row),
                        "traceback": tb,
                        "transpile_meta": meta_dict,
                    }, append=True)
                    print(f"{circuit_id:<30} {backend_id:<9} ERR   wall_med= tvd= reason={reason}")
                    continue

            # TVD computation
            if status == "OK":
                try:
                    if args.tvd_ref == "qiskit":
                        if ref_counts is None:
                            raise RuntimeError(ref_err or "No qiskit ref counts")
                        tvd_val = tvd_from_counts(counts_dut or {}, ref_counts)
                    else:
                        # baseline qpp counts
                        if circuit_id not in qpp_cache:
                            # compute baseline once
                            base_kernel, _ = transpile_to_cudaq_kernel(qasm_path)
                            base_counts, _, _, _ = cudaq_sample_once(base_kernel, "qpp-cpu", args.shots, width_hint=width_hint, seed=stable_seed(args.seed, circuit_id, 'qpp-baseline'))
                            qpp_cache[circuit_id] = base_counts
                        tvd_val = tvd_from_counts(counts_dut or {}, qpp_cache[circuit_id])

                    passed = (tvd_val <= float(args.tvd_thresh))
                except Exception as e:
                    status = "ERR"
                    reason = f"TVD failed: {e}"
                    tb = traceback.format_exc()
                    row = BenchRow(
                        timestamp=utc_now_iso(),
                        circuit=circuit_id,
                        backend=backend_id,
                        target_used=target_used,
                        qasm_path=str(qasm_path),
                        shots=args.shots,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        qubits=qubits,
                        clbits=clbits,
                        depth=depth,
                        time_sec=med_wall,
                        time_mean=mean_wall,
                        time_std=std_wall,
                        sample_mean=mean_sample,
                        tvd_ref=args.tvd_ref,
                        tvd=None,
                        tvd_thresh=float(args.tvd_thresh),
                        passed=None,
                        status=status,
                        reason=reason,
                    )
                    append_csv_row(results_csv, fieldnames, asdict(row))
                    write_jsonl(results_jsonl, {
                        "type": "row",
                        **asdict(row),
                        "traceback": tb,
                        "transpile_meta": meta_dict,
                    }, append=True)
                    print(f"{circuit_id:<30} {backend_id:<9} ERR   wall_med= tvd= reason={reason}")
                    continue

            # Dump counts if requested
            if args.dump_counts and status == "OK":
                out_counts_dir = Path("results") / "counts"
                ensure_dir(out_counts_dir)
                out_path = out_counts_dir / f"{circuit_id}__{backend_id}.json"
                payload = {
                    "circuit": circuit_id,
                    "backend": backend_id,
                    "target_used": target_used,
                    "shots": args.shots,
                    "counts": counts_dut,
                    "tvd_ref": args.tvd_ref,
                    "tvd": tvd_val,
                    "meta": meta_dict,
                }
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            # Write row
            row = BenchRow(
                timestamp=utc_now_iso(),
                circuit=circuit_id,
                backend=backend_id,
                target_used=target_used,
                qasm_path=str(qasm_path),
                shots=args.shots,
                warmup=args.warmup,
                repeats=args.repeats,
                qubits=qubits,
                clbits=clbits,
                depth=depth,
                time_sec=med_wall,
                time_mean=mean_wall,
                time_std=std_wall,
                sample_mean=mean_sample,
                tvd_ref=args.tvd_ref,
                tvd=tvd_val,
                tvd_thresh=float(args.tvd_thresh),
                passed=passed,
                status=status,
                reason=reason,
            )
            append_csv_row(results_csv, fieldnames, asdict(row))
            write_jsonl(results_jsonl, {
                "type": "row",
                **asdict(row),
                "transpile_meta": meta_dict,
            }, append=True)

            # Console line similar to yours
            if status == "OK":
                print(f"{circuit_id:<30} {backend_id:<9} OK    wall_med={med_wall} tvd={tvd_val}")
            else:
                print(f"{circuit_id:<30} {backend_id:<9} {status:<4} wall_med= tvd= reason={reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
