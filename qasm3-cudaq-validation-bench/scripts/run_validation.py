from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List

from qcvb.qasm_load import list_qasm_files
from qcvb.qiskit_runner import run_qiskit_sampling
from qcvb.transpile_adapter import qasm3_to_cudaq_kernel, TranspilerNotConfigured
from qcvb.cudaq_runner import run_cudaq_sampling
from qcvb.metrics import tvd, pass_fail_tvd
from qcvb.io_utils import write_csv, append_jsonl


def _strip_key(k: str) -> str:
    # Qiskit may include spaces between registers; remove them.
    return str(k).replace(" ", "")


def _counts_to_c0_first_from_classical_key(
    counts: Dict[str, int],
    n_clbits: int,
    ref_order: str,
) -> Dict[str, int]:
    """
    Convert counts whose KEYS are classical-bit strings into canonical "c0..c_{n-1}" (c0-first) keys.

    ref_order:
      - "qiskit": key is "c_{n-1} ... c0" (Qiskit typical get_counts format)
      - "c0":     key is already "c0 c1 ... c_{n-1}"
    """
    out: Dict[str, int] = {}
    for k, v in counts.items():
        kk = _strip_key(k)
        if len(kk) != n_clbits:
            # If something odd happens, skip safely (shouldn't for your tests).
            continue

        if ref_order == "qiskit":
            # Qiskit: leftmost is highest classical bit -> reverse to c0-first
            canon = kk[::-1]
        elif ref_order == "c0":
            canon = kk
        else:
            raise ValueError(f"Unknown ref_order: {ref_order}")

        out[canon] = out.get(canon, 0) + int(v)
    return out


def _dut_qubitkey_to_c0_first_classicalkey(
    qubit_key: str,
    n_qubits: int,
    n_clbits: int,
    dut_order: str,
    meas_q2c: List[int],
) -> str:
    """
    Convert a DUT key (interpreted as *qubit bits*) into canonical classical key "c0..".
    - dut_order="cudaq": key is "q0 q1 ... q_{n-1}"
    - dut_order="qiskit": key is "q_{n-1} ... q0" (rare, but support)
    meas_q2c[q] = classical bit index that receives measure(q), or -1 if not measured.
    """
    kk = _strip_key(qubit_key)
    if len(kk) != n_qubits:
        # If CUDA-Q returns something unexpected, fall back without crashing.
        # Pad/truncate to n_qubits (best-effort).
        if len(kk) < n_qubits:
            kk = kk.rjust(n_qubits, "0")
        else:
            kk = kk[-n_qubits:]

    # Extract qubit bits as array qb[q] in {0,1}
    qb = [0] * n_qubits
    if dut_order == "cudaq":
        # kk[i] corresponds to q[i]
        for i in range(n_qubits):
            qb[i] = 1 if kk[i] == "1" else 0
    elif dut_order == "qiskit":
        # kk[0] corresponds to q[n-1]
        for i in range(n_qubits):
            qb[i] = 1 if kk[n_qubits - 1 - i] == "1" else 0
    else:
        raise ValueError(f"Unknown dut_order: {dut_order}")

    # Build classical bits c[0..]
    cb = [0] * n_clbits
    for q in range(n_qubits):
        c = meas_q2c[q] if q < len(meas_q2c) else -1
        if c is None or c < 0 or c >= n_clbits:
            continue
        cb[c] = qb[q]

    return "".join("1" if cb[i] else "0" for i in range(n_clbits))


def _normalize_dut_counts_to_c0_first(
    dut_counts: Dict[str, int],
    n_qubits: int,
    n_clbits: int,
    meas_q2c: List[int],
    dut_order: str,
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in dut_counts.items():
        canon = _dut_qubitkey_to_c0_first_classicalkey(k, n_qubits, n_clbits, dut_order, meas_q2c)
        out[canon] = out.get(canon, 0) + int(v)
    return out


def _choose_best_alignment(
    ref_counts_raw: Dict[str, int],
    dut_counts_raw: Dict[str, int],
    n_qubits: int,
    n_clbits: int,
    meas_q2c: List[int],
) -> Tuple[Dict[str, int], Dict[str, int], str, str, float]:
    """
    Try small set of plausible conventions and pick the one minimizing TVD.
    Returns: (ref_canon, dut_canon, chosen_ref_order, chosen_dut_order, dist)
    """
    ref_mode_env = os.environ.get("QCVB_REF_BITSTRING_ORDER", "auto").strip().lower()
    dut_mode_env = os.environ.get("QCVB_DUT_BITSTRING_ORDER", "auto").strip().lower()

    ref_modes = ["qiskit", "c0"] if ref_mode_env == "auto" else [ref_mode_env]
    dut_modes = ["cudaq", "qiskit"] if dut_mode_env == "auto" else [dut_mode_env]

    best = None

    for rmode in ref_modes:
        ref_canon = _counts_to_c0_first_from_classical_key(ref_counts_raw, n_clbits, rmode)
        for dmode in dut_modes:
            dut_canon = _normalize_dut_counts_to_c0_first(dut_counts_raw, n_qubits, n_clbits, meas_q2c, dmode)
            dist = tvd(ref_canon, dut_canon, width_hint=n_clbits)
            cand = (dist, ref_canon, dut_canon, rmode, dmode)
            if best is None or cand[0] < best[0]:
                best = cand

    assert best is not None
    dist, ref_canon, dut_canon, rmode, dmode = best
    return ref_canon, dut_canon, rmode, dmode, dist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm-dir", type=Path, default=Path("qasm3_tests"))
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--backend", type=str, default="qpp-cpu")
    ap.add_argument("--tvd-thresh", type=float, default=1e-3)
    ap.add_argument("--csv-out", type=Path, default=Path("results/validation.csv"))
    ap.add_argument("--jsonl-out", type=Path, default=Path("results/validation.jsonl"))
    args = ap.parse_args()

    files = list_qasm_files(args.qasm_dir)
    if not files:
        print(f"No QASM files found under {args.qasm_dir}")
        return 2

    rows: list[dict] = []
    debug = os.environ.get("QCVB_DEBUG_COUNTS", "0").strip() == "1"

    for qasm_path in files:
        print(f"\n=== {qasm_path} ===")

        # Reference run (Qiskit)
        try:
            ref = run_qiskit_sampling(qasm_path, shots=args.shots)
        except Exception as e:
            rec = {"circuit": qasm_path.name, "status": "ERROR", "where": "qiskit", "error": repr(e)}
            append_jsonl(args.jsonl_out, rec)
            print("Qiskit ERROR:", e)
            continue

        # DUT run (QASM3 -> CUDA-Q -> sample)
        try:
            kernel = qasm3_to_cudaq_kernel(qasm_path)
        except TranspilerNotConfigured as e:
            print("\nTranspiler not configured.")
            print(e)
            return 3
        except Exception as e:
            rec = {"circuit": qasm_path.name, "status": "ERROR", "where": "transpiler", "error": repr(e)}
            append_jsonl(args.jsonl_out, rec)
            print("Transpiler ERROR:", e)
            continue

        try:
            dut = run_cudaq_sampling(kernel, backend=args.backend, shots=args.shots)
        except Exception as e:
            rec = {
                "circuit": qasm_path.name,
                "status": "ERROR",
                "where": "cudaq",
                "backend": args.backend,
                "error": repr(e),
            }
            append_jsonl(args.jsonl_out, rec)
            print("CUDA-Q ERROR:", e)
            continue

        n_qubits = int(getattr(ref, "num_qubits", 0) or 0)
        n_clbits = int(getattr(ref, "num_clbits", 0) or 0)
        if n_clbits <= 0:
            # Most of your circuits have bit[n]=qubit[n]. If missing, assume same as qubits.
            n_clbits = n_qubits

        meas_q2c = getattr(kernel, "__qcvb_meas_q2c__", None)
        if meas_q2c is None:
            # If no mapping is attached, assume identity for first n_clbits.
            meas_q2c = list(range(n_qubits))
        meas_q2c = [int(x) for x in meas_q2c]

        # Canonicalize + compare
        ref_canon, dut_canon, chosen_ref_order, chosen_dut_order, dist = _choose_best_alignment(
            ref.counts, dut.counts, n_qubits, n_clbits, meas_q2c
        )
        ok = pass_fail_tvd(dist, threshold=args.tvd_thresh)

        if debug:
            print("DEBUG raw ref.counts:", dict(ref.counts))
            print("DEBUG raw dut.counts:", dict(dut.counts))
            print("DEBUG meas_q2c:", meas_q2c)
            print("DEBUG chosen_ref_order:", chosen_ref_order)
            print("DEBUG chosen_dut_order:", chosen_dut_order)
            print("DEBUG canon ref:", ref_canon)
            print("DEBUG canon dut:", dut_canon)

        rec = {
            "circuit": qasm_path.name,
            "backend": args.backend,
            "shots": args.shots,
            "qubits": n_qubits,
            "clbits": n_clbits,
            "depth": getattr(ref, "depth", None),
            "cudaq_time_sec": dut.time_sec,
            "tvd": dist,
            "pass": ok,
            "status": "PASS" if ok else "FAIL",
            "ref_key_order": chosen_ref_order,
            "dut_key_order": chosen_dut_order,
        }
        rows.append(rec)
        append_jsonl(args.jsonl_out, rec)
        print(rec)

    write_csv(args.csv_out, rows)
    print(f"\nWrote: {args.csv_out} and {args.jsonl_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

