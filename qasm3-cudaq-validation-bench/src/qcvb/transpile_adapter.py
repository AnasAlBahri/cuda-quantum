from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re


class TranspilerNotConfigured(RuntimeError):
    pass


# These are ONLY used in normal Python (parsing). Do NOT reference these names inside @cudaq.kernel.
OP_H = 1
OP_X = 2
OP_Y = 3
OP_Z = 4
OP_S = 5
OP_SDG = 6
OP_T = 7
OP_TDG = 8
OP_RX = 9
OP_RY = 10
OP_RZ = 11
OP_CX = 12
OP_CZ = 13
OP_RESET = 14

OP_MEASURE = 99   # ignored for sampling
OP_BARRIER = 100  # ignored for sampling


def _parse_meas_assignments_from_qasm(qasm_path: Path, qc) -> list[int]:
    """
    Robustly extract q->c mapping from OpenQASM 3 assignment form:
        c[i] = measure q[j];

    Returns:
        meas_q2c: list length qc.num_qubits, with meas_q2c[q] = cbit_index or -1
    """
    meas_q2c = [-1] * qc.num_qubits

    # Map register name -> register object for lookup
    qreg_by_name = {r.name: r for r in qc.qregs}
    creg_by_name = {r.name: r for r in qc.cregs}

    text = qasm_path.read_text()

    # Matches:  c[0] = measure a[0];
    pat = re.compile(r"^\s*([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*=\s*measure\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;\s*$")

    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        c_name, c_idx_s, q_name, q_idx_s = m.groups()
        c_idx = int(c_idx_s)
        q_idx = int(q_idx_s)

        if q_name not in qreg_by_name or c_name not in creg_by_name:
            continue

        qreg = qreg_by_name[q_name]
        creg = creg_by_name[c_name]

        if q_idx < 0 or q_idx >= len(qreg):
            continue
        if c_idx < 0 or c_idx >= len(creg):
            continue

        qb = qreg[q_idx]
        cb = creg[c_idx]
        q_global = qc.find_bit(qb).index
        c_global = qc.find_bit(cb).index
        meas_q2c[q_global] = c_global

    return meas_q2c


def _qiskit_qasm3_to_arrays(
    qasm_path: Path,
) -> tuple[int, int, list[int], list[int], list[int], list[int], list[float], int]:
    """
    Returns:
      n_qubits, n_clbits, op_ids, q0s, q1s, meas_q2c, params, depth_effective

    - meas_q2c: list length n_qubits, meas_q2c[q] = classical_bit_index or -1
    - depth_effective excludes barrier/measure.
    """
    from qiskit import qasm3

    qc = qasm3.load(str(qasm_path))

    op_ids: list[int] = []
    q0s: list[int] = []
    q1s: list[int] = []
    params: list[float] = []

    depth_eff = 0

    # Track final measurement mapping q -> c
    meas_q2c: list[int] = [-1] * qc.num_qubits
    saw_measure = False

    def add(op: int, q0: int, q1: int, p: float, *, counts_toward_depth: bool = True) -> None:
        nonlocal depth_eff
        op_ids.append(op)
        q0s.append(q0)
        q1s.append(q1)
        params.append(p)
        if counts_toward_depth:
            depth_eff += 1

    for inst, qargs, cargs in qc.data:
        name = inst.name

        # IMPORTANT: qargs/cargs indices must be GLOBAL, not register-local.
        qs = [qc.find_bit(qb).index for qb in qargs] if qargs else []
        cs = [qc.find_bit(cb).index for cb in cargs] if cargs else []

        if name == "h":
            add(OP_H, qs[0], -1, 0.0)
        elif name == "x":
            add(OP_X, qs[0], -1, 0.0)
        elif name == "y":
            add(OP_Y, qs[0], -1, 0.0)
        elif name == "z":
            add(OP_Z, qs[0], -1, 0.0)
        elif name == "s":
            add(OP_S, qs[0], -1, 0.0)
        elif name == "sdg":
            add(OP_SDG, qs[0], -1, 0.0)
        elif name == "t":
            add(OP_T, qs[0], -1, 0.0)
        elif name == "tdg":
            add(OP_TDG, qs[0], -1, 0.0)
        elif name == "rx":
            add(OP_RX, qs[0], -1, float(inst.params[0]))
        elif name == "ry":
            add(OP_RY, qs[0], -1, float(inst.params[0]))
        elif name == "rz":
            add(OP_RZ, qs[0], -1, float(inst.params[0]))
        elif name == "cx":
            add(OP_CX, qs[0], qs[1], 0.0)
        elif name == "cz":
            add(OP_CZ, qs[0], qs[1], 0.0)
        elif name == "reset":
            add(OP_RESET, qs[0], -1, 0.0)
        elif name == "measure":
            saw_measure = True
            # Try the easy path first (if Qiskit exposes cargs)
            if len(qs) == 1 and len(cs) == 1:
                meas_q2c[qs[0]] = cs[0]
            add(OP_MEASURE, qs[0] if qs else 0, -1, 0.0, counts_toward_depth=False)
        elif name == "barrier":
            add(OP_BARRIER, 0, -1, 0.0, counts_toward_depth=False)
        else:
            raise NotImplementedError(f"Fallback does not support gate '{name}' in {qasm_path.name}")

    # If we saw measure but didn't reliably get mapping via qc.data, parse from QASM text.
    if saw_measure and all(x == -1 for x in meas_q2c):
        meas_q2c = _parse_meas_assignments_from_qasm(qasm_path, qc)

    allowed = {
        OP_H, OP_X, OP_Y, OP_Z, OP_S, OP_SDG, OP_T, OP_TDG,
        OP_RX, OP_RY, OP_RZ, OP_CX, OP_CZ, OP_RESET,
        OP_MEASURE, OP_BARRIER
    }
    bad = sorted(set(op_ids) - allowed)
    if bad:
        raise RuntimeError(f"Unexpected opcodes {bad} produced from {qasm_path.name}")

    return qc.num_qubits, qc.num_clbits, op_ids, q0s, q1s, meas_q2c, params, depth_eff


def _fallback_qiskit_to_cudaq_kernel(qasm_path: Path) -> Any:
    import cudaq

    n, ncl, op_ids, q0s, q1s, meas_q2c, params, depth_eff = _qiskit_qasm3_to_arrays(qasm_path)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(n)

        for i in range(len(op_ids)):
            op = op_ids[i]
            q0 = q0s[i]
            q1 = q1s[i]
            p = params[i]

            if op == 1:
                h(q[q0])
            elif op == 2:
                x(q[q0])
            elif op == 3:
                y(q[q0])
            elif op == 4:
                z(q[q0])
            elif op == 5:
                s(q[q0])
            elif op == 6:
                sdg(q[q0])
            elif op == 7:
                t(q[q0])
            elif op == 8:
                tdg(q[q0])
            elif op == 9:
                rx(p, q[q0])
            elif op == 10:
                ry(p, q[q0])
            elif op == 11:
                rz(p, q[q0])
            elif op == 12:
                cx(q[q0], q[q1])
            elif op == 13:
                cz(q[q0], q[q1])
            elif op == 14:
                reset(q[q0])
            else:
                pass

    try:
        setattr(kernel, "__qcvb_num_qubits__", int(n))
        setattr(kernel, "__qcvb_num_clbits__", int(ncl))
        setattr(kernel, "__qcvb_depth__", int(depth_eff))
        setattr(kernel, "__qcvb_meas_q2c__", [int(x) for x in meas_q2c])
    except Exception:
        pass

    return kernel


def qasm3_to_cudaq_kernel(qasm_path: Path) -> Any:
    mode = os.environ.get("QCVB_TRANSPILER_MODE", "fallback_qiskit").strip()
    if mode != "fallback_qiskit":
        raise TranspilerNotConfigured(
            "Real transpiler not wired yet. For now run:\n"
            "  export QCVB_TRANSPILER_MODE=fallback_qiskit"
        )
    return _fallback_qiskit_to_cudaq_kernel(qasm_path)
