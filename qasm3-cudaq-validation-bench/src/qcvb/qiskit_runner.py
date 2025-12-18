from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from qcvb.metrics import normalize_counts

@dataclass(frozen=True)
class QiskitSample:
    counts: dict[str, int]
    num_qubits: int
    depth: int

def run_qiskit_sampling(qasm_path: Path, shots: int = 20000, seed: int = 1234) -> QiskitSample:
    """
    Load OpenQASM 3 with qiskit.qasm3.load, then compute probabilities via Statevector
    (no qiskit-aer needed), then sample measurement outcomes.
    """
    from qiskit import qasm3
    from qiskit.quantum_info import Statevector

    qc = qasm3.load(str(qasm_path))
    n = qc.num_qubits
    d = qc.depth()

    # Remove final measurements so we can build a statevector.
    qc_nom = qc.remove_final_measurements(inplace=False)

    sv = Statevector.from_instruction(qc_nom)
    probs = np.asarray(sv.probabilities(), dtype=float)  # length 2**n

    rng = np.random.default_rng(seed)
    samples = rng.choice(len(probs), size=shots, p=probs)

    counts: dict[str, int] = {}
    for s in samples:
        b = format(int(s), f"0{n}b")  # bitstring length n
        counts[b] = counts.get(b, 0) + 1

    counts = normalize_counts(counts)
    return QiskitSample(counts=counts, num_qubits=n, depth=d)
