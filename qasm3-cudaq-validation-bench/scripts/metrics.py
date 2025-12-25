from __future__ import annotations
from typing import Any, Mapping

def _to_bitstring(k: Any) -> str:
    s = str(k).replace(" ", "")
    if s.startswith("0b"):
        s = s[2:]
    return s

def _normalize_counts(counts: Mapping[Any, int], nbits: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in counts.items():
        s = _to_bitstring(k)
        s = s.zfill(nbits)
        out[s] = out.get(s, 0) + int(v)
    return out

def marginal_p1(counts: Mapping[Any, int], nbits: int) -> list[float]:
    C = _normalize_counts(counts, nbits)
    shots = sum(C.values()) or 1

    p1 = [0.0] * nbits
    # Qiskit convention: key is c[n-1]...c[0]
    for s, c in C.items():
        s = s.zfill(nbits)
        for i in range(nbits):
            pos = nbits - 1 - i
            if s[pos] == "1":
                p1[i] += c
    return [x / shots for x in p1]

def avg_abs_marginal_diff(a: Mapping[Any, int], b: Mapping[Any, int], nbits: int) -> float:
    pa = marginal_p1(a, nbits)
    pb = marginal_p1(b, nbits)
    return sum(abs(x - y) for x, y in zip(pa, pb)) / max(1, nbits)

def hamming_weight_tvd(a: Mapping[Any, int], b: Mapping[Any, int], nbits: int) -> float:
    A = _normalize_counts(a, nbits)
    B = _normalize_counts(b, nbits)
    sa = sum(A.values()) or 1
    sb = sum(B.values()) or 1

    wa = [0.0] * (nbits + 1)
    wb = [0.0] * (nbits + 1)

    for s, c in A.items():
        wa[s.count("1")] += c / sa
    for s, c in B.items():
        wb[s.count("1")] += c / sb

    return 0.5 * sum(abs(x - y) for x, y in zip(wa, wb))
