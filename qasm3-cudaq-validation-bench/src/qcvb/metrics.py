from __future__ import annotations
from typing import Mapping, Any, Optional
import numbers

def _parse_key_to_int_or_bits(k: Any) -> tuple[Optional[int], Optional[str]]:
    # Numeric keys (int, numpy int, etc.)
    if isinstance(k, numbers.Integral):
        return int(k), None

    s = str(k).strip().replace(" ", "")
    if not s:
        return None, None

    # Remove common prefixes
    if s.startswith("0b"):
        bits = s[2:]
        if set(bits) <= {"0", "1"}:
            return None, bits

    if s.startswith("0x"):
        try:
            return int(s, 16), None
        except Exception:
            return None, None

    # Pure bitstring?
    if set(s) <= {"0", "1"}:
        return None, s

    # Decimal integer string?
    if s.isdigit():
        try:
            return int(s, 10), None
        except Exception:
            return None, None

    return None, None

def normalize_counts(counts: Mapping[Any, Any], *, width_hint: Optional[int] = None) -> dict[str, int]:
    items = list(counts.items())

    parsed: list[tuple[Optional[int], Optional[str], Any]] = []
    max_bits_len = 0
    max_int = 0

    for k, v in items:
        ik, bs = _parse_key_to_int_or_bits(k)
        parsed.append((ik, bs, v))
        if bs is not None:
            max_bits_len = max(max_bits_len, len(bs))
        if ik is not None:
            max_int = max(max_int, ik)

    inferred_width = max_bits_len
    if max_int > 0:
        inferred_width = max(inferred_width, max_int.bit_length())
    inferred_width = max(1, inferred_width)

    width = width_hint if (width_hint is not None and width_hint > 0) else inferred_width

    out: dict[str, int] = {}
    for ik, bs, v in parsed:
        if bs is not None:
            ks = bs.zfill(width)
        elif ik is not None:
            ks = format(int(ik), f"0{width}b")
        else:
            raise ValueError(f"Unrecognized count key: {k!r}")

        try:
            iv = int(v)
        except Exception as e:
            raise TypeError(f"Count value for key {ks!r} is not int-convertible: {v!r} ({type(v)})") from e

        out[ks] = out.get(ks, 0) + iv

    return out

def tvd(counts_a: Mapping[Any, Any], counts_b: Mapping[Any, Any], *, width_hint: Optional[int] = None) -> float:
    import os
    if os.environ.get('QCVB_DEBUG_COUNTS') == '1':
        print('DEBUG tvd counts_a:', dict(counts_a))
        print('DEBUG tvd counts_b:', dict(counts_b))
        print('DEBUG tvd width_hint:', width_hint)
    a = normalize_counts(counts_a, width_hint=width_hint)
    b = normalize_counts(counts_b, width_hint=width_hint)

    na = sum(a.values())
    nb = sum(b.values())
    if na <= 0 or nb <= 0:
        raise ValueError(f"Non-positive shot totals: na={na}, nb={nb}")

    keys = set(a) | set(b)
    s = 0.0
    for k in keys:
        pa = a.get(k, 0) / na
        pb = b.get(k, 0) / nb
        s += abs(pa - pb)
    return 0.5 * s

def pass_fail_tvd(dist: float, threshold: float = 2e-2) -> bool:
    return dist < threshold
