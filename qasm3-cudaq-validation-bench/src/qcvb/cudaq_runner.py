from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import time

from qcvb.metrics import normalize_counts


@dataclass(frozen=True)
class CudaqSampleResult:
    counts: dict[str, int]        # bitstring -> count
    num_qubits: int
    time_sec: float


def _extract_counts_dict(sample_res: Any) -> dict[Any, int]:
    """
    Extract a plain dict of {outcome -> count} from a CUDA-Q SampleResult.

    Your CUDA-Q build returns a SampleResult with __iter__/__getitem__/__len__
    but no .counts() / .to_dict(), so we reconstruct counts by iterating keys.
    """
    # 1) Try common APIs first (other CUDA-Q builds)
    if hasattr(sample_res, "counts") and callable(getattr(sample_res, "counts")):
        m = sample_res.counts()
        if isinstance(m, Mapping):
            return dict(m)

    if hasattr(sample_res, "counts") and isinstance(getattr(sample_res, "counts"), Mapping):
        return dict(getattr(sample_res, "counts"))

    for name in ("get_counts", "to_dict", "as_dict"):
        if hasattr(sample_res, name) and callable(getattr(sample_res, name)):
            m = getattr(sample_res, name)()
            if isinstance(m, Mapping):
                return dict(m)

    # 2) Your build: treat SampleResult as an indexable container.
    #    - If iteration yields (key, value) pairs, dict(...) will work.
    #    - Otherwise iteration yields keys; use __getitem__ to fetch counts.
    if hasattr(sample_res, "__iter__") and hasattr(sample_res, "__getitem__"):
        try:
            it = list(sample_res)
        except Exception as e:
            raise TypeError(
                "Could not iterate over CUDA-Q SampleResult."
                f" type={type(sample_res)} err={e}"
            ) from e

        # Case A: iter gives pairs
        if it and isinstance(it[0], tuple) and len(it[0]) == 2:
            try:
                d = dict(it)
                return {k: int(v) for k, v in d.items()}
            except Exception:
                pass

        # Case B: iter gives keys
        d: dict[Any, int] = {}
        for k in it:
            try:
                d[k] = int(sample_res[k])
            except Exception as e:
                raise TypeError(
                    f"SampleResult iteration returned key={k!r}, but indexing failed: {e}"
                ) from e
        return d

    # 3) Last resort
    raise TypeError(
        "Could not extract counts from CUDA-Q sample result. "
        f"type={type(sample_res)} attrs={dir(sample_res)[:40]}"
    )


def _infer_num_qubits_from_counts(raw_counts: Mapping[Any, Any]) -> int:
    max_len = 0
    max_int = 0

    for k in raw_counts.keys():
        if isinstance(k, int):
            if k > max_int:
                max_int = k
            continue

        s = str(k).strip().replace(" ", "")
        if s.startswith("0b"):
            s = s[2:]
        if s and set(s) <= {"0", "1"}:
            if len(s) > max_len:
                max_len = len(s)

    if max_len > 0:
        return max_len

    return max(1, max_int.bit_length())


def run_cudaq_sampling(kernel: Any, *, backend: str, shots: int) -> CudaqSampleResult:
    import cudaq

    if shots <= 0:
        raise ValueError(f"shots must be > 0, got {shots}")

    cudaq.set_target(backend)

    t0 = time.perf_counter()

    # CUDA-Q API compat
    try:
        sample_res = cudaq.sample(kernel, shots_count=shots)
    except TypeError:
        sample_res = cudaq.sample(kernel, shots=shots)

    t1 = time.perf_counter()

    raw_counts = _extract_counts_dict(sample_res)
    n = _infer_num_qubits_from_counts(raw_counts)

    counts = normalize_counts(raw_counts, width_hint=n)

    total = sum(counts.values())
    if total != shots:
        raise RuntimeError(
            f"CUDA-Q counts sum to {total}, expected {shots}. "
            f"Raw={raw_counts} Norm={counts}"
        )

    return CudaqSampleResult(counts=counts, num_qubits=n, time_sec=(t1 - t0))

