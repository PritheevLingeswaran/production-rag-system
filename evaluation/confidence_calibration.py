from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    count: int
    avg_conf: float
    accuracy: float


def expected_calibration_error(confs: List[float], correct: List[int], n_bins: int = 10) -> Tuple[float, List[CalibrationBin]]:
    assert len(confs) == len(correct)
    buckets: List[List[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confs):
        b = min(n_bins - 1, int(c * n_bins))
        buckets[b].append(i)

    total = len(confs) or 1
    ece = 0.0
    out: List[CalibrationBin] = []
    for b, idxs in enumerate(buckets):
        if not idxs:
            continue
        lo, hi = b / n_bins, (b + 1) / n_bins
        avg = sum(confs[i] for i in idxs) / len(idxs)
        acc = sum(correct[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / total) * abs(avg - acc)
        out.append(CalibrationBin(lo=lo, hi=hi, count=len(idxs), avg_conf=avg, accuracy=acc))
    return ece, out
