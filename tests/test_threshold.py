import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from threshold_analysis import at, best_f1_threshold  # noqa: E402


def test_at_counts_and_per_10k():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    r = at(y, p, 0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["per_10k"]["false_alerts"] == 2000.0


def test_best_f1_threshold_perfect():
    t, f1 = best_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert f1 == 1.0 and t == 0.8
