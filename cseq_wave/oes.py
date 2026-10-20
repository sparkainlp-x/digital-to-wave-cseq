"""OES-32 style block scoring for the readout transient.

Block score (same form as the address-phase-demo OES32 block score):

    S = 0.45 * max|x| + 0.35 * RMS(x) + 0.20 * mean|x|

An event (here: "recall lock-in detected") is declared when S >= 0.50.
Applied to the coherence trajectory of a recall run, this turns the
wave computer's analog readout into a discrete detection event --
the same detection layer used across the OES telemetry work.
"""

import numpy as np

THRESHOLD = 0.50


def oes32_score(x):
    """OES-32 block score for a 1-D signal block x."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2)))
    mean_abs = float(np.mean(np.abs(x)))
    return 0.45 * peak + 0.35 * rms + 0.20 * mean_abs


def detect_events(x, threshold=THRESHOLD):
    """Return (score, detected) for a signal block."""
    score = oes32_score(x)
    return score, bool(score >= threshold)
