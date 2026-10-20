"""Self-calibrating detection thresholds (pattern from oes512q-latch).

The OES-32 flag cannot tell true recall from confabulation, and neither
can the post-recall overlap spectrum: once the dynamics settle on an
attractor, a never-stored ("ghost") cue looks as sharp as a stored one.
Measured here (256 modes, 4 patterns, 20% flips + 0.30 jitter, 240 trials
over 4 seeds): post-recall winner margins of stored cues go down to 0.395,
while ghost margins reach 0.973.

The separating signal is on the CUE side, before the dynamics run: the
matched-filter overlap of the raw cue against the stored patterns.

    stored cue: max overlap mean 0.324, min 0.170
    ghost cue : max overlap mean 0.0087, q0.99 0.029, max 0.034

This module sets the cue-overlap threshold tau as the q0.99 of the ghost
distribution, estimated from seeded null trials against the hologram in
use (the same idea as the self-calibrated threshold in oes512q-latch).
By construction about 1% of ghost cues pass (1.3% measured). Calibration
values are SYNTHETIC: they describe this simulator's null behaviour only.
"""

import numpy as np

from .encode import bits_to_wave


def calibrate_cue_threshold(
    mem,
    seed=7,
    n_null=256,
    flip_prob=0.2,
    jitter=0.3,
    quantile=0.99,
    floor=0.01,
):
    """Calibrate the cue-overlap threshold from null (ghost) trials.

    Draws n_null random never-stored patterns, decoheres them exactly
    like a real cue, and measures each one's max matched-filter overlap
    against the hologram. tau = quantile of that null distribution,
    floored at `floor`. Deterministic for a given (mem state, seed).
    """
    rng = np.random.default_rng(seed)
    null_overlaps = []
    for _ in range(n_null):
        ghost = rng.integers(0, 2, mem.n_modes)
        cue = bits_to_wave(ghost).decohere(rng, flip_prob=flip_prob, jitter=jitter)
        null_overlaps.append(float(np.max(mem.overlaps(cue.phase))))
    null_overlaps = np.asarray(null_overlaps)
    tau = float(max(np.quantile(null_overlaps, quantile), floor))
    return {
        "tau_cue_overlap": tau,
        "null_n": int(n_null),
        "null_quantile": float(quantile),
        "null_median": float(np.median(null_overlaps)),
        "null_q99": float(np.quantile(null_overlaps, 0.99)),
        "null_max": float(null_overlaps.max()),
        "flip_prob": float(flip_prob),
        "jitter": float(jitter),
        "seed": int(seed),
        "evidence_tag": "SYNTHETIC",
        "method": (
            "tau = q%.2f of max matched-filter cue overlap over %d seeded "
            "null (never-stored) trials against this hologram"
            % (quantile, n_null)
        ),
    }
