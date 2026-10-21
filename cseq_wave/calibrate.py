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

Calibration schema (v0.3.0, ``schema_version == "cseq-cal/1"``)
--------------------------------------------------------------
A calibration is bound to the exact hologram it was measured on:

    schema_version    "cseq-cal/1"
    memory_digest     HolographicMemory.digest() (patterns, shape, order)
    mode_count        mem.n_modes
    pattern_count     mem.n_patterns
    null_protocol     NULL_PROTOCOL (how ghost cues are generated)
    noise             {"flip_prob": float, "jitter": float}
    null_n            number of null samples (sample count)
    null_quantile     quantile used for tau, in (0, 1)
    seed              integer RNG seed
    tau_cue_overlap   the threshold, finite, in [0, 1]

``validate_calibration(cal, mem)`` rejects any calibration that is not a
dict, lacks a field, has the wrong schema version, or whose digest /
mode count / pattern count do not match the hologram (stale: patterns
were added or changed after calibration; mismatched: another hologram).

Compatibility: legacy dicts such as ``{"tau_cue_overlap": 0.03}`` are
NOT accepted as verified calibration; assess_recall returns WITHHELD.
Re-run calibrate_cue_threshold on the finalized memory instead.
"""

import numpy as np

from .encode import bits_to_wave
from .validation import is_real_number

SCHEMA_VERSION = "cseq-cal/1"
NULL_PROTOCOL = (
    "ghost bits ~ rng.integers(0, 2, n_modes); cue = bits_to_wave(ghost)"
    ".decohere(rng, flip_prob, jitter); score = max(mem.overlaps(cue.phase))"
)
REQUIRED_FIELDS = (
    "schema_version", "memory_digest", "mode_count", "pattern_count",
    "null_protocol", "noise", "null_n", "null_quantile", "seed", "tau_cue_overlap",
)


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
        "schema_version": SCHEMA_VERSION,
        "memory_digest": mem.digest(),
        "mode_count": int(mem.n_modes),
        "pattern_count": int(mem.n_patterns),
        "null_protocol": NULL_PROTOCOL,
        "noise": {"flip_prob": float(flip_prob), "jitter": float(jitter)},
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


def _int_field(cal, key):
    v = cal[key]
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ValueError(f"calibration.{key} must be an integer")
    return int(v)


def validate_calibration(cal, mem):
    """Raise ValueError unless `cal` is a current cseq-cal/1 calibration of `mem`."""
    if not isinstance(cal, dict):
        raise ValueError("calibration must be a dict")
    missing = [k for k in REQUIRED_FIELDS if k not in cal]
    if missing:
        raise ValueError("legacy or incomplete calibration; missing " + ", ".join(missing))
    if cal["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported calibration schema {cal['schema_version']!r}")
    if cal["null_protocol"] != NULL_PROTOCOL:
        raise ValueError("calibration null-generation protocol differs")
    if _int_field(cal, "mode_count") != mem.n_modes:
        raise ValueError("calibration mode_count does not match memory")
    if _int_field(cal, "pattern_count") != mem.n_patterns:
        raise ValueError("calibration pattern_count does not match memory (stale?)")
    if cal["memory_digest"] != mem.digest():
        raise ValueError("calibration memory_digest does not match memory (stale or mismatched)")
    if _int_field(cal, "null_n") < 1:
        raise ValueError("calibration null_n must be >= 1")
    _int_field(cal, "seed")
    q = cal["null_quantile"]
    if not is_real_number(q) or not 0.0 < float(q) < 1.0:
        raise ValueError("calibration null_quantile must be in (0, 1)")
    noise = cal["noise"]
    if not isinstance(noise, dict) or set(noise) != {"flip_prob", "jitter"}:
        raise ValueError("calibration noise must be {flip_prob, jitter}")
    if not all(is_real_number(v) and float(v) >= 0 for v in noise.values()):
        raise ValueError("calibration noise values must be finite and nonnegative")
    tau = cal["tau_cue_overlap"]
    if not is_real_number(tau) or not 0.0 <= float(tau) <= 1.0:
        raise ValueError("calibration tau_cue_overlap must be finite in [0, 1]")
    return float(tau)
