"""Separate-path re-computation of reported metrics.

Same idea as the verifier in sparkholo-coherence-agent: every number the
verdict engine reports is re-computed by this SEPARATE code path --
direct complex inner products written independently of HolographicMemory
-- and compared within tolerance. A mismatch means the report is
WITHHELD, never silently corrected.

This catches reporting bugs and tampering. It does not catch a wrong
model: verifier and memory share the same mathematics by design, the
same way sparkholo's verifier shares the metric definitions.
"""

import numbers

import numpy as np

from .encode import slice_bpsk
from .validation import is_real_number, validate_phase

TOL = 5e-3  # TARGET: agreement tolerance between reported and recomputed


def recompute_all(stored_phases, cue_phase, recalled_phase, target_phase):
    """Recompute every reportable metric from raw phases, independently.

    stored_phases: list of stored pattern phase arrays (the hologram's
    contents, read back out -- the verifier trusts nothing else).
    Returns a dict of metrics computed from scratch.
    """
    cue_phase = validate_phase(cue_phase, name="cue phase")
    n = cue_phase.shape[0]
    recalled_phase = validate_phase(recalled_phase, n, name="recalled phase")
    target_phase = validate_phase(target_phase, n, name="reference phase")
    stored_phases = [validate_phase(p, n, name="stored pattern phase") for p in stored_phases]
    if not stored_phases:
        raise ValueError("no stored patterns to verify against")
    cue_q = np.exp(1j * cue_phase)

    # cue-side matched filter (independent of HolographicMemory.overlaps)
    cue_overlaps = []
    for p in stored_phases:
        x = np.exp(1j * np.asarray(p, dtype=float))
        cue_overlaps.append(float(abs(np.vdot(x, cue_q)) ** 2) / n**2)
    cue_overlaps = np.asarray(cue_overlaps)
    s = np.sort(cue_overlaps)

    # fidelity: C_seq of the relative phase, sliced first (independent
    # of wave.alignment / HolographicMemory.fidelity)
    rec_sliced = slice_bpsk(np.asarray(recalled_phase, dtype=float))
    rel = rec_sliced - np.asarray(target_phase, dtype=float)
    fidelity = float(abs(np.mean(np.exp(1j * rel))))

    # bit accuracy from the sliced recalled phases
    rec_bits = (rec_sliced == np.pi).astype(int)
    tgt_wrapped = np.angle(np.exp(1j * np.asarray(target_phase, dtype=float)))
    tgt_bits = (np.abs(tgt_wrapped) > np.pi / 2).astype(int)

    return {
        "cue_overlap_max": float(cue_overlaps.max()),
        "cue_overlap_winner": int(np.argmax(cue_overlaps)),
        "cue_margin": float(s[-1] - s[-2]) if len(s) > 1 else float(s[-1]),
        "fidelity_cseq": fidelity,
        "bit_accuracy": float(np.mean(rec_bits == tgt_bits)),
    }


REQUIRED_METRICS = ("cue_overlap_max", "cue_overlap_winner", "fidelity_cseq", "bit_accuracy")
OPTIONAL_METRICS = ("cue_margin",)
UNIT_INTERVAL_METRICS = ("cue_overlap_max", "fidelity_cseq", "bit_accuracy", "cue_margin")


def _metric_problem(key, value, n_patterns):
    """Return a reason string if `value` is not a valid `key` metric, else None."""
    if key == "cue_overlap_winner":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (numbers.Integral, np.integer)):
            return f"winner must be an integer, got {type(value).__name__}"
        if int(value) < 0:
            return "winner must be nonnegative"
        if n_patterns is not None and int(value) >= n_patterns:
            return f"winner {int(value)} out of range [0, {n_patterns})"
        return None
    if not is_real_number(value):
        return f"must be a finite real number, got {value!r}"
    if key in UNIT_INTERVAL_METRICS and not 0.0 <= float(value) <= 1.0:
        return f"out of range [0, 1]: {float(value)}"
    return None


def verify_claims(reported, recomputed, tol=TOL, n_patterns=None):
    """Compare a report's claims against the recomputation, fail-closed.

    Returns (all_ok, checks); each check is a dict with claim, reported,
    recomputed, ok and (on failure) reason. Rules:
    - both arguments must be nonempty dicts;
    - REQUIRED_METRICS must be present in BOTH dicts (missing => failure);
    - reported keys outside REQUIRED_METRICS + OPTIONAL_METRICS are rejected;
    - every reported metric is checked; extra recomputed metrics are fine;
    - bool/str/NaN/inf values and out-of-range values fail;
    - winner is an exact nonnegative integer (< n_patterns when given);
    - tol must be a finite, nonnegative real number.
    """
    checks = []

    def fail(claim, reason, rep=None, rec=None):
        checks.append({"claim": claim, "reported": rep, "recomputed": rec, "ok": False, "reason": reason})

    if not is_real_number(tol) or float(tol) < 0:
        fail("tolerance", f"tolerance must be finite and nonnegative, got {tol!r}")
        return False, checks
    if n_patterns is not None:
        if isinstance(n_patterns, bool) or not isinstance(n_patterns, (numbers.Integral, np.integer)) or n_patterns < 1:
            fail("n_patterns", f"invalid memory size {n_patterns!r}")
            return False, checks
    for label, d in (("reported", reported), ("recomputed", recomputed)):
        if not isinstance(d, dict) or not d:
            fail(label, f"{label} metrics must be a nonempty dict")
    if checks:
        return False, checks

    allowed = set(REQUIRED_METRICS) | set(OPTIONAL_METRICS)
    for key in REQUIRED_METRICS:
        if key not in reported:
            fail(key, "required metric missing from reported", rec=recomputed.get(key))
        if key not in recomputed:
            fail(key, "required metric missing from recomputed", rep=reported.get(key))
    for key, rep in reported.items():
        if key not in allowed:
            fail(key, "unknown reported metric", rep=rep)
            continue
        if key not in recomputed:
            if key not in REQUIRED_METRICS:
                fail(key, "reported metric has no recomputed counterpart", rep=rep)
            continue
        rec = recomputed[key]
        reason = _metric_problem(key, rep, n_patterns)
        if reason:
            fail(key, "reported " + reason, rep, rec)
            continue
        reason = _metric_problem(key, rec, n_patterns)
        if reason:
            fail(key, "recomputed " + reason, rep, rec)
            continue
        if key == "cue_overlap_winner":
            ok = int(rep) == int(rec)
            why = None if ok else "winner index differs"
        else:
            ok = abs(float(rep) - float(rec)) <= float(tol)
            why = None if ok else f"differs by more than tol={float(tol)}"
        c = {"claim": key, "reported": rep, "recomputed": rec, "ok": bool(ok)}
        if why:
            c["reason"] = why
        checks.append(c)
    return all(c["ok"] for c in checks), checks
