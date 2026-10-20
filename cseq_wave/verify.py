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

import numpy as np

from .encode import slice_bpsk

TOL = 5e-3  # TARGET: agreement tolerance between reported and recomputed


def recompute_all(stored_phases, cue_phase, recalled_phase, target_phase):
    """Recompute every reportable metric from raw phases, independently.

    stored_phases: list of stored pattern phase arrays (the hologram's
    contents, read back out -- the verifier trusts nothing else).
    Returns a dict of metrics computed from scratch.
    """
    cue_q = np.exp(1j * np.asarray(cue_phase, dtype=float))
    n = cue_q.shape[0]

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


def verify_claims(reported, recomputed, tol=TOL):
    """Compare a report's claims against the recomputation.

    Returns (all_ok, checks) where checks is a list of per-claim dicts.
    Only keys present in BOTH dicts are checked; integer claims
    (winner index) must match exactly, floats within tol.
    """
    checks = []
    for key, rep in reported.items():
        if key not in recomputed:
            continue
        rec = recomputed[key]
        if isinstance(rep, (int, np.integer)) and isinstance(rec, (int, np.integer)):
            ok = int(rep) == int(rec)
        else:
            ok = abs(float(rep) - float(rec)) <= tol
        checks.append(
            {"claim": key, "reported": rep, "recomputed": rec, "ok": bool(ok)}
        )
    return all(c["ok"] for c in checks), checks
