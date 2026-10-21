"""Fail-closed recall verdicts (v0.3.0, SYNTHETIC).

A recall is called VERIFIED_RECALL only when every gate passes. Any
failure, missing input or exception gives a non-verified verdict.

1. CUE GATE: the raw cue's max matched-filter overlap must reach tau,
   calibrated from never-stored cues on this hologram (calibrate.py).
   This needs no ground truth. A ghost cue fails here however confidently
   the dynamics settle. If a target index is given, the cue's winner must
   also be that index. Both the reported AND the independently
   recomputed cue overlap must reach tau. The calibration must be a
   cseq-cal/1 record bound to this exact hologram (calibrate.py);
   legacy {"tau_cue_overlap": x} dicts give WITHHELD.
2. RECALL GATE: the sliced fidelity must reach 0.99 (TARGET) *and* the
   bit accuracy must reach 0.99 (TARGET), each on BOTH the reported and
   the recomputed path. Agreement within tolerance never rescues a value
   below its threshold (reported 0.992 vs recomputed 0.988 fails). Fidelity alone is sign-blind:
   a fully inverted recall has fidelity 1.0 and bit accuracy 0.0.
   Reference pattern: the given target if known (ground-truth mode);
   otherwise the stored pattern that won the cue gate (self-consistency
   mode, which checks that the dynamics stayed on the pattern the cue
   pointed at, not that this pattern is the "right" answer).
3. VERIFIER GATE: the reported numbers are re-computed on a separate code
   path (verify.py); any mismatch gives WITHHELD. This catches reporting
   bugs and tampering, not modelling errors.

Verdicts:
    VERIFIED_RECALL          all gates pass
    CONFABULATION_SUSPECTED  cue gate failed (never stored, or too degraded)
    UNVERIFIED_RECALL        cue gate passed but the recall gate failed
    WITHHELD                 verifier mismatch or internal error

The OES-32 score of the post-recall overlap spectrum is reported but is
not a gate: it fired on most never-stored cues in v0.1.0's sweeps.
"""

import numpy as np

from .calibrate import validate_calibration
from .claims import BOUNDARY_STATEMENT, tag
from .validation import validate_index, validate_phase
from .oes import detect_events
from .verify import recompute_all, verify_claims

FIDELITY_TARGET = 0.99  # TARGET
BIT_ACCURACY_TARGET = 0.99  # TARGET


def _withheld(error):
    return {
        "verdict": "WITHHELD",
        "mode": None,
        "gates": {"cue_gate": False, "recall_gate": False, "verifier_gate": False},
        "metrics": {},
        "verification_checks": [],
        "error": error,
        "boundary": BOUNDARY_STATEMENT,
    }


def assess_recall(mem, cue, recalled_phase, calibration, target_phase=None, target_index=None):
    """Assess one recall and return a verdict dict (see module docstring).

    Pass target_phase (and optionally target_index) only when the ground
    truth is known, e.g. in a synthetic experiment. Without them the
    verdict uses the self-consistency reference described above.
    """
    try:
        if calibration is None:
            return _withheld("missing calibration")
        tau = validate_calibration(calibration, mem)
        if mem.n_patterns < 1:
            return _withheld("memory holds no patterns")
        n = mem.n_modes
        cue_phase = validate_phase(cue.phase, n, name="cue phase")
        recalled_phase = validate_phase(recalled_phase, n, name="recalled phase")
        if target_index is not None:
            target_index = validate_index(target_index, mem.n_patterns, name="target_index")
        cue_ov = mem.overlaps(cue_phase)
        winner = int(np.argmax(cue_ov))
        if target_phase is None:
            # self-consistency: reference is the pattern the cue pointed at;
            # this does NOT establish the recall is the "right" answer.
            mode = "self_consistency"
            reference = validate_phase(mem._patterns[winner], n, name="reference phase")
        else:
            mode = "ground_truth"
            reference = validate_phase(target_phase, n, name="target phase")

        rec_bits = (np.abs(np.angle(np.exp(1j * recalled_phase))) > np.pi / 2).astype(int)
        ref_bits = (np.abs(np.angle(np.exp(1j * reference))) > np.pi / 2).astype(int)
        reported = {
            "cue_overlap_max": float(np.max(cue_ov)),
            "cue_overlap_winner": winner,
            "fidelity_cseq": float(mem.fidelity(recalled_phase, reference)),
            "bit_accuracy": float(np.mean(rec_bits == ref_bits)),
        }
        recomputed = recompute_all(mem._patterns, cue_phase, recalled_phase, reference)
        verified, checks = verify_claims(reported, recomputed, n_patterns=mem.n_patterns)
        return _decide(reported, recomputed, verified, checks, tau, target_index, mode, mem, recalled_phase)
    except Exception as exc:  # fail-closed
        return _withheld(f"{type(exc).__name__}: {exc}")


def _decide(reported, recomputed, verified, checks, tau, target_index, mode, mem, recalled_phase):
    """Apply the gates on BOTH metric paths. Separated so it can be tested."""
    try:
        winner = reported["cue_overlap_winner"]
        cue_ok = reported["cue_overlap_max"] >= tau and recomputed["cue_overlap_max"] >= tau
        if target_index is not None:
            cue_ok = cue_ok and winner == target_index and recomputed["cue_overlap_winner"] == target_index
        recall_ok = all(
            d["fidelity_cseq"] >= FIDELITY_TARGET and d["bit_accuracy"] >= BIT_ACCURACY_TARGET
            for d in (reported, recomputed)
        )

        if not verified:
            verdict = "WITHHELD"
        elif not cue_ok:
            verdict = "CONFABULATION_SUSPECTED"
        elif not recall_ok:
            verdict = "UNVERIFIED_RECALL"
        else:
            verdict = "VERIFIED_RECALL"

        oes_score, oes_flag = detect_events(mem.overlaps(recalled_phase))
        return {
            "verdict": verdict,
            "mode": mode,
            "gates": {
                "cue_gate": bool(cue_ok),
                "recall_gate": bool(recall_ok),
                "verifier_gate": bool(verified),
            },
            "metrics": {
                "cue_overlap_max": tag(reported["cue_overlap_max"], "SYNTHETIC"),
                "cue_overlap_winner": tag(winner, "SYNTHETIC"),
                "tau_cue_overlap": tag(tau, "SYNTHETIC"),
                "fidelity_cseq": tag(reported["fidelity_cseq"], "SYNTHETIC"),
                "bit_accuracy": tag(reported["bit_accuracy"], "SYNTHETIC"),
                "fidelity_target": tag(FIDELITY_TARGET, "TARGET"),
                "bit_accuracy_target": tag(BIT_ACCURACY_TARGET, "TARGET"),
                "oes32_spectrum_score": tag(float(oes_score), "SYNTHETIC"),
                "oes32_spectrum_flag": tag(bool(oes_flag), "SYNTHETIC"),
                "oes32_threshold": tag(0.50, "TARGET"),
            },
            "verification_checks": checks,
            "boundary": BOUNDARY_STATEMENT,
        }
    except Exception as exc:  # fail-closed
        return _withheld(f"{type(exc).__name__}: {exc}")
