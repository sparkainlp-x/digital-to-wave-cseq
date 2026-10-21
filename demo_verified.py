"""Verified demo (v0.3.0, SYNTHETIC): recall verdicts with a calibrated cue gate.

1. Store 4 random 256-mode BPSK patterns in one hologram.
2. Calibrate the cue-overlap threshold tau from seeded never-stored cues.
3. Recall a stored pattern from a cue with 20% flips + 0.30 rad jitter,
   and assess it twice: without the ground truth (self-consistency mode)
   and with it (ground-truth mode).
4. Run a never-stored cue through the same verdict, without a target.
5. Screen the report text, then write an evidence passport.

Classical simulation; not quantum hardware; not a validation of the
Coherent Sequence theory.

Run:  python demo_verified.py [--seed 7]
"""

import argparse
import sys

import numpy as np

from cseq_wave import HolographicMemory, bits_to_wave, evidence_passport
from cseq_wave.calibrate import calibrate_cue_threshold
from cseq_wave.claims import assert_clean
from cseq_wave.evidence import save_passport
from cseq_wave.verdict import assess_recall

N_MODES, FLIP, JITTER, STEPS = 256, 0.2, 0.3, 40


def _v(verdict, key):
    return verdict["metrics"][key]["value"]


def run(seed=7):
    rng = np.random.default_rng(seed)
    labels = ["WAVEFORM", "COHERENT", "SEQUENCE", "PHASEMEM"]
    codebook = {label: rng.integers(0, 2, N_MODES) for label in labels}
    mem = HolographicMemory(N_MODES)
    for label in labels:
        mem.store(bits_to_wave(codebook[label]))
    mem.finalize()

    cal = calibrate_cue_threshold(mem, seed=seed + 1000, n_null=256, flip_prob=FLIP, jitter=JITTER)

    target = "SEQUENCE"
    t_idx = labels.index(target)
    cue = bits_to_wave(codebook[target]).decohere(rng, flip_prob=FLIP, jitter=JITTER)
    rec, _ = mem.recall(cue, steps=STEPS)
    main_blind = assess_recall(mem, cue, rec, cal)
    main_truth = assess_recall(mem, cue, rec, cal, target_phase=codebook[target] * np.pi, target_index=t_idx)

    ghost_bits = rng.integers(0, 2, N_MODES)
    ghost_cue = bits_to_wave(ghost_bits).decohere(rng, flip_prob=FLIP, jitter=JITTER)
    ghost_rec, _ = mem.recall(ghost_cue, steps=STEPS)
    ghost = assess_recall(mem, ghost_cue, ghost_rec, cal)
    ghost_max_recalled = float(np.max(mem.overlaps(ghost_rec)))

    params = {
        "n_modes": N_MODES, "n_patterns": len(labels), "flip_prob": FLIP,
        "jitter": JITTER, "recall_steps": STEPS,
        "calibration": {k: cal[k] for k in ("tau_cue_overlap", "null_n", "null_median", "null_q99", "null_max", "seed")},
    }
    metrics = {
        "main_verdict_self_consistency": main_blind["verdict"],
        "main_verdict_ground_truth": main_truth["verdict"],
        "main_cue_overlap": _v(main_truth, "cue_overlap_max"),
        "main_fidelity_cseq": _v(main_truth, "fidelity_cseq"),
        "main_bit_accuracy": _v(main_truth, "bit_accuracy"),
        "main_oes32_spectrum_score": _v(main_truth, "oes32_spectrum_score"),
        "control_verdict": ghost["verdict"],
        "control_cue_overlap": _v(ghost, "cue_overlap_max"),
        "control_recalled_max_overlap": ghost_max_recalled,
        "control_oes32_spectrum_score": _v(ghost, "oes32_spectrum_score"),
        "control_oes32_spectrum_flag": _v(ghost, "oes32_spectrum_flag"),
    }
    return evidence_passport(seed, params, metrics), cal


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="evidence_passport_verified.json")
    args = ap.parse_args(argv)
    pp, cal = run(args.seed)
    m = pp["metrics"]
    lines = [
        "=" * 64,
        "COHERENT SEQUENCE WAVE COMPUTER - verified demo v0.3.0 (SYNTHETIC)",
        "=" * 64,
        f"calibration: tau = {cal['tau_cue_overlap']:.4f} (null q0.99 of {cal['null_n']} never-stored cues, "
        f"median {cal['null_median']:.4f})",
        "--- stored pattern 'SEQUENCE', 20% flips + 0.30 rad jitter ---",
        f"verdict, no ground truth : {m['main_verdict_self_consistency']}",
        f"verdict, ground truth    : {m['main_verdict_ground_truth']}",
        f"cue overlap {m['main_cue_overlap']:.4f} vs tau {cal['tau_cue_overlap']:.4f}; "
        f"fidelity {m['main_fidelity_cseq']:.4f}; bit accuracy {m['main_bit_accuracy']:.4f}",
        "--- never-stored cue (no right answer exists) ---",
        f"verdict, no ground truth : {m['control_verdict']}",
        f"cue overlap {m['control_cue_overlap']:.4f} vs tau {cal['tau_cue_overlap']:.4f}; "
        f"recalled state's max overlap with a stored pattern {m['control_recalled_max_overlap']:.4f}",
        f"OES-32 on its recalled spectrum: {m['control_oes32_spectrum_score']:.4f} -> flag "
        f"{'SET' if m['control_oes32_spectrum_flag'] else 'not set'} (not used as a gate)",
        "boundary: " + "Classical simulation (synthetic signals only). Not quantum hardware; "
        "not a validation of the Coherent Sequence theory; no physical or cosmological claim.",
        "=" * 64,
    ]
    report = "\n".join(lines)
    assert_clean(report)
    print(report)
    save_passport(pp, args.out)
    print(f"evidence passport written: {args.out}  sha256={pp['sha256'][:16]}...")
    ok = (
        m["main_verdict_self_consistency"] == "VERIFIED_RECALL"
        and m["main_verdict_ground_truth"] == "VERIFIED_RECALL"
        and m["control_verdict"] == "CONFABULATION_SUSPECTED"
    )
    print("RESULT:", "PASS (stored cue verified, never-stored cue flagged)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
