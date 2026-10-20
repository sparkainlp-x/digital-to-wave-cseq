"""Demonstration: the Coherent Sequence Wave Computer (SYNTHETIC).

1. Encode random bit patterns as wave-states (BPSK phases).
2. Store them in one holographic memory (complex Hopfield coupling).
3. Recall one pattern from a decohered cue (20% phase flips).
4. Read recall fidelity with the C_seq formula on the relative phase.
5. Score the recall trajectory with OES-32 (a convergence flag).
6. Controls: a never-stored cue and a uniform-noise cue, to show what the
   OES-32 flag and the overlap spectrum do when no right answer exists.
7. Emit an evidence passport (JSON) with a SHA-256 digest.

Classical simulation; not quantum hardware; not a validation of the
Coherent Sequence theory.

Run:
    python demo.py --seed 7
"""

import argparse
import sys

import numpy as np

from cseq_wave import (
    HolographicMemory,
    WaveState,
    bits_to_wave,
    detect_events,
    evidence_passport,
    wave_to_bits,
)
from cseq_wave.evidence import save_passport

RECALL_STEPS = 25


def _margin(spectrum):
    top = np.sort(spectrum)
    return float(top[-1] - top[-2]) if top.size > 1 else float(top[-1])


def run(seed=7, n_modes=256, n_patterns=5, flip_prob=0.20):
    rng = np.random.default_rng(seed)

    # random codebook: each message -> n_modes-bit BPSK pattern
    codebook = rng.integers(0, 2, size=(n_patterns, n_modes))
    mem = HolographicMemory(n_modes)
    for m in range(n_patterns):
        mem.store(bits_to_wave(codebook[m]))
    mem.finalize()

    target = seed % n_patterns  # deterministic choice
    target_wave = bits_to_wave(codebook[target])
    cue = target_wave.decohere(rng, flip_prob=flip_prob)

    recalled_phase, traj = mem.recall(cue, steps=RECALL_STEPS)
    cue_overlaps = mem.overlaps(cue.phase)
    recalled_overlaps = mem.overlaps(recalled_phase)

    recalled_bits = wave_to_bits(WaveState(np.ones(n_modes), recalled_phase))
    bit_acc = float(np.mean(recalled_bits == codebook[target]))
    fidelity = mem.fidelity(recalled_phase, target_wave.phase)
    fidelity_raw = mem.fidelity(recalled_phase, target_wave.phase, slice_output=False)
    score, locked = detect_events(traj)

    # control 1: a pattern that was never stored (no right answer exists)
    ghost_bits = rng.integers(0, 2, n_modes)
    ghost_rec, ghost_traj = mem.recall(bits_to_wave(ghost_bits), steps=RECALL_STEPS)
    ghost_ov = mem.overlaps(ghost_rec)
    ghost_score, ghost_locked = detect_events(ghost_traj)
    # control 2: uniform random phases
    noise = WaveState(np.ones(n_modes), rng.uniform(-np.pi, np.pi, n_modes))
    _, noise_traj = mem.recall(noise, steps=RECALL_STEPS)
    noise_score, noise_locked = detect_events(noise_traj)

    params = {
        "n_modes": n_modes,
        "n_patterns": n_patterns,
        "flip_prob": flip_prob,
        "recall_steps": RECALL_STEPS,
        "oes_threshold": 0.50,
        "codebook": "seeded uniform random BPSK",
    }
    metrics = {
        "target_index": target,
        "bit_accuracy": bit_acc,
        "fidelity_cseq": fidelity,
        "fidelity_cseq_unsliced": fidelity_raw,
        "oes32_score": score,
        "oes32_flag": locked,
        "cue_overlap_scores": cue_overlaps,
        "recalled_overlap_scores": recalled_overlaps,
        "recalled_winner_margin": _margin(recalled_overlaps),
        "control_unstored_max_overlap": float(np.max(ghost_ov)),
        "control_unstored_winner_margin": _margin(ghost_ov),
        "control_unstored_fidelity_vs_own_bits": mem.fidelity(ghost_rec, ghost_bits * np.pi),
        "control_unstored_oes32_score": ghost_score,
        "control_unstored_oes32_flag": ghost_locked,
        "control_noise_oes32_score": noise_score,
        "control_noise_oes32_flag": noise_locked,
    }
    return evidence_passport(seed, params, metrics)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-modes", type=int, default=256)
    ap.add_argument("--n-patterns", type=int, default=5)
    ap.add_argument("--flip-prob", type=float, default=0.20)
    ap.add_argument("--out", default="evidence_passport.json")
    args = ap.parse_args(argv)

    pp = run(args.seed, args.n_modes, args.n_patterns, args.flip_prob)
    m = pp["metrics"]
    fmt = lambda xs: "  ".join(f"{s:.2f}" for s in xs)  # noqa: E731

    print("=" * 64)
    print("COHERENT SEQUENCE WAVE COMPUTER - demo (SYNTHETIC)")
    print("=" * 64)
    print(f"seed={args.seed}  modes={args.n_modes}  patterns={args.n_patterns}")
    print(f"target pattern: #{m['target_index']}   cue decoherence: {args.flip_prob:.0%} phase flips")
    print(f"bit accuracy:            {m['bit_accuracy']:.4f}")
    print(f"fidelity (C_seq):        {m['fidelity_cseq']:.4f}   (unsliced {m['fidelity_cseq_unsliced']:.4f})")
    print("overlap scores, decohered cue (source ~ (1-2p)^2):")
    print("   " + fmt(m["cue_overlap_scores"]))
    print("overlap scores, recalled state (winner should be ~1):")
    print("   " + fmt(m["recalled_overlap_scores"]))
    print(f"OES-32 score:            {m['oes32_score']:.4f}  -> convergence flag {'SET' if m['oes32_flag'] else 'not set'}")
    print("--- controls (no right answer exists) ---")
    print(f"never-stored cue: max overlap {m['control_unstored_max_overlap']:.2f}, "
          f"OES-32 {m['control_unstored_oes32_score']:.4f} -> flag {'SET' if m['control_unstored_oes32_flag'] else 'not set'}")
    print(f"uniform-noise cue: OES-32 {m['control_noise_oes32_score']:.4f} -> flag {'SET' if m['control_noise_oes32_flag'] else 'not set'}")
    print("note: the OES-32 flag marks convergence, not a correct recall.")
    print("=" * 64)
    save_passport(pp, args.out)
    print(f"evidence passport written: {args.out}  sha256={pp['sha256'][:16]}...")

    ok = m["bit_accuracy"] == 1.0 and m["fidelity_cseq"] == 1.0 and m["oes32_flag"]
    print("RESULT:", "PASS (exact recall at this seed)" if ok else "FAIL (recall not exact at this seed)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
