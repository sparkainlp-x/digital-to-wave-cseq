"""Demonstration: the Coherent Sequence Wave Computer.

1. Encode text -> wave-states (BPSK phases).
2. Store N patterns in a holographic memory (interference).
3. Recall one pattern from a decohered cue (20% phase flips).
4. Read the recall fidelity AS C_seq of the relative phase.
5. Score the recall transient with OES-32; declare a lock-in event.
6. Emit an evidence passport (JSON) with a SHA-256 digest.

Run:
    python demo.py --seed 7
"""

import argparse
import json

import numpy as np

from cseq_wave import (
    HolographicMemory,
    bits_to_wave,
    detect_events,
    evidence_passport,
    text_to_bits,
    wave_to_bits,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-modes", type=int, default=256)
    ap.add_argument("--n-patterns", type=int, default=5)
    ap.add_argument("--flip-prob", type=float, default=0.20)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- 1. build a random codebook: each message -> 256-bit BPSK pattern
    codebook = rng.integers(0, 2, size=(args.n_patterns, args.n_modes))

    # ---- 2. store all patterns in the hologram
    mem = HolographicMemory(args.n_modes)
    for m in range(args.n_patterns):
        mem.store(bits_to_wave(codebook[m]))
    mem.finalize()

    # ---- 3. pick target, make a decohered cue (20% of phases flipped)
    target = args.seed % args.n_patterns  # deterministic choice
    target_wave = bits_to_wave(codebook[target])
    cue = target_wave.decohere(rng, flip_prob=args.flip_prob)

    # ---- 4. recall by interference dynamics
    recalled_phase, traj = mem.recall(cue, steps=25)
    overlaps = mem.overlaps(cue.phase)

    # ---- 5. readouts
    recalled_bits = wave_to_bits(
        type(target_wave)(np.ones(args.n_modes), recalled_phase)
    )
    bit_acc = float(np.mean(recalled_bits == codebook[target]))
    fidelity = mem.fidelity(recalled_phase, target_wave.phase)  # AS C_seq
    score, locked = detect_events(traj)

    print("=" * 64)
    print("COHERENT SEQUENCE WAVE COMPUTER — demo")
    print("=" * 64)
    print(f"seed={args.seed}  modes={args.n_modes}  patterns={args.n_patterns}")
    print(f"target pattern: #{target}   cue decoherence: {args.flip_prob:.0%} phase flips")
    print(f"bit accuracy:            {bit_acc:.4f}")
    print(f"fidelity (C_seq):        {fidelity:.4f}")
    print(f"overlap scores (winner should be ~1):")
    print("   " + "  ".join(f"{s:.2f}" for s in overlaps))
    print(f"OES-32 score:            {score:.4f}  -> lock-in {'DETECTED' if locked else 'not detected'}")
    print("=" * 64)

    # ---- 6. evidence passport
    passport = evidence_passport(
        seed=args.seed,
        parameters={
            "n_modes": args.n_modes,
            "n_patterns": args.n_patterns,
            "flip_prob": args.flip_prob,
            "recall_steps": 25,
            "oes_threshold": 0.50,
        },
        metrics={
            "bit_accuracy": bit_acc,
            "fidelity_cseq": fidelity,
            "oes32_score": score,
            "lock_in_detected": locked,
            "overlap_scores": [float(s) for s in overlaps],
        },
    )
    with open("evidence_passport.json", "w", encoding="utf-8") as f:
        json.dump(passport, f, indent=2, sort_keys=True)
    print(f"evidence passport written: evidence_passport.json  sha256={passport['sha256'][:16]}...")


if __name__ == "__main__":
    main()
