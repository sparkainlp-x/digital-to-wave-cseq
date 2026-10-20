"""SYNTHETIC capacity and control sweeps for cseq_wave.

Writes results/sweeps.json and results/sweeps.md. Deterministic (seeded).

Run:  python experiments/capacity_sweep.py [--trials 100]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cseq_wave import HolographicMemory, WaveState, bits_to_wave, oes32_score, wave_to_bits  # noqa: E402
from cseq_wave.evidence import canonical_digest  # noqa: E402

N = 256
FLIP = 0.20
PATTERN_COUNTS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]


def _memory(rng, p):
    cb = rng.integers(0, 2, (p, N))
    m = HolographicMemory(N)
    for b in cb:
        m.store(bits_to_wave(b))
    m.finalize()
    return cb, m


def capacity(trials):
    rows = []
    for p in PATTERN_COUNTS:
        acc, exact, fs, fr = [], [], [], []
        for s in range(trials):
            rng = np.random.default_rng(10000 * p + s)
            cb, m = _memory(rng, p)
            rec, _ = m.recall(bits_to_wave(cb[0]).decohere(rng, flip_prob=FLIP))
            a = float(np.mean(wave_to_bits(WaveState(np.ones(N), rec)) == cb[0]))
            acc.append(a)
            exact.append(a == 1.0)
            fs.append(m.fidelity(rec, cb[0] * np.pi))
            fr.append(m.fidelity(rec, cb[0] * np.pi, slice_output=False))
        rows.append({
            "patterns": p, "load": round(p / N, 4),
            "bit_accuracy_mean": round(float(np.mean(acc)), 4),
            "exact_recall_rate": round(float(np.mean(exact)), 4),
            "fidelity_sliced_mean": round(float(np.mean(fs)), 4),
            "fidelity_unsliced_mean": round(float(np.mean(fr)), 4),
        })
    return rows


def controls(trials, p=5):
    out = {k: [] for k in ("stored", "unstored", "noise", "empty")}
    ghost_max, ghost_fid = [], []
    for s in range(trials):
        rng = np.random.default_rng(1000 + s)
        cb, m = _memory(rng, p)
        _, tr = m.recall(bits_to_wave(cb[0]).decohere(rng, flip_prob=FLIP))
        out["stored"].append(oes32_score(tr))
        g = rng.integers(0, 2, N)
        rec, tr = m.recall(bits_to_wave(g))
        out["unstored"].append(oes32_score(tr))
        ghost_max.append(float(np.max(m.overlaps(rec))))
        ghost_fid.append(m.fidelity(rec, g * np.pi))
        noise = WaveState(np.ones(N), rng.uniform(-np.pi, np.pi, N))
        _, tr = m.recall(noise)
        out["noise"].append(oes32_score(tr))
        e = HolographicMemory(N)
        e.finalize()
        _, tr = e.recall(noise)
        out["empty"].append(oes32_score(tr))
    rows = [{
        "cue": k, "oes32_mean": round(float(np.mean(v)), 4),
        "oes32_min": round(float(np.min(v)), 4),
        "fire_rate": round(float(np.mean(np.asarray(v) >= 0.5)), 4),
    } for k, v in out.items()]
    conf = {
        "unstored_max_overlap_mean": round(float(np.mean(ghost_max)), 4),
        "unstored_rate_max_overlap_over_0.9": round(float(np.mean(np.asarray(ghost_max) > 0.9)), 4),
        "unstored_fidelity_vs_own_bits_mean": round(float(np.mean(ghost_fid)), 4),
    }
    return rows, conf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args(argv)
    cap = capacity(a.trials)
    ctl, conf = controls(2 * a.trials)
    data = {"evidence_label": "SYNTHETIC", "n_modes": N, "flip_prob": FLIP,
            "trials_per_cell": a.trials, "control_trials": 2 * a.trials,
            "capacity": cap, "oes32_controls": ctl, "confabulation": conf}
    data["sha256"] = canonical_digest(data)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "sweeps.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    lines = ["# Sweeps (SYNTHETIC)", "",
             f"N = {N} modes, {FLIP:.0%} phase flips, 25 recall steps, {a.trials} seeded trials per row.", "",
             "## Capacity", "",
             "Each row uses its own seeds, so with 100 trials small reversals between rows "
             "(for example 10 vs 15 patterns) are sampling noise, not a real effect.", "",
             "| patterns | load P/N | bit accuracy | exact recall | fidelity (sliced) | fidelity (unsliced) |",
             "|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['patterns']} | {r['load']:.3f} | {r['bit_accuracy_mean']:.4f} | {r['exact_recall_rate']:.2f} | "
              f"{r['fidelity_sliced_mean']:.4f} | {r['fidelity_unsliced_mean']:.4f} |" for r in cap]
    lines += ["", f"## OES-32 on the recall trajectory ({2 * a.trials} trials, 5 patterns)", "",
              "| cue | mean score | min score | fire rate |", "|---|---:|---:|---:|"]
    lines += [f"| {r['cue']} | {r['oes32_mean']:.4f} | {r['oes32_min']:.4f} | {r['fire_rate']:.2f} |" for r in ctl]
    lines += ["", "## Never-stored cues", "",
              f"- mean max overlap of the recalled state with a stored pattern: {conf['unstored_max_overlap_mean']:.3f}",
              f"- share converging onto a stored pattern (max overlap > 0.9): {conf['unstored_rate_max_overlap_over_0.9']:.2f}",
              f"- mean fidelity against the cue's own bits: {conf['unstored_fidelity_vs_own_bits_mean']:.3f}",
              "", f"sha256: `{data['sha256']}`", ""]
    with open(os.path.join(a.outdir, "sweeps.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
