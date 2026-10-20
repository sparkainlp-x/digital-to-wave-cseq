"""SYNTHETIC evaluation of the v0.2.0 cue gate and verdicts.

Writes results/cue_gate.json and results/cue_gate.md. Deterministic.

Run:  python experiments/cue_gate_sweep.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cseq_wave import HolographicMemory, assess_recall, bits_to_wave, calibrate_cue_threshold  # noqa: E402
from cseq_wave.evidence import canonical_digest  # noqa: E402

N, JITTER, STEPS = 256, 0.3, 40
FLIPS = [0.2, 0.3, 0.35, 0.4, 0.45]


def _memory(rng, p):
    cb = rng.integers(0, 2, (p, N))
    m = HolographicMemory(N)
    for b in cb:
        m.store(bits_to_wave(b))
    m.finalize()
    return cb, m


def _margin(o):
    s = np.sort(o)
    return float(s[-1] - s[-2])


def separation(seeds=4, trials=60):
    tc, gc, tm, gm = [], [], [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        cb, m = _memory(rng, 4)
        for t in range(trials):
            c = bits_to_wave(cb[t % 4]).decohere(rng, flip_prob=0.2, jitter=JITTER)
            tc.append(float(m.overlaps(c.phase).max()))
            r, _ = m.recall(c, steps=STEPS)
            tm.append(_margin(m.overlaps(r)))
            g = bits_to_wave(rng.integers(0, 2, N)).decohere(rng, flip_prob=0.2, jitter=JITTER)
            gc.append(float(m.overlaps(g.phase).max()))
            r, _ = m.recall(g, steps=STEPS)
            gm.append(_margin(m.overlaps(r)))
    r4 = lambda x: round(float(x), 4)  # noqa: E731
    return {
        "trials": seeds * trials,
        "stored_cue_max_overlap_mean": r4(np.mean(tc)), "stored_cue_max_overlap_min": r4(np.min(tc)),
        "ghost_cue_max_overlap_mean": r4(np.mean(gc)), "ghost_cue_max_overlap_q99": r4(np.quantile(gc, 0.99)),
        "ghost_cue_max_overlap_max": r4(np.max(gc)),
        "stored_post_recall_margin_min": r4(np.min(tm)), "ghost_post_recall_margin_max": r4(np.max(gm)),
    }


def gate(models=20, trials=25, ghosts=4):
    rows = []
    for p in (4, 20):
        for flip in FLIPS:
            counts = {"VERIFIED_RECALL": 0, "UNVERIFIED_RECALL": 0, "CONFABULATION_SUSPECTED": 0, "WITHHELD": 0}
            ghost_pass, taus = [], []
            for seed in range(models):
                rng = np.random.default_rng(500 + seed)
                cb, m = _memory(rng, p)
                cal = calibrate_cue_threshold(m, seed=9000 + seed, n_null=256, flip_prob=0.2, jitter=JITTER)
                taus.append(cal["tau_cue_overlap"])
                for t in range(trials):
                    c = bits_to_wave(cb[t % p]).decohere(rng, flip_prob=flip, jitter=JITTER)
                    r, _ = m.recall(c, steps=STEPS)
                    counts[assess_recall(m, c, r, cal, target_phase=cb[t % p] * np.pi, target_index=t % p)["verdict"]] += 1
                    if flip == 0.2:
                        for _ in range(ghosts):
                            g = bits_to_wave(rng.integers(0, 2, N)).decohere(rng, flip_prob=0.2, jitter=JITTER)
                            ghost_pass.append(float(m.overlaps(g.phase).max()) >= cal["tau_cue_overlap"])
            n = models * trials
            rows.append({
                "patterns": p, "flip_prob": flip, "trials": n,
                "verified_rate": round(counts["VERIFIED_RECALL"] / n, 4),
                "unverified_rate": round(counts["UNVERIFIED_RECALL"] / n, 4),
                "confabulation_suspected_rate": round(counts["CONFABULATION_SUSPECTED"] / n, 4),
                "withheld_rate": round(counts["WITHHELD"] / n, 4),
                "ghost_false_accept_rate": round(float(np.mean(ghost_pass)), 4) if ghost_pass else None,
                "tau_mean": round(float(np.mean(taus)), 4),
            })
    return rows


def mixture():
    rng = np.random.default_rng(3)
    cb, m = _memory(rng, 4)
    cal = calibrate_cue_threshold(m, seed=1, flip_prob=0.2, jitter=JITTER)
    mix = cb[0].copy()
    mix[N // 2:] = cb[1][N // 2:]
    cue = bits_to_wave(mix)
    r, _ = m.recall(cue, steps=STEPS)
    v = assess_recall(m, cue, r, cal)
    return {
        "cue": "first half of pattern 0 + second half of pattern 1",
        "cue_overlaps": [round(float(x), 4) for x in m.overlaps(cue.phase)],
        "recalled_overlaps": [round(float(x), 4) for x in m.overlaps(r)],
        "tau": round(cal["tau_cue_overlap"], 4),
        "cue_gate": v["gates"]["cue_gate"], "verdict": v["verdict"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, default=20)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args(argv)
    data = {"evidence_label": "SYNTHETIC", "n_modes": N, "jitter": JITTER, "recall_steps": STEPS,
            "separation": separation(), "gate": gate(models=a.models), "mixture_cue": mixture()}
    data["sha256"] = canonical_digest(data)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "cue_gate.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    s, mx = data["separation"], data["mixture_cue"]
    L = ["# Cue gate and verdicts (SYNTHETIC)", "",
         f"N = {N} modes, {JITTER} rad jitter, {STEPS} recall steps. tau = q0.99 of 256 never-stored cues per hologram.", "",
         f"## Separation (4 patterns, 20% flips, {s['trials']} trials of each kind)", "",
         "| signal | stored cue | never-stored cue |", "|---|---:|---:|",
         f"| cue max overlap, mean | {s['stored_cue_max_overlap_mean']:.4f} | {s['ghost_cue_max_overlap_mean']:.4f} |",
         f"| cue max overlap, worst case | min {s['stored_cue_max_overlap_min']:.4f} | q0.99 {s['ghost_cue_max_overlap_q99']:.4f}, max {s['ghost_cue_max_overlap_max']:.4f} |",
         f"| post-recall winner margin, worst case | min {s['stored_post_recall_margin_min']:.4f} | max {s['ghost_post_recall_margin_max']:.4f} |",
         "", "The cue-side overlap separates the two groups; the post-recall margin does not.", "",
         f"## Verdicts on stored cues by decoherence ({a.models} holograms x 25 cues per row, ground-truth mode)", "",
         "| patterns | flips | verified | unverified | confabulation suspected | ghost false accept | tau |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data["gate"]:
        fa = "-" if r["ghost_false_accept_rate"] is None else f"{r['ghost_false_accept_rate']:.4f}"
        L.append(f"| {r['patterns']} | {r['flip_prob']} | {r['verified_rate']:.3f} | {r['unverified_rate']:.3f} | "
                 f"{r['confabulation_suspected_rate']:.3f} | {fa} | {r['tau_mean']:.4f} |")
    L += ["", "Ghost false accepts were measured at 20% flips only (4 ghosts per stored cue). At 40% flips and above, "
          "most genuinely stored cues are rejected as suspected confabulation: the gate trades misses for safety. "
          "With 20 patterns the cue gate passes, but recall itself fails under 0.3 rad jitter "
          "(continuous phases leave the BPSK attractors), so the verdict is UNVERIFIED_RECALL. "
          "That is the verdict doing its job, not a gate failure.", "",
          "## Known blind spot: mixture cue", "",
          f"Cue = {mx['cue']}. Cue overlaps {mx['cue_overlaps']} (tau {mx['tau']}), so the cue gate "
          f"{'passes' if mx['cue_gate'] else 'fails'}. Recalled overlaps {mx['recalled_overlaps']}: the dynamics land on a "
          f"spurious mixture. Verdict: {mx['verdict']}.", "", f"sha256: `{data['sha256']}`", ""]
    with open(os.path.join(a.outdir, "cue_gate.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
