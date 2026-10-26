"""POST HOC check (not preregistered, written after the round-3 results were seen).

Round 3 found that with correlated patterns (pairwise correlation rho) the
25-step recall accuracy plateaus near (1 + sqrt(rho)) / 2. That is the
expected agreement between any stored pattern and the common +-1 pattern c
used by the generator. This script tests that reading directly: for exact
cues (q = 0), how often does the recalled state end within 0.10 of c rather
than near the target?

Usage: python experiments/correlated_collapse_check.py [--json out.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_round3 as b3  # noqa: E402
from cseq_wave.encode import wave_to_bits  # noqa: E402
from cseq_wave.wave import WaveState  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args(argv)
    rows = []
    for rho in (0.1, 0.3):
        for n in (64, 128, 256):
            for ratio in (0.0625, 0.125, 0.25):
                p = int(round(n * ratio))
                rng = np.random.default_rng([20261028, n, int(rho * 10), p])
                near_c, near_t, acc, agree_t_c = 0, 0, [], []
                for _ in range(args.trials):
                    c = rng.integers(0, 2, size=n)
                    agree = rng.random((p, n)) < (1 + math.sqrt(rho)) / 2
                    patterns = np.where(agree, c, 1 - c)
                    mem = b3.build_memory(patterns)
                    phases, _ = mem.recall(WaveState(np.ones(n), patterns[0] * np.pi), steps=25)
                    got = wave_to_bits(WaveState(np.ones(n), phases))
                    near_c += int(np.mean(got != c) <= 0.10)
                    near_t += int(np.mean(got != patterns[0]) <= 0.10)
                    acc.append(float(np.mean(got == patterns[0])))
                    agree_t_c.append(float(np.mean(patterns[0] == c)))
                rows.append({"rho": rho, "n_modes": n, "p_over_n": ratio, "trials": args.trials,
                             "predicted_agreement_with_c": round((1 + math.sqrt(rho)) / 2, 4),
                             "mean_target_agreement_with_c": round(float(np.mean(agree_t_c)), 4),
                             "mean_recall_accuracy_vs_target": round(float(np.mean(acc)), 4),
                             "final_within_0.10_of_c": near_c, "final_within_0.10_of_target": near_t})
    for r in rows:
        print(r)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
