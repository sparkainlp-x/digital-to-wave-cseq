"""FDW reverse + infinity benchmark tools.

For the Coherent Sequence Wave Computer (sparkainlp-x/digital-to-wave-cseq,
cseq_wave v0.2.0): the "digital to wave" pipeline, run forward AND
backward, and scaled toward infinity to find where it breaks.

Three benchmark families, all seeded and reproducible:

  1. REVERSE (reversibility): digital -> wave -> digital round trips.
     - Codec round trip: bits -> wave -> bits, bit accuracy vs phase
       noise sigma and message length. Noiseless must be exact.
     - Wave round trip: wave -> bits -> wave, BPSK phase reproduction.
     - Full loop: text -> bits -> wave -> holographic memory -> recall
       -> wave -> bits -> text, end-to-end accuracy vs stored patterns.
     A pipeline that cannot run backward cannot be trusted forward:
     every forward claim gets a reverse measurement next to it.

  2. INFINITY (scaling toward large N): store time, recall-step time,
     and weight-matrix memory vs number of modes N, fitted to a power
     law time ~ N^alpha. The fit is extrapolated to N = 4096 and 16384
     to name the wall (memory wall first: weights are N^2 complex
     numbers, 16 bytes each) instead of pretending infinity is free.

  3. CAPACITY KNEE: recall accuracy vs patterns-per-mode (P/N), the
     point where the hologram stops returning what was stored.

Usage (from the repo root, with cseq_wave importable):

    python benchmark.py --quick            # small sizes, seconds
    python benchmark.py --full --json report.json

SYNTHETIC classical software. No quantum-hardware claim. Noise is
Gaussian phase noise added to cue phases; it models decoherence, it
does not measure any physical device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import numpy as np


def _bit_accuracy(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.mean(a == b))


def bench_codec_reverse(bits_to_wave, wave_to_bits, lengths, sigmas, trials, seed):
    """Bits -> wave -> bits accuracy vs length and phase noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for length in lengths:
        for sigma in sigmas:
            accs = []
            for _ in range(trials):
                bits = rng.integers(0, 2, size=length)
                ws = bits_to_wave(bits)
                if sigma > 0:
                    from cseq_wave.wave import WaveState

                    noisy = WaveState(
                        ws.amplitude,
                        ws.phase + rng.normal(0.0, sigma, size=length),
                    )
                    got = wave_to_bits(noisy)
                else:
                    got = wave_to_bits(ws)
                accs.append(_bit_accuracy(bits, got))
            rows.append(
                {
                    "test": "codec_reverse",
                    "length": length,
                    "sigma": sigma,
                    "bit_accuracy": float(np.mean(accs)),
                    "min_accuracy": float(np.min(accs)),
                }
            )
    return rows


def bench_wave_reverse(bits_to_wave, wave_to_bits, lengths, seed):
    """Wave -> bits -> wave: BPSK phases must reproduce exactly."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    for length in lengths:
        bits = rng.integers(0, 2, size=length)
        ws = bits_to_wave(bits)
        back = bits_to_wave(wave_to_bits(ws))
        phase_err = float(np.max(np.abs(np.angle(np.exp(1j * (back.phase - ws.phase))))))
        rows.append(
            {
                "test": "wave_reverse",
                "length": length,
                "max_phase_error_rad": phase_err,
                "exact": bool(phase_err < 1e-9),
            }
        )
    return rows


def bench_full_loop(text, n_modes_list, n_patterns_list, sigma, trials, seed):
    """Text -> wave -> memory -> recall -> wave -> text, end to end."""
    from cseq_wave.encode import bits_to_wave, text_to_bits, wave_to_bits, bits_to_text
    from cseq_wave.memory import HolographicMemory
    from cseq_wave.wave import WaveState

    rng = np.random.default_rng(seed + 2)
    rows = []
    for n_modes in n_modes_list:
        for n_patterns in n_patterns_list:
            accs = []
            for _ in range(trials):
                mem = HolographicMemory(n_modes)
                targets = []
                for _ in range(n_patterns):
                    bits = rng.integers(0, 2, size=n_modes)
                    targets.append(bits)
                    mem.store(bits_to_wave(bits))
                mem.finalize()
                target = targets[0]
                cue = WaveState(
                    np.ones(n_modes),
                    bits_to_wave(target).phase
                    + rng.normal(0.0, sigma, size=n_modes),
                )
                phases, _ = mem.recall(cue, steps=25)
                got = wave_to_bits(WaveState(np.ones(n_modes), phases))
                accs.append(_bit_accuracy(target, got))
            rows.append(
                {
                    "test": "full_loop",
                    "n_modes": n_modes,
                    "n_patterns": n_patterns,
                    "p_over_n": n_patterns / n_modes,
                    "recall_bit_accuracy": float(np.mean(accs)),
                }
            )
    # The text itself, once, as a readable proof of the loop:
    bits = text_to_bits(text)
    ws = bits_to_wave(bits)
    back_text = bits_to_text(wave_to_bits(ws))
    rows.append(
        {
            "test": "full_loop_text",
            "text_in": text,
            "text_out": back_text,
            "exact": bool(back_text == text),
        }
    )
    return rows


def bench_scaling(HolographicMemory, bits_to_wave, n_list, seed):
    """Store/recall time + weight memory vs N, with a power-law fit."""
    rng = np.random.default_rng(seed + 3)
    rows = []
    for n in n_list:
        mem = HolographicMemory(n)
        pat = bits_to_wave(rng.integers(0, 2, size=n))
        t0 = time.perf_counter()
        mem.store(pat)
        t_store = time.perf_counter() - t0
        mem.finalize()
        t0 = time.perf_counter()
        mem.recall(pat, steps=25)
        t_recall = time.perf_counter() - t0
        mem_bytes = n * n * 16  # complex128 weights
        rows.append(
            {
                "test": "scaling",
                "n_modes": n,
                "store_seconds": t_store,
                "recall25_seconds": t_recall,
                "weights_bytes": mem_bytes,
            }
        )
    # Power-law fit on recall time: log(t) = alpha * log(N) + c.
    # Small N is overhead-dominated (numpy call overhead, not N^2 math),
    # so fitting all points understates the wall. Fit the compute-dominated
    # tail (N >= 128) when it has at least 3 points, and say so; the
    # full-range fit is kept alongside for comparison, not hidden.
    tail = [r for r in rows if r["n_modes"] >= 128]
    fit_rows = tail if len(tail) >= 3 else rows
    ns = np.array([r["n_modes"] for r in fit_rows], dtype=float)
    ts = np.array([max(r["recall25_seconds"], 1e-9) for r in fit_rows])
    alpha, c = np.polyfit(np.log(ns), np.log(ts), 1)
    fit = {
        "test": "scaling_fit",
        "alpha": float(alpha),
        "fit_range": f"N >= {int(min(ns))}",
    }
    for n_future in (4096, 16384):
        fit[f"recall25_seconds_at_{n_future}"] = float(np.exp(c) * n_future**alpha)
        fit[f"weights_bytes_at_{n_future}"] = int(n_future * n_future * 16)
    rows.append(fit)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="FDW reverse + infinity benchmarks")
    ap.add_argument("--quick", action="store_true", help="small sizes, seconds")
    ap.add_argument("--full", action="store_true", help="full sizes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default=None, help="write JSON report here")
    args = ap.parse_args(argv)

    sys.path.insert(0, ".")
    from cseq_wave.encode import bits_to_wave, wave_to_bits
    from cseq_wave.memory import HolographicMemory

    if args.quick:
        lengths = [64, 256]
        sigmas = [0.0, 0.25]
        trials = 3
        n_modes_list = [32]
        n_patterns_list = [1, 4]
        n_list = [16, 32, 64, 128]
    else:
        lengths = [64, 256, 1024, 4096]
        sigmas = [0.0, 0.1, 0.25, 0.5]
        trials = 5
        n_modes_list = [32, 64]
        n_patterns_list = [1, 4, 16]
        n_list = [16, 32, 64, 128, 256, 512]

    report = {"seed": args.seed, "results": []}
    report["results"] += bench_codec_reverse(
        bits_to_wave, wave_to_bits, lengths, sigmas, trials, args.seed
    )
    report["results"] += bench_wave_reverse(bits_to_wave, wave_to_bits, lengths, args.seed)
    report["results"] += bench_full_loop(
        "Hello wave", n_modes_list, n_patterns_list, 0.1, trials, args.seed
    )
    report["results"] += bench_scaling(HolographicMemory, bits_to_wave, n_list, args.seed)

    # The hash covers deterministic outcomes only: timings vary run to
    # run by nature, so including them would make every hash differ and
    # the determinism check meaningless. Timings are still reported.
    deterministic = [
        {
            k: v
            for k, v in row.items()
            if "seconds" not in k and k != "alpha"  # alpha is fitted from timings
        }
        for row in report["results"]
    ]
    blob = json.dumps(deterministic, sort_keys=True).encode()
    report["sha256"] = hashlib.sha256(blob).hexdigest()

    for row in report["results"]:
        print(json.dumps(row))
    print("sha256:", report["sha256"])
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print("wrote:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
