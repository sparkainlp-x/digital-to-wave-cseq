# Benchmark tools: reverse, scaling and capacity (SYNTHETIC)

`benchmark.py` (repo root) runs three seeded benchmark families on the `cseq_wave` pipeline:

1. **Reverse**: digital to wave to digital round trips (codec, wave, and the "Hello wave" loop).
2. **Scaling**: store and recall time and weight-matrix memory against the number of modes N.
3. **Capacity knee**: recall bit accuracy against patterns per mode (P/N).

```sh
python benchmark.py --quick                         # seconds
python benchmark.py --full --json report.json       # full report (report.json is git-ignored)
```

All results are **synthetic classical software**. No quantum-hardware claim is made. Noise is Gaussian phase noise added to cue phases; it does not measure any physical device.

`benchmark.py` is unchanged from the version supplied. The `cseq_wave` imports it uses (`wave.WaveState`, `encode.bits_to_wave/wave_to_bits/text_to_bits/bits_to_text`, `memory.HolographicMemory.store/finalize/recall`) exist in the current package and needed no adapter.

## Deterministic results (full run, seed 42)

Identical in three consecutive full runs (same sha256, see below).

| Test | Result |
| --- | --- |
| Codec round trip, sigma 0, 0.1, 0.25 | bit accuracy 1.000 at lengths 64, 256, 1024, 4096 (5 trials each) |
| Codec round trip, sigma 0.5 | 1.000 at length 64; 0.999 (256), 0.998 (1024), 0.998 (4096) |
| Wave round trip | max phase error 0.0 rad at all four lengths |
| "Hello wave" text loop | exact |
| Recall accuracy, P/N 0.016, 0.031, 0.0625 | 1.000 |
| Recall accuracy, P/N 0.125 (N=32, 4 patterns) | 0.906 (not part of the headline set) |
| Recall accuracy, P/N 0.25 (N=64, 16 patterns) | 0.753 |
| Recall accuracy, P/N 0.50 (N=32, 16 patterns) | 0.6625 |

**Scope of the capacity numbers:** they come from only N = 32 and N = 64, five trials per cell, one cue noise level (sigma 0.1), 25 recall steps, and the first stored pattern as the target. The knee shape is indicative, not a capacity law, and P/N is confounded with N in this grid.

## Measured vs extrapolated

The full run **measures** N = 16 to 512 only. Everything at N = 4096 and 16384 produced by `benchmark.py` is a power-law **extrapolation** (the report keys `recall25_seconds_at_4096` and `..._at_16384` are predictions, not measurements).

| Quantity | N = 512 and below | N = 4096 | N = 16384 |
| --- | --- | --- | --- |
| Weight memory (complex128, 16 N^2 bytes) | measured allocation size: 4 MiB at 512 | 256 MiB (exact arithmetic; 256 MiB allocated and used in the direct run below) | 4 GiB (exact arithmetic; **not allocated or run**) |
| Recall time, 25 steps | measured: 1.2 ms at N = 512 (run 1) | **extrapolated** by the fit: 3.8 ms, 4.8 ms, 86 ms in three runs; **measured directly**: 0.09 s (8 BLAS threads) or 0.49 s (1 thread) | **extrapolated** only: 8.5 ms, 11 ms, 0.56 s in three runs; **not measured** |
| Fitted exponent alpha (N >= 128) | 0.58, 0.57, 1.35 in three identical-seed runs | | |

The memory figures are exact (N^2 complex numbers at 16 bytes each). The timing figures are **not reliable**:

- The N >= 128 fit uses only 3 points (128, 256, 512), all in a regime where recall time is dominated by call overhead and multi-threaded BLAS start-up, so alpha moves a lot between identical runs (0.57 to 1.35 here). It is a poor guide to large N.
- Direct measurements at larger N (`experiments/measure_large_n.py`, same machine, 8 cores, numpy 2.5.4 with scipy-openblas) show recall time growing faster than the fit: with 1 thread, 14.6 ms at N = 1024, 60 ms at 2048, 486 ms at 4096 (about N^2.5 over that range, with cache effects). With 8 threads: 3 ms, 14 ms, 86 ms.
- Recall time at N = 4096 is therefore machine- and thread-dependent: about 0.09 s to 0.5 s here. The supplied write-up's "about 0.3 to 0.4 s" falls in that range but was not reproduced by the extrapolation in this environment (3.8 ms to 86 ms), and the "about 2.2 to 3.5 s at N = 16384" figure was not reproduced either. N = 16384 was not run: building the matrix needs 4 GiB for the weights plus a temporary of equal size inside `store`.
- The "alpha about 1.4 to 1.5" claim was seen in one of three runs (1.35) and not in the other two. Treat alpha as unstable at these sizes.

What does hold: weight memory grows as N^2, so memory is the first hard limit (256 MiB at N = 4096, 4 GiB at N = 16384).

## Determinism

The report hash covers deterministic outcomes only. Fields containing "seconds", and the timing-fitted `alpha`, are excluded, because timings differ from run to run. Three consecutive full runs (seed 42) gave the same value:

```
sha256: 39edd70a25f4ccc370f0bc2378eb12388e470736e3ec4ffe5a20c1e03c34b45e
```

`tests/test_benchmark.py` runs `benchmark.py --quick` twice and checks that the hashes agree and that the key results hold. The `--quick` hash is `44430417efe0c6228205f941a1a414212bf358f7e1f4ac5c2f74678425e1a43e` on the machine used here; the test compares two runs rather than a fixed value, so it does not depend on the numpy version. In `--quick` mode the scaling fit uses all points (fewer than 3 points have N >= 128) and its extrapolations are not meaningful.

## Two bugs found while building the tool (kept in the record)

1. Fitting all N understated the wall (alpha about 0.89, overhead dominated). Fix: fit the N >= 128 tail and report the fit range. As shown above, this tail is itself still short and noisy.
2. The determinism hash first included timings, then timing-derived predictions, then the timing-fitted alpha, so identical runs hashed differently. Fix: the hash excludes everything derived from timings. Adding a new timing-derived field without excluding it makes the determinism check fail.
