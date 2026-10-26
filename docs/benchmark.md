# Benchmark tools: reverse, scaling and capacity (SYNTHETIC)

Two tools live in the repo root:

| File | Role |
| --- | --- |
| `benchmark.py` | The first benchmark, supplied unchanged. Quick to run; its scaling fit is unstable (see "Original tool" below). |
| `benchmark_rigorous.py` | The upgraded benchmark: locked analysis plan, direct timings with confidence intervals, de-confounded capacity grid, null baselines, adversarial reverse tests. Settings live in `benchmark_protocol.json`. |
| `benchmark_round4.py` | Round 4: held-out early-stopping test, spurious-state characterisation, BLAS thread counts for complex128 recall. Settings live in `benchmark_protocol_round4.json`. |
| `benchmark_round3.py` | Round 3: why recall degrades (cross-talk, spurious states, mirror, dynamics), correlated patterns, multiplicity re-analysis, cache microbenchmark, complex128 timed to N = 14336. Settings live in `benchmark_protocol_round3.json`. |
| `benchmark_round2.py` | Round 2: bit-flip and phase-noise denoising test with paired CIs, load-controlled timings, complex64 timing at N = 16384. Settings live in `benchmark_protocol_round2.json`. See "Round 2" below. |

All results are **synthetic classical software**. No quantum-hardware claim is made. Noise is Gaussian phase noise added to cue phases; it does not measure any physical device.

## Reproduction

```sh
pip install -r requirements.txt
python benchmark_rigorous.py --quick                              # CI smoke: seconds, prints the deterministic hash
python benchmark_rigorous.py --full --json report_rigorous.json   # manual, under two minutes on this 8-core box
python -m pytest -q                                               # includes tests/test_benchmark_rigorous.py
```

The full run is manual only (the `Benchmark (full, manual)` workflow can be started from the Actions tab); CI runs the quick mode twice and diffs the hash.

## Protocol lock

`benchmark_protocol.json` fixes the analysis plan before the full run is inspected: seeds, fit range (N >= 512), repeats (11, or 7 for N >= 3072), warm-up runs (2), bootstrap resamples (2000), the knee threshold (mean bit accuracy < 0.95), the hold-out check and the original claims to be tested. Its SHA-256 (`cseq_wave.evidence.canonical_digest`, the same canonical hash as the evidence passports) is stored in `benchmark_protocol.lock` and recorded in every report as `protocol_sha256`. The script refuses to run if the two disagree unless `--allow-protocol-change` is given, in which case the report says `protocol_locked: false`.

Protocol hash for this version: `826e3e6870f9e3b95b605e2e70701067b970909f196b739d8e662dd06f094c9f`.

Limits of the lock: it is a commitment recorded in git, not an external timestamp. The protocol was written before the full run, but after the quick runs and after the earlier `benchmark.py` results were known; the fit range N >= 512 was fixed before any rigorous full run.

## Determinism

`deterministic_sha256` covers the protocol hash, profile, seed, reverse results, capacity results and the scaling plan (sizes, repeats, warm-up). It excludes every timing and everything derived from timings (alpha, CIs, slopes, extrapolations) and the environment. Six full runs on this machine (two batches of three, same code) and repeated quick runs gave identical hashes within each profile:

```
full : 3a906ad0db1ca828a9e54f6902e3ffeb96e0f43a0ffe98382c1a3d76445cf5a0
```

The quick hash is not pinned; `tests/test_benchmark_rigorous.py` and CI check that two quick runs agree.

## Results (full run, this machine)

Machine: Intel(R) Xeon(R) Processor, 8 cores, Python 3.13.5, numpy 2.5.4 with scipy-openblas 0.3.34.106.0 (OpenBLAS 0.3.34.106.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=64). Thread modes were run in separate subprocesses: `blas_threads_1` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS` to 1; `blas_threads_default` leaves them unset. The report stores the CPU, BLAS build, thread variables and load average for each mode.

### 1. Scaling (recall, 25 steps)

All N below were **measured directly** (2 warm-up runs, then 11 repeats, or 7 for N >= 3072). N = 16384 was **not measured**: it needs 4 GiB of weights and about 8 GiB peak during `store`, over the 4 GiB cap in the protocol.

**One BLAS thread**

| N | median (s) | IQR (s) | 95% CI of median (s) | effective GFLOP/s | repeats |
|---|---|---|---|---|---|
| 64 | 0.000380 | 0.000013 | [0.000377, 0.000391] | 2.2 | 11 |
| 128 | 0.000502 | 0.000019 | [0.000497, 0.000517] | 6.5 | 11 |
| 192 | 0.000744 | 0.000086 | [0.000683, 0.000773] | 9.9 | 11 |
| 256 | 0.000944 | 0.000057 | [0.000923, 0.001017] | 13.9 | 11 |
| 384 | 0.002052 | 0.000111 | [0.002013, 0.002129] | 14.4 | 11 |
| 512 | 0.004380 | 0.000110 | [0.004278, 0.004429] | 12.0 | 11 |
| 768 | 0.008835 | 0.000420 | [0.008621, 0.009103] | 13.4 | 11 |
| 1024 | 0.015284 | 0.000262 | [0.015261, 0.015534] | 13.7 | 11 |
| 1536 | 0.035780 | 0.004870 | [0.034665, 0.041509] | 13.2 | 11 |
| 2048 | 0.062874 | 0.019412 | [0.062414, 0.097851] | 13.3 | 11 |
| 3072 | 0.139288 | 0.003603 | [0.137929, 0.143967] | 13.6 | 7 |
| 4096 | 0.480522 | 0.152097 | [0.387362, 0.567530] | 7.0 | 7 |
| 6144 | 1.239649 | 0.219884 | [1.209301, 1.557721] | 6.1 | 7 |
| 8192 | 2.252363 | 0.097343 | [2.208151, 2.316621] | 6.0 | 7 |

Local log-log slopes between adjacent sizes (bootstrap 95% CI): 64→128: 0.40 [0.35, 0.45], 128→192: 0.97 [0.69, 1.07], 192→256: 0.83 [0.68, 1.17], 256→384: 1.92 [1.73, 2.04], 384→512: 2.63 [2.46, 2.73], 512→768: 1.73 [1.66, 1.83], 768→1024: 1.91 [1.80, 2.02], 1024→1536: 2.10 [2.01, 2.46], 1536→2048: 1.96 [1.44, 3.54], 2048→3072: 1.96 [0.87, 2.05], 3072→4096: 4.30 [3.44, 4.89], 4096→6144: 2.34 [1.89, 3.18], 6144→8192: 2.08 [1.23, 2.24]

**Default BLAS threads**

| N | median (s) | IQR (s) | 95% CI of median (s) | effective GFLOP/s | repeats |
|---|---|---|---|---|---|
| 64 | 0.000336 | 0.000005 | [0.000334, 0.000340] | 2.4 | 11 |
| 128 | 0.000566 | 0.000045 | [0.000542, 0.000595] | 5.8 | 11 |
| 192 | 0.000593 | 0.000031 | [0.000582, 0.000621] | 12.4 | 11 |
| 256 | 0.000733 | 0.000053 | [0.000707, 0.000764] | 17.9 | 11 |
| 384 | 0.000809 | 0.000136 | [0.000804, 0.000948] | 36.5 | 11 |
| 512 | 0.001093 | 0.000192 | [0.001025, 0.001301] | 47.9 | 11 |
| 768 | 0.001751 | 0.000170 | [0.001738, 0.001930] | 67.4 | 11 |
| 1024 | 0.003099 | 0.001848 | [0.003045, 0.005558] | 67.7 | 11 |
| 1536 | 0.006518 | 0.000717 | [0.006390, 0.007344] | 72.4 | 11 |
| 2048 | 0.010573 | 0.000788 | [0.010342, 0.011476] | 79.3 | 11 |
| 3072 | 0.022671 | 0.002182 | [0.021348, 0.025305] | 83.3 | 7 |
| 4096 | 0.052412 | 0.005425 | [0.049204, 0.056449] | 64.0 | 7 |
| 6144 | 0.176604 | 0.084807 | [0.164313, 0.259493] | 42.7 | 7 |
| 8192 | 0.303315 | 0.005921 | [0.300001, 0.309178] | 44.3 | 7 |

Local log-log slopes: 64→128: 0.75 [0.69, 0.83], 128→192: 0.11 [-0.04, 0.28], 192→256: 0.74 [0.53, 0.92], 256→384: 0.24 [0.14, 0.66], 384→512: 1.05 [0.44, 1.66], 512→768: 1.16 [0.73, 1.43], 768→1024: 1.98 [1.65, 4.02], 1024→1536: 1.83 [0.36, 2.13], 1536→2048: 1.68 [1.27, 1.97], 2048→3072: 1.88 [1.68, 2.18], 3072→4096: 2.91 [2.33, 3.22], 4096→6144: 3.00 [2.74, 4.05], 6144→8192: 1.88 [0.53, 2.15]

Reading the tables:

- **Overhead to compute transition.** For one thread the local slope is below 1 up to N of about 256 (call overhead dominates) and about 2 from N = 256 on, which is the N^2 matvec cost. With default threads it stays near or below 1 until about N = 768 and effective throughput climbs from about 2 GFLOP/s to about 80 GFLOP/s.
- **A step above N^2 near N = 3072 to 4096.** The slope jumps to about 3 to 4 and throughput roughly halves (one thread: about 13.5 to about 7 GFLOP/s). This coincides with the weight matrix growing from 144 MiB to 256 MiB; a cache or memory-bandwidth effect is a plausible explanation but it was not tested here.
- **Fit on N >= 512 (fixed a priori; 9 points).**
  - One thread: alpha = 2.309, 95% bootstrap CI [2.284, 2.357].
  - Default threads: alpha = 2.078, 95% bootstrap CI [2.013, 2.139].
  - Theoretical matvec cost: alpha = 2 (25 steps of N^2 complex multiply-adds). Neither CI contains 2, because of the step described above. The fit is a single power law over a range that includes that step, so it is a summary, not a mechanism.
- **Run-to-run variability is larger than the within-run CI.** Three further full runs in a second batch gave alpha 2.309, 2.337, 2.316 (one thread) and 2.078, 2.146, 2.023 (default). An earlier batch of three runs, executed while other jobs loaded the shared machine (1-minute load average up to about 30; load was not yet recorded in that batch), gave 2.275, 2.275, 2.283 and 2.028, 1.886, 1.711 (one of its default-thread CIs, [1.41, 1.80], overlapped 1.4 to 1.5). The second batch ran at 1-minute load of about 3 to 12 (see the report), still not idle. Treat the bootstrap CI as sampling noise inside one run only.
- **Extrapolation to N = 16384 (labelled, not measured).** Point estimates from the three runs of the second batch: about 9.8, 10.3 and 10.1 s (one thread; bootstrap intervals about [9.2, 11.4], [10.1, 10.6], [9.8, 10.3]) and about 1.0, 1.3 and 0.9 s (default threads). These intervals reflect fit noise only. The **hold-out check** (fit on N = 512 to 2048, predict N = 4096 and 8192, which were measured) found the prediction at 0.36 to 0.49 of the measured time with one thread and 0.35 to 0.94 with default threads, so the extrapolation underestimates and the real value at 16384 is probably higher. It remains unmeasured.

### 2. Capacity (P/N de-confounded from N)

Same P/N values at N = 32, 64, 128, 256; 40 trials per cell (fresh random patterns each trial; target is the first stored pattern; 25 recall steps); noise sigma in {0, 0.1, 0.25} radians of Gaussian phase noise on the cue. CIs are percentile bootstrap over trials (2000 resamples); the report also holds Wilson intervals for the exact-recall fraction.

Reference baselines:

- **cue only**: hard-slice the noisy cue itself, no memory.
- **nearest stored pattern**: return the stored pattern with the largest matched-filter overlap with the cue.
- **random pattern**: an independent random bit string (null, expected 0.5).

**sigma = 0.0** (mean bit accuracy with 95% bootstrap CI over 40 trials; baselines averaged over N)

| P/N | N=32 | N=64 | N=128 | N=256 | cue only | nearest stored pattern | random pattern |
|---|---|---|---|---|---|---|---|
| 0.03125 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.498 |
| 0.0625 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.498 |
| 0.09375 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.992 [0.981, 1.000] | 1.000 [0.999, 1.000] | 1.000 | 1.000 | 0.496 |
| 0.125 | 1.000 [1.000, 1.000] | 0.986 [0.970, 0.998] | 0.982 [0.962, 0.997] | 0.978 [0.961, 0.991] | 1.000 | 1.000 | 0.513 |
| 0.1875 | 0.994 [0.985, 1.000] | 0.928 [0.898, 0.955] | 0.918 [0.886, 0.949] | 0.855 [0.812, 0.897] | 1.000 | 1.000 | 0.498 |
| 0.25 | 0.913 [0.874, 0.946] | 0.839 [0.793, 0.882] | 0.744 [0.709, 0.778] | 0.714 [0.692, 0.739] | 1.000 | 1.000 | 0.501 |
| 0.375 | 0.830 [0.787, 0.875] | 0.728 [0.698, 0.761] | 0.685 [0.662, 0.705] | 0.664 [0.650, 0.679] | 1.000 | 1.000 | 0.493 |
| 0.5 | 0.774 [0.739, 0.809] | 0.700 [0.673, 0.730] | 0.682 [0.659, 0.708] | 0.646 [0.631, 0.660] | 1.000 | 1.000 | 0.507 |

**sigma = 0.1** (mean bit accuracy with 95% bootstrap CI over 40 trials; baselines averaged over N)

| P/N | N=32 | N=64 | N=128 | N=256 | cue only | nearest stored pattern | random pattern |
|---|---|---|---|---|---|---|---|
| 0.03125 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.499 |
| 0.0625 | 1.000 [1.000, 1.000] | 0.997 [0.991, 1.000] | 0.985 [0.970, 0.996] | 0.963 [0.948, 0.976] | 1.000 | 1.000 | 0.503 |
| 0.09375 | 0.987 [0.966, 1.000] | 0.922 [0.894, 0.947] | 0.883 [0.855, 0.911] | 0.848 [0.827, 0.870] | 1.000 | 1.000 | 0.502 |
| 0.125 | 0.907 [0.872, 0.939] | 0.878 [0.852, 0.905] | 0.787 [0.760, 0.812] | 0.740 [0.718, 0.762] | 1.000 | 1.000 | 0.504 |
| 0.1875 | 0.838 [0.810, 0.866] | 0.781 [0.754, 0.807] | 0.721 [0.695, 0.749] | 0.690 [0.675, 0.706] | 1.000 | 1.000 | 0.495 |
| 0.25 | 0.772 [0.737, 0.808] | 0.716 [0.686, 0.743] | 0.700 [0.675, 0.725] | 0.670 [0.657, 0.683] | 1.000 | 1.000 | 0.499 |
| 0.375 | 0.743 [0.719, 0.768] | 0.701 [0.674, 0.728] | 0.668 [0.648, 0.688] | 0.631 [0.620, 0.643] | 1.000 | 1.000 | 0.499 |
| 0.5 | 0.708 [0.678, 0.737] | 0.662 [0.637, 0.686] | 0.669 [0.651, 0.686] | 0.641 [0.629, 0.653] | 1.000 | 1.000 | 0.503 |

**sigma = 0.25** (mean bit accuracy with 95% bootstrap CI over 40 trials; baselines averaged over N)

| P/N | N=32 | N=64 | N=128 | N=256 | cue only | nearest stored pattern | random pattern |
|---|---|---|---|---|---|---|---|
| 0.03125 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.999, 1.000] | 1.000 | 1.000 | 0.489 |
| 0.0625 | 1.000 [1.000, 1.000] | 0.988 [0.972, 1.000] | 0.959 [0.937, 0.977] | 0.953 [0.937, 0.968] | 1.000 | 1.000 | 0.505 |
| 0.09375 | 0.987 [0.970, 1.000] | 0.936 [0.908, 0.961] | 0.876 [0.846, 0.904] | 0.812 [0.786, 0.839] | 1.000 | 1.000 | 0.503 |
| 0.125 | 0.894 [0.863, 0.923] | 0.854 [0.824, 0.882] | 0.766 [0.741, 0.791] | 0.734 [0.712, 0.756] | 1.000 | 1.000 | 0.496 |
| 0.1875 | 0.795 [0.762, 0.827] | 0.737 [0.708, 0.766] | 0.698 [0.675, 0.720] | 0.674 [0.658, 0.688] | 1.000 | 1.000 | 0.499 |
| 0.25 | 0.805 [0.773, 0.836] | 0.719 [0.696, 0.742] | 0.675 [0.655, 0.694] | 0.655 [0.641, 0.670] | 1.000 | 1.000 | 0.503 |
| 0.375 | 0.755 [0.724, 0.784] | 0.679 [0.650, 0.710] | 0.656 [0.638, 0.676] | 0.638 [0.625, 0.652] | 1.000 | 1.000 | 0.486 |
| 0.5 | 0.722 [0.691, 0.754] | 0.679 [0.658, 0.700] | 0.653 [0.636, 0.672] | 0.632 [0.618, 0.646] | 1.000 | 1.000 | 0.496 |

Knee (smallest tested P/N whose mean bit accuracy is below 0.95, as fixed in the protocol):

| sigma | N = 32 | N = 64 | N = 128 | N = 256 | Stable within one grid step? |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.25 | 0.1875 | 0.1875 | 0.1875 | yes |
| 0.1 | 0.125 | 0.09375 | 0.09375 | 0.09375 | yes |
| 0.25 | 0.125 | 0.09375 | 0.09375 | 0.09375 | yes |

What this shows and what it does not:

- **The knee location is similar across N** by the pre-declared rule (same grid step for N >= 64, one step later at N = 32), but the grid is coarse (steps of 1.33x to 1.5x), and at fixed P/N accuracy still falls as N grows. For example at sigma = 0.1 and P/N = 0.0625 the accuracy is 1.000, 0.997, 0.985, 0.963 for N = 32, 64, 128, 256. The original `benchmark.py` capacity numbers mixed P/N and N; the "1.000 at P/N 0.03 to 0.06" statement holds for N = 32 and 64 only.
- **Accuracy at P/N 0.25 and 0.5 depends on N** (for example sigma = 0.1: 0.772, 0.716, 0.700, 0.670 at P/N 0.25), so it is not a single curve in P/N.
- **The noise levels tested do not test denoising.** A Gaussian phase error of 0.25 rad almost never pushes a BPSK phase past the pi/2 decision boundary, so the cue-only baseline (and the nearest-stored-pattern baseline) are 1.000 in every cell, and the recalled bit accuracy is **below** both whenever it is below 1. Here the memory adds crosstalk rather than error correction. Sigma = 0.1 and 0.25 give nearly the same curves for that reason. Round 2 (below) adds bit-flip and larger phase-noise cues and finds that recall denoises only at low load.
- At sigma = 0 (exact cue) accuracy still falls from P/N = 0.125, so stored patterns stop being stable fixed points near that load. The classic Hopfield capacity for random patterns is usually quoted as about 0.14 N; this benchmark did not test that figure directly, and its knee uses a 0.95 accuracy threshold at 25 steps.
- Null check: the random-pattern baseline, averaged over N, is 0.486 to 0.513 in every row, as expected for chance.

### 3. Reverse (round trips)

- Property loop over 500 seeds (random lengths 1 to 300 including odd lengths; random, all-zero, all-one, alternating, single-one and single-zero patterns; random Unicode text): codec exact at sigma = 0, wave round trip phase error exactly 0.0, text round trip exact, codec also exact at sigma = 0.1. **0 failures** in 500 seeds for each of the four checks.
- Adversarial cases are recorded in the report. Findings: `bits_to_wave` accepts empty input but `wave_to_bits` rejects it (`ValueError: wave phase: must be nonempty`), so an empty string cannot go through the wave codec; `bits_to_text` of 3 bits returns an empty string (truncates to whole bytes); 2-D or non-0/1 input is rejected; length-1 messages round trip. A `HolographicMemory(1)` cannot store anything useful: removing the diagonal leaves a zero weight, and recall returns bit 0.

## What changed relative to the original claims

| Original claim | Rigorous result |
| --- | --- |
| Scaling alpha about 1.4 to 1.5 | Not reproduced. On N >= 512, alpha = 2.27 to 2.34 (one thread) and 1.71 to 2.15 (default threads) across six runs; the 3-point N >= 128 fit was unstable. |
| Recall at N = 4096 about 0.3 to 0.4 s | One thread: medians 0.35, 0.37, 0.49 (heavy-load batch) and 0.48, 0.51, 0.50 s (second batch), so 2 of 6 runs fall in the range. Default threads: 0.045 to 0.095 s. Machine- and thread-dependent. |
| Recall at N = 16384 about 2.2 to 3.5 s | Not measured. Extrapolation: about 9 to 11 s (one thread) and about 0.9 to 1.3 s (default), probably underestimates. The claim lies between the two modes and is unverified. |
| Memory 256 MB at N = 4096, 4 GB at N = 16384 | Exact arithmetic (16 N^2 bytes); 256 MiB allocated in the N = 4096 run; 16384 not allocated. `store` peaks at about twice the weight size. |
| Recall 1.000 at P/N 0.03 to 0.06 | True at N = 32, 64; at sigma = 0.1 and P/N = 0.0625, 0.985 at N = 128 and 0.963 at N = 256. |
| Recall 0.753 at P/N 0.25; 0.663 at 0.50 | The 5-trial values came from N = 64 and N = 32. With 40 trials: 0.716 [0.686, 0.743] at N = 64 and 0.708 [0.678, 0.737] at N = 32 (sigma = 0.1); both old values fall just outside the new intervals, and accuracy varies with N. |
| Codec exact to sigma 0.25; 0.998 to 0.999 at sigma 0.5 | Exact at sigma 0 and 0.1 in all 500 property seeds; sigma 0.5 measured only in the original tool. |
| Wave round trip phase error 0.0, "Hello wave" exact | Reproduced (0 failures in 500 seeds). |

## Round 2: does recall denoise? Load-controlled scaling and N = 16384

Round 2 answers the gap round 1 flagged (Gaussian phase noise up to 0.25 rad never flips a bit, so it could not test denoising) and two scaling follow-ups. The plan, `benchmark_protocol_round2.json`, and its lock `benchmark_protocol_round2.lock` (hash `a06cd2e65d18d2eac28b57a1a74e8291ab5b81b7f31f28f7637685d8f8bcc27d`) were committed in `fed230c`, before `benchmark_round2.py` (commit `b5cc07f`) existed and before any round-2 experiment was run. Plan text: [round2_plan.md](round2_plan.md). The same limits as the round-1 lock apply: a git commitment, not an independent timestamp, and the author already knew the round-1 results. A zip bundle of the plans and hashes (with proposed Zenodo metadata) was prepared locally; nothing has been uploaded anywhere.

```sh
python benchmark_round2.py --quick                            # CI smoke, deterministic hash printed
python benchmark_round2.py --full --json report_round2.json   # manual; about 10 to 15 minutes on this shared box, longer under load
```

Raw reports of the two full runs described below are in `results/round2/full_run1.json` and `full_run2.json`. Both runs have the identical `deterministic_sha256` = `e10744690963837ca2efe8bfe11cb6d6dea5b5a25c50945461f68c49e56bb661`, which covers the protocol hash, seed, every denoising result, the complex64 agreement check and the scaling plan, and excludes timings, load and memory-availability decisions.

### Round 2a. Denoising

**Question.** If the cue is a stored pattern with some bits corrupted, does `HolographicMemory.recall` (25 steps) return something closer to the stored pattern than the cue itself?

**Design (as locked).** Target = first of P stored random BPSK patterns. Cue corruption: independent bit flips with probability q in {0.05, 0.10, 0.20, 0.30} (control q = 0), or Gaussian phase noise sigma in {0.9, 1.2} rad (expected flip fraction about 0.08 and 0.19; the realised fraction is in the report and matched, for example 0.08 and 0.19). N in {64, 128, 256}, P/N in {1/32, 1/16, 3/32, 1/8, 3/16, 1/4}, 50 trials per cell, 108 non-control cells. Baselines: cue only (slice the corrupted cue), nearest stored pattern (matched filter, a lookup that uses the stored list), random pattern. Primary endpoint: per-trial gain = recall bit accuracy minus cue bit accuracy, paired bootstrap 95% CI (2000 resamples). Verdict per cell: D (CI above 0), X (CI below 0), ~ (otherwise). No multiplicity correction. Control check (q = 0): cue accuracy is exactly 1.000 in every control cell, as it must be.

**Answer: recall denoises only at low load, and it does not beat the nearest-stored-pattern lookup anywhere.**

- At P/N <= 0.0625 (1/16) all 36 cells are D: recall brings bit accuracy to 0.94 to 1.000, even at q = 0.30 (cue accuracy about 0.70); the weakest D cell is phase noise sigma = 0.9, N = 256, P/N = 1/16 (0.939 vs 0.917).
- At P/N = 3/32 and 1/8 the picture is mixed (25 D, 7 X, 4 ~ of 36 cells).
- At P/N >= 3/16 **no cell denoises** (0 of 36); 30 are X (recall worse than the cue it started from) and 6 are ~. At P/N = 1/4 recall accuracy is 0.64 to 0.80 for every corruption level while the cue itself is 0.70 to 0.95.
- Overall: 61 D, 37 X, 10 ~ of 108 cells. The system-wide statement "recall denoises" is therefore false; the supported statement is "recall denoises in the cells marked D (low load) and degrades the cue in the cells marked X (high load)".
- Against the nearest-stored-pattern baseline, recall is never better (0 of 108 cells); it is worse in 79 cells and inconclusive in 29. The nearest-stored baseline is exactly 1.000 in 103 of 108 cells and 0.938 to 0.980 in the other five (all at N = 64, q = 0.30, P/N >= 1/16). This baseline needs the list of stored patterns, so it is a lookup rather than a competing analog dynamics; it shows that on these random, uncorrelated patterns a digital lookup solves the task that recall solves only at low load.
- Where is the transition? With bit flips, the largest P/N with verdict D is 1/8 for every N and q except N = 256, q = 0.05 where it is 3/32. The result at P/N = 1/8 is already weaker at larger N (for example q = 0.20: gain +0.119, +0.133, +0.049 at N = 64, 128, 256). That matches round 1, where exact cues lose accuracy from P/N about 0.1 to 0.19. Because the grid is coarse, the transition is only located between 1/8 and 3/16 for this setup.

**Verdict counts per corruption level** (18 cells each: 3 values of N x 6 values of P/N; D = denoises, X = degrades, ~ = inconclusive)

| cue corruption | D | X | ~ |
| --- | --- | --- | --- |
| bit flip q = 0.05 | 11 | 5 | 2 |
| bit flip q = 0.1 | 12 | 5 | 1 |
| bit flip q = 0.2 | 12 | 5 | 1 |
| bit flip q = 0.3 | 12 | 3 | 3 |
| phase noise sigma = 0.9 | 6 | 11 | 1 |
| phase noise sigma = 1.2 | 8 | 8 | 2 |
| all 108 cells | 61 | 37 | 10 |

**Verdict counts per load P/N** (18 cells each: 3 N x 6 corruption levels)

| P/N | D | X | ~ |
| --- | --- | --- | --- |
| 0.03125 | 18 | 0 | 0 |
| 0.0625 | 18 | 0 | 0 |
| 0.09375 | 14 | 2 | 2 |
| 0.125 | 11 | 5 | 2 |
| 0.1875 | 0 | 13 | 5 |
| 0.25 | 0 | 17 | 1 |

Each cell below reads: recall bit accuracy vs cue-only bit accuracy: gain [paired bootstrap 95% CI] verdict. 50 trials per cell.

**bit-flip probability q = 0.05**

| P/N | N = 64 | N = 128 | N = 256 |
| --- | --- | --- | --- |
| 0.03125 | 1.000 vs 0.957: +0.043 [+0.036, +0.050] D | 1.000 vs 0.955: +0.045 [+0.040, +0.050] D | 1.000 vs 0.951: +0.049 [+0.046, +0.053] D |
| 0.0625 | 1.000 vs 0.951: +0.049 [+0.043, +0.056] D | 1.000 vs 0.949: +0.051 [+0.045, +0.056] D | 1.000 vs 0.949: +0.051 [+0.047, +0.054] D |
| 0.09375 | 0.999 vs 0.949: +0.051 [+0.043, +0.058] D | 0.995 vs 0.950: +0.045 [+0.034, +0.052] D | 0.990 vs 0.952: +0.037 [+0.024, +0.048] D |
| 0.125 | 0.982 vs 0.952: +0.030 [+0.014, +0.043] D | 0.973 vs 0.954: +0.019 [+0.001, +0.035] D | 0.962 vs 0.951: +0.010 [-0.017, +0.035] ~ |
| 0.1875 | 0.941 vs 0.946: -0.005 [-0.032, +0.021] ~ | 0.863 vs 0.951: -0.087 [-0.123, -0.051] X | 0.799 vs 0.951: -0.152 [-0.187, -0.117] X |
| 0.25 | 0.774 vs 0.943: -0.168 [-0.198, -0.137] X | 0.730 vs 0.947: -0.217 [-0.242, -0.191] X | 0.706 vs 0.950: -0.244 [-0.265, -0.222] X |

**bit-flip probability q = 0.1**

| P/N | N = 64 | N = 128 | N = 256 |
| --- | --- | --- | --- |
| 0.03125 | 1.000 vs 0.892: +0.108 [+0.100, +0.117] D | 1.000 vs 0.901: +0.099 [+0.092, +0.106] D | 1.000 vs 0.892: +0.107 [+0.102, +0.113] D |
| 0.0625 | 1.000 vs 0.904: +0.096 [+0.087, +0.106] D | 0.999 vs 0.899: +0.099 [+0.092, +0.106] D | 0.998 vs 0.899: +0.099 [+0.093, +0.105] D |
| 0.09375 | 0.995 vs 0.890: +0.105 [+0.093, +0.117] D | 0.977 vs 0.894: +0.083 [+0.065, +0.099] D | 0.983 vs 0.903: +0.080 [+0.066, +0.093] D |
| 0.125 | 0.974 vs 0.909: +0.065 [+0.045, +0.082] D | 0.961 vs 0.905: +0.056 [+0.028, +0.078] D | 0.932 vs 0.898: +0.034 [+0.005, +0.060] D |
| 0.1875 | 0.927 vs 0.899: +0.028 [-0.000, +0.054] ~ | 0.838 vs 0.895: -0.058 [-0.092, -0.023] X | 0.793 vs 0.900: -0.107 [-0.141, -0.071] X |
| 0.25 | 0.797 vs 0.905: -0.107 [-0.144, -0.071] X | 0.755 vs 0.909: -0.154 [-0.179, -0.126] X | 0.679 vs 0.899: -0.220 [-0.237, -0.203] X |

**bit-flip probability q = 0.2**

| P/N | N = 64 | N = 128 | N = 256 |
| --- | --- | --- | --- |
| 0.03125 | 1.000 vs 0.806: +0.194 [+0.184, +0.205] D | 1.000 vs 0.801: +0.199 [+0.188, +0.210] D | 1.000 vs 0.800: +0.200 [+0.192, +0.208] D |
| 0.0625 | 1.000 vs 0.797: +0.203 [+0.189, +0.218] D | 0.998 vs 0.797: +0.201 [+0.192, +0.210] D | 0.992 vs 0.795: +0.197 [+0.189, +0.204] D |
| 0.09375 | 0.962 vs 0.795: +0.167 [+0.141, +0.189] D | 0.975 vs 0.809: +0.165 [+0.143, +0.186] D | 0.929 vs 0.794: +0.135 [+0.108, +0.160] D |
| 0.125 | 0.912 vs 0.793: +0.119 [+0.086, +0.149] D | 0.934 vs 0.801: +0.133 [+0.109, +0.156] D | 0.850 vs 0.801: +0.049 [+0.013, +0.082] D |
| 0.1875 | 0.824 vs 0.803: +0.021 [-0.014, +0.058] ~ | 0.743 vs 0.799: -0.056 [-0.086, -0.022] X | 0.701 vs 0.801: -0.099 [-0.118, -0.080] X |
| 0.25 | 0.741 vs 0.798: -0.057 [-0.093, -0.018] X | 0.713 vs 0.802: -0.089 [-0.116, -0.063] X | 0.673 vs 0.801: -0.127 [-0.144, -0.111] X |

**bit-flip probability q = 0.3**

| P/N | N = 64 | N = 128 | N = 256 |
| --- | --- | --- | --- |
| 0.03125 | 1.000 vs 0.695: +0.305 [+0.291, +0.318] D | 0.986 vs 0.697: +0.290 [+0.277, +0.302] D | 1.000 vs 0.694: +0.306 [+0.298, +0.313] D |
| 0.0625 | 0.950 vs 0.693: +0.257 [+0.228, +0.284] D | 0.969 vs 0.701: +0.268 [+0.243, +0.288] D | 0.977 vs 0.701: +0.276 [+0.257, +0.291] D |
| 0.09375 | 0.906 vs 0.692: +0.214 [+0.178, +0.247] D | 0.894 vs 0.697: +0.197 [+0.159, +0.229] D | 0.898 vs 0.704: +0.194 [+0.158, +0.229] D |
| 0.125 | 0.837 vs 0.703: +0.133 [+0.091, +0.175] D | 0.829 vs 0.708: +0.121 [+0.080, +0.163] D | 0.778 vs 0.700: +0.078 [+0.043, +0.114] D |
| 0.1875 | 0.733 vs 0.703: +0.030 [-0.009, +0.072] ~ | 0.717 vs 0.699: +0.018 [-0.010, +0.044] ~ | 0.655 vs 0.698: -0.043 [-0.061, -0.025] X |
| 0.25 | 0.688 vs 0.699: -0.011 [-0.046, +0.025] ~ | 0.638 vs 0.715: -0.077 [-0.102, -0.054] X | 0.638 vs 0.698: -0.061 [-0.079, -0.043] X |

**phase-noise sigma (rad) = 0.9**

| P/N | N = 64 | N = 128 | N = 256 |
| --- | --- | --- | --- |
| 0.03125 | 1.000 vs 0.909: +0.091 [+0.082, +0.099] D | 1.000 vs 0.924: +0.076 [+0.070, +0.082] D | 1.000 vs 0.920: +0.080 [+0.076, +0.084] D |
| 0.0625 | 0.982 vs 0.925: +0.057 [+0.042, +0.069] D | 0.959 vs 0.911: +0.048 [+0.026, +0.066] D | 0.939 vs 0.917: +0.022 [+0.006, +0.038] D |
| 0.09375 | 0.931 vs 0.922: +0.009 [-0.015, +0.034] ~ | 0.854 vs 0.920: -0.066 [-0.092, -0.039] X | 0.782 vs 0.916: -0.134 [-0.157, -0.111] X |
| 0.125 | 0.801 vs 0.925: -0.124 [-0.157, -0.091] X | 0.727 vs 0.925: -0.198 [-0.224, -0.169] X | 0.721 vs 0.919: -0.198 [-0.217, -0.179] X |
| 0.1875 | 0.752 vs 0.928: -0.176 [-0.209, -0.143] X | 0.708 vs 0.915: -0.206 [-0.230, -0.185] X | 0.673 vs 0.919: -0.246 [-0.261, -0.232] X |
| 0.25 | 0.715 vs 0.911: -0.196 [-0.223, -0.167] X | 0.665 vs 0.917: -0.252 [-0.271, -0.233] X | 0.652 vs 0.915: -0.263 [-0.277, -0.249] X |

**phase-noise sigma (rad) = 1.2**

| P/N | N = 64 | N = 128 | N = 256 |
| --- | --- | --- | --- |
| 0.03125 | 1.000 vs 0.815: +0.185 [+0.171, +0.198] D | 1.000 vs 0.812: +0.188 [+0.179, +0.197] D | 1.000 vs 0.814: +0.186 [+0.178, +0.193] D |
| 0.0625 | 0.998 vs 0.807: +0.190 [+0.176, +0.205] D | 0.970 vs 0.811: +0.158 [+0.141, +0.176] D | 0.951 vs 0.808: +0.144 [+0.122, +0.162] D |
| 0.09375 | 0.895 vs 0.816: +0.079 [+0.047, +0.110] D | 0.872 vs 0.810: +0.062 [+0.034, +0.088] D | 0.799 vs 0.808: -0.008 [-0.031, +0.014] ~ |
| 0.125 | 0.832 vs 0.802: +0.030 [-0.001, +0.060] ~ | 0.756 vs 0.808: -0.052 [-0.080, -0.025] X | 0.724 vs 0.809: -0.085 [-0.107, -0.061] X |
| 0.1875 | 0.716 vs 0.795: -0.078 [-0.104, -0.050] X | 0.683 vs 0.817: -0.134 [-0.159, -0.112] X | 0.679 vs 0.809: -0.131 [-0.152, -0.108] X |
| 0.25 | 0.698 vs 0.799: -0.101 [-0.138, -0.065] X | 0.656 vs 0.813: -0.158 [-0.179, -0.136] X | 0.645 vs 0.811: -0.166 [-0.181, -0.152] X |

### Round 2a. What is and is not established

- Established (for this simulation, random BPSK patterns, 25 steps, 50 trials per cell): the D / X / ~ table above, including that recall is worse than doing nothing at P/N >= 3/16 and never better than a nearest-stored lookup.
- Not established: why recall gets worse at high load (cross-talk, spurious fixed points and global-phase mirror states are all plausible; none was tested, and the mirror is not folded in the accuracy). Behaviour for correlated or structured patterns, other step counts, other N, or P/N between the grid points. Anything about hardware.
- The 108 cell verdicts use per-cell 95% intervals without correction. Cells that are close to a boundary (CI lower bound near 0) should not be over-read; the large effects at P/N <= 1/16 and >= 3/16 do not depend on this.
- The Wilson intervals for exact-recall and nearest-stored retrieval fractions are in the JSON report (`exact_recall_ci_wilson`, `nearest_stored_retrieval_ci_wilson`). Bit accuracy uses bootstrap CIs because bits within a trial are not independent.

### Round 2b. Load-controlled scaling and N = 16384

Every timing batch (one thread mode, one dtype, one N, all repeats) records `os.getloadavg()` before and after, and the **external busy cores** during the batch (system CPU busy seconds from `/proc/stat` minus the worker's own CPU seconds, divided by wall time). The 1-minute load average is recorded but is not the gate, because it includes the benchmark's own threads. A batch with external load above 0.5 cores is discarded and rerun, up to 4 attempts; a batch that never passes is flagged and excluded from fits, slopes and extrapolations. Discarded attempts are stored in the report. The box is shared, and another job kept about one core busy for part of the session, so some batches were excluded; every exclusion is listed below.

**complex64 at N = 16384.** A complex64 coupling matrix (8N^2 bytes, 2 GiB at N = 16384, peak about 2.06 GiB) built in 64 MiB row chunks from 8 random patterns, and the same 25-step recall loop in complex64, was timed when the protocol's memory rule allowed (peak <= 4 GiB and <= half of MemAvailable at that moment). It is a different dtype than the original complex128 claim. Dense matvec time does not depend on the matrix values, which is why 8 stored patterns suffice for timing. A check at N = 256 and 512 with P = 4 and 8 (20 seeds each) found the complex64 recall bits identical to the complex128 recall bits in 80 of 80 cases.

**complex128 recall time, median seconds over repeats (load-controlled batches only; `excluded` = no attempt met the load threshold, or not run)**

| N | 1 thread, run 1 | 1 thread, run 2 | default, run 1 | default, run 2 |
| --- | --- | --- | --- | --- |
| 64 | 0.000386 | 0.000389 | 0.000346 | 0.000431 |
| 128 | 0.000521 | 0.000577 | excluded | 0.000546 |
| 192 | 0.000674 | 0.000790 | 0.000565 | 0.000671 |
| 256 | 0.000939 | 0.001108 | excluded | 0.000767 |
| 384 | 0.002486 | 0.002671 | 0.001138 | 0.001114 |
| 512 | 0.004176 | 0.004872 | 0.001381 | 0.001350 |
| 768 | 0.009078 | 0.010135 | 0.002370 | 0.002835 |
| 1024 | 0.015727 | 0.016797 | 0.003035 | 0.004885 |
| 1536 | 0.036239 | 0.037717 | 0.013150 | 0.009221 |
| 2048 | 0.060615 | 0.064152 | 0.018847 | 0.011607 |
| 3072 | 0.165737 | 0.154048 | 0.040632 | 0.027459 |
| 4096 | excluded | 0.469098 | 0.078340 | 0.047587 |
| 6144 | 1.250261 | excluded | 0.237058 | 0.176756 |
| 8192 | 2.309112 | 2.303511 | 0.304380 | 0.321186 |

**complex128 fit on N >= 512 (fixed a priori; load-controlled points only)**

| thread mode | run | points in fit | alpha | 95% bootstrap CI | extrapolated N=16384 (s) [interval] | hold-out: predicted / measured at 4096, 8192 |
| --- | --- | --- | --- | --- | --- | --- |
| one BLAS thread | 1 | 8 | 2.309 | [2.295, 2.321] | 9.83 [9.58, 10.12] | 8192: 0.39 |
| one BLAS thread | 2 | 8 | 2.226 | [2.212, 2.247] | 8.51 [8.24, 9.14] | 4096: 0.49, 8192: 0.37 |
| default threads | 1 | 9 | 2.072 | [2.046, 2.117] | 1.42 [1.37, 1.54] | 4096: 0.94, 8192: 0.98 |
| default threads | 2 | 9 | 1.921 | [1.892, 1.953] | 0.90 [0.86, 0.98] | 4096: 0.84, 8192: 0.37 |

**complex64 recall time at large N (measured directly; 25 steps; 8N^2-byte matrix)**

| thread mode | run | N | median (s) | IQR (s) | 95% CI of median (s) | GFLOP/s | peak memory (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| one BLAS thread | 1 | 4096 | excluded (load) | | | | |
| one BLAS thread | 1 | 8192 | excluded (load) | | | | |
| one BLAS thread | 1 | 16384 | 4.6687 | 0.0994 | [4.5191, 4.7424] | 11.5 | 2.06 |
| one BLAS thread | 2 | 4096 | 0.1579 | 0.0075 | [0.1486, 0.1651] | 21.2 | 0.19 |
| one BLAS thread | 2 | 8192 | 1.1477 | 0.0161 | [1.1406, 1.1602] | 11.7 | 0.56 |
| one BLAS thread | 2 | 16384 | 4.7971 | 0.2422 | [4.5862, 4.9877] | 11.2 | 2.06 |
| default threads | 1 | 4096 | 0.0227 | 0.0133 | [0.0214, 0.0440] | 148.0 | 0.19 |
| default threads | 1 | 8192 | 0.1581 | 0.0155 | [0.1395, 0.1702] | 84.9 | 0.56 |
| default threads | 1 | 16384 | 0.6274 | 0.0116 | [0.6075, 0.6819] | 85.6 | 2.06 |
| default threads | 2 | 4096 | 0.0227 | 0.0012 | [0.0212, 0.0255] | 148.1 | 0.19 |
| default threads | 2 | 8192 | 0.1471 | 0.0062 | [0.1458, 0.1654] | 91.2 | 0.56 |
| default threads | 2 | 16384 | 0.6293 | 0.0055 | [0.6255, 0.6942] | 85.3 | 2.06 |

**complex128 at N = 16384: ESTIMATE, NOT MEASURED** (complex64 time at 16384 x measured complex128/complex64 ratio at the reference N; same thread mode; assumes the ratio at 16384 equals the ratio at the reference N, which is untested)

| thread mode | run | reference N | complex128/complex64 ratio | estimated complex128 seconds at 16384 |
| --- | --- | --- | --- | --- |
| one BLAS thread | 1 | none | no load-controlled pair at 4096 or 8192 | not computable |
| one BLAS thread | 2 | 8192 (primary) | 2.01 | 9.63 |
| one BLAS thread | 2 | 4096 | 2.97 | 14.25 |
| default threads | 1 | 8192 (primary) | 1.93 | 1.21 |
| default threads | 1 | 4096 | 3.46 | 2.17 |
| default threads | 2 | 8192 (primary) | 2.18 | 1.37 |
| default threads | 2 | 4096 | 2.10 | 1.32 |

**Load control** (gate: external busy cores <= 0.5, up to 4 attempts)

| thread mode | run | batches | accepted first try | accepted after retry | never accepted (excluded) |
| --- | --- | --- | --- | --- | --- |
| one BLAS thread | 1 | 17 | 10 | 4 | 3: complex128 N=4096, complex64 N=4096, complex64 N=8192 |
| one BLAS thread | 2 | 17 | 13 | 3 | 1: complex128 N=6144 |
| default threads | 1 | 17 | 12 | 3 | 2: complex128 N=128, complex128 N=256 |
| default threads | 2 | 17 | 16 | 1 | 0 |

Reading these tables:

- **N = 16384 was measured directly, with complex64 only.** Median 4.67 and 4.80 s with one BLAS thread, 0.627 and 0.629 s with default threads (8 cores). The complex128 time at 16384 was **not measured** (it needs about 8 GiB); the only complex128 numbers there are the two labelled estimates (extrapolated fit, and complex64 time times the complex128/complex64 ratio at 8192 or 4096). The ratio itself moved between 1.9 and 3.5 across runs and reference sizes, so the estimates span a factor of about 1.8 (default threads, 1.2 to 2.2 s) and about 1.5 (one thread, 9.6 to 14.3 s from a single usable run). The fit extrapolations (one thread 8.5 to 9.8 s, default 0.9 to 1.4 s) are of similar size to the ratio estimates, and the hold-out checks say the fit underestimates for one thread (predicted/measured 0.37 to 0.49).
- **The original claim (recall at N = 16384 in 2.2 to 3.5 s) is still not reproduced.** It sits between the default-thread estimates (0.9 to 2.2 s) and the one-thread estimates (8.5 to 14.3 s) and is unverified for complex128. The directly measured complex64 times (0.63 s, 4.7 s) are for a different dtype.
- **Alpha, N >= 512, load-controlled points only:** one thread 2.309 and 2.226, default threads 2.072 and 1.921 across the two runs (within-run 95% bootstrap CIs about 0.03 to 0.05 wide). Across the six round-1 runs and the four round-2 runs the one-thread range is 2.23 to 2.34 and the default-thread range 1.71 to 2.15. The original claim of 1.4 to 1.5 is not reproduced, and the within-run CI still understates run-to-run variation (for example default threads at N = 4096: 0.078 s in run 1 and 0.048 s in run 2, both load-gated).
- **Load control does not make timings reproducible.** It removes batches that overlapped with other CPU-heavy jobs; it cannot see cache or memory-bandwidth interference, VM scheduling or frequency changes. Treat the CIs as sampling noise within one run.
- **The step above N^2 reappeared.** One thread: complex128 3072 -> 4096 is 0.154 -> 0.469 s in run 2 (local slope about 3.9, as in round 1), and complex64 4096 -> 8192 is 0.158 -> 1.148 s (slope 2.9) before returning to about 2 at 8192 -> 16384. Working-set size is a plausible cause; it was not tested here.

### Round 2c. Updated claims versus results

| Original claim | Round-1 result | Round-2 result |
| --- | --- | --- |
| Scaling alpha about 1.4 to 1.5 | Not reproduced (alpha 2.27 to 2.34 one thread, 1.71 to 2.15 default) | Not reproduced (2.23 to 2.31 one thread, 1.92 to 2.07 default, load-controlled) |
| Recall at N = 4096 about 0.3 to 0.4 s | One thread 0.35 to 0.51 s, default 0.045 to 0.095 s | One thread 0.469 s (one usable batch), default 0.048 to 0.078 s: outside the range in the usable batches |
| Recall at N = 16384 about 2.2 to 3.5 s | Not measured; extrapolation 9 to 11 s one thread, about 1 s default | complex64 measured (0.63 s default, 4.7 to 4.8 s one thread); complex128 still not measured, estimates 1.2 to 2.2 s default and 9.6 to 14.3 s one thread |
| Memory at N = 16384 is 4 GiB | Exact arithmetic | Unchanged for complex128; complex64 needs 2 GiB and was allocated and used |
| "Recall denoises" (implied by the capacity story) | Not testable with sigma <= 0.25 rad | True only at low load (P/N <= 1/16: 36 of 36 cells); at P/N >= 3/16 recall is worse than the cue (30 of 36 cells) and recall never beats a nearest-stored lookup |

## Round 3: why does recall degrade, correlated patterns, multiplicity, cache, large complex128

Round 3 follows a locked plan, `benchmark_protocol_round3.json` and `benchmark_protocol_round3.lock` (hash `b6365f500ea5a6c1e60c8eb4db1200c6c5980e027dd409e3895a1676ff5637d4`), committed in `bf9b92f`, before `benchmark_round3.py` (`98300bc`) existed and before any round-3 experiment ran. Plan text, with the support/reject rule of every hypothesis: [round3_plan.md](round3_plan.md). The raw reports of the two full runs are in `results/round3/`. The same limits as before apply: the lock is a git commitment, not an independent timestamp, and the author knew the round-1 and round-2 results (and the cross-talk and Hadamard algebra) when writing it. Nothing has been uploaded anywhere.

```sh
python benchmark_round3.py --quick                            # CI smoke
python benchmark_round3.py --full --json report_round3.json   # manual; 6 to 30 minutes on this shared box
python experiments/correlated_collapse_check.py               # post hoc check, seconds
```

Both full runs have the identical `deterministic_sha256` = `a0f0b96e32855a4c2f926dd208d6154d606ab718a3abcf66ee0e25995e71fad7` (mechanism, correlated-pattern and multiplicity results, plan). Timings differ between runs and are not in the hash.

**Deviations and additions (all disclosed).** (1) No change to the locked plan. (2) `benchmark_round2.bench_denoising` gained an optional `trial_sink` argument so the round-2 cells could be re-analysed; the returned results are byte-identical (tested) and the round-2 hash is unaffected. (3) In the quick profile the correlated-pattern rule uses a required drop of 1 cell instead of 6; quick mode is a smoke test only. (4) Multiple-testing families for the mechanism experiment were formed per pattern kind and per step (the plan named only `mechanism_t25`): random patterns at steps 25 and 100, Hadamard patterns at steps 25 and 100. (5) Marked "exploratory" or "post hoc" below: the step-1 tie fraction, the read of aggregate L2 in the default-thread cache result, the accuracy-versus-step peak, and the common-pattern collapse check. None of these has a pre-declared rule.

### Round 3a. Why does recall get worse than the cue at high load?

**The most important finding: at high load the stored patterns are not stable fixed points of the dynamics. With random patterns, recall drifts away from the stored pattern into states that are not near any stored pattern, and this happens even for an exact, uncorrupted cue. It is caused by cross-talk between the patterns: with exactly orthogonal patterns the degradation disappears.** Round 2 used 25 steps and therefore understated it: accuracy keeps falling between steps 25 and 100.

x-cells (random patterns, q > 0, gain over the cue at step 25 significantly negative after Benjamini-Hochberg, family of 24 cells): 8 cells: N = 64: q 0.1, P/N 0.25; N = 128: q 0.1 at P/N 0.1875 and 0.25, q 0.3 at 0.25; N = 256: q 0.1 at 0.1875 and 0.25, q 0.3 at 0.1875 and 0.25. All hypothesis verdicts below pool these 8 cells x 50 trials = 400 trials.

| Hypothesis | Pre-declared rule (short) | Result | Verdict |
| --- | --- | --- | --- |
| (a) cross-talk from other stored patterns | one-step error of an exact stored pattern matches the Gaussian prediction Q(sqrt((N-1)/(P-1))) in >= 75% of testable cells AND orthogonal Hadamard patterns remove the degradation in >= 80% of x-cells with >= +0.05 accuracy | a1: 8 of 8 testable cells agree (for example N = 256, P/N = 1/4: measured 0.0223, 95% CI [0.0194, 0.0252], predicted 0.0221). a2: the Hadamard one-step error is exactly 0 in every q = 0 cell; in 8 of 8 x-cells the Hadamard cell does not degrade (all 24 Hadamard q > 0 cells denoise at steps 25 and 100); mean accuracy at step 25 is +0.230 higher than with random patterns | **supported** |
| (b) spurious states | pooled fraction of final states (step 100) farther than tau = 0.10 from every stored pattern and its negation, Wilson lower bound >= 0.50 | 397 of 400, Wilson 95% [0.978, 0.997]; by tau: 1.000 (0.02), 0.998 (0.05), 0.993 (0.10), 0.873 (0.20) | **supported** |
| (c) sign mirror | pooled fraction near a negated stored pattern, Wilson lower bound >= 0.10 (rejected if upper <= 0.05) | 0 of 400, Wilson 95% [0.000, 0.0095] | **rejected** |
| (d) truncation: 25 steps is not converged | accuracy changes by >= 0.02 between steps 25 and 100 and >= 25% of trials change bits | mean accuracy at step 100 minus step 25 = -0.088, 95% CI [-0.099, -0.076]; 93.5% of trials have different bits at 25 and 100; all 8 x-cells still degrade at step 100 | **supported** (as a statement about convergence; the direction is the opposite of "more steps would fix it": more steps make it worse) |
| (d) oscillation: period-2 cycles | Wilson lower bound >= 0.10 (rejected if upper <= 0.05) | 0 of 400 period-2; 344 fixed points and 56 still changing bits in steps 98 to 100 (cause not examined: longer cycles or tie-bit flicker are candidates) | **rejected** |

**Random patterns, N = 256** (mean bit accuracy vs the target at step t; last column: trials of 50 ending retrieved-target / spurious at step 100, tau = 0.10)

| P/N | q | cue | step 1 | step 5 | step 25 | step 100 | target / spurious @100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0625 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.984 | 48 / 2 |
| 0.125 | 0.0 | 1.000 | 0.998 | 0.996 | 0.980 | 0.674 | 1 / 49 |
| 0.1875 | 0.0 | 1.000 | 0.991 | 0.964 | 0.840 | 0.627 | 0 / 50 |
| 0.25 | 0.0 | 1.000 | 0.978 | 0.896 | 0.705 | 0.621 | 0 / 50 |
| 0.0625 | 0.1 | 0.897 | 1.000 | 1.000 | 1.000 | 0.990 | 48 / 2 |
| 0.125 | 0.1 | 0.896 | 0.988 | 0.993 | 0.920 | 0.663 | 0 / 50 |
| 0.1875 | 0.1 | 0.903 | 0.970 | 0.941 | 0.782 | 0.634 | 0 / 50 |
| 0.25 | 0.1 | 0.902 | 0.944 | 0.857 | 0.697 | 0.623 | 0 / 50 |
| 0.0625 | 0.3 | 0.704 | 0.951 | 0.997 | 0.983 | 0.912 | 32 / 18 |
| 0.125 | 0.3 | 0.704 | 0.875 | 0.894 | 0.778 | 0.655 | 0 / 50 |
| 0.1875 | 0.3 | 0.694 | 0.818 | 0.770 | 0.671 | 0.627 | 0 / 50 |
| 0.25 | 0.3 | 0.701 | 0.788 | 0.719 | 0.623 | 0.592 | 0 / 50 |

**Orthogonal (Hadamard) patterns, N = 256** (mean bit accuracy vs the target at step t; last column: trials of 50 ending retrieved-target / spurious at step 100, tau = 0.10)

| P/N | q | cue | step 1 | step 5 | step 25 | step 100 | target / spurious @100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0625 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 50 / 0 |
| 0.125 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 50 / 0 |
| 0.1875 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 50 / 0 |
| 0.25 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 50 / 0 |
| 0.0625 | 0.1 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 50 / 0 |
| 0.125 | 0.1 | 0.899 | 1.000 | 1.000 | 1.000 | 1.000 | 50 / 0 |
| 0.1875 | 0.1 | 0.900 | 0.997 | 1.000 | 1.000 | 1.000 | 50 / 0 |
| 0.25 | 0.1 | 0.903 | 0.991 | 1.000 | 1.000 | 1.000 | 50 / 0 |
| 0.0625 | 0.3 | 0.698 | 0.957 | 1.000 | 1.000 | 1.000 | 50 / 0 |
| 0.125 | 0.3 | 0.700 | 0.877 | 0.984 | 0.987 | 0.980 | 47 / 3 |
| 0.1875 | 0.3 | 0.701 | 0.811 | 0.926 | 0.932 | 0.928 | 35 / 15 |
| 0.25 | 0.3 | 0.702 | 0.779 | 0.864 | 0.884 | 0.874 | 23 / 27 |

**Gain over the cue, Benjamini-Hochberg verdicts (24 q > 0 cells per pattern kind; D = denoises, X = degrades, ~ = inconclusive)**

| patterns | step | D | X | ~ |
| --- | --- | --- | --- | --- |
| random | 25 | 12 | 8 | 4 |
| random | 100 | 7 | 15 | 2 |
| hadamard | 25 | 24 | 0 | 0 |
| hadamard | 100 | 24 | 0 | 0 |

**Exact cue (q = 0), random patterns: does a stored pattern stay put?** (mean bit accuracy vs the stored target; trials of 50 ending near the target at step 100)

| N | P/N | step 1 | step 25 | step 100 | near target @100 (tau 0.10) |
| --- | --- | --- | --- | --- | --- |
| 64 | 0.0625 | 1.0000 | 1.000 | 0.997 | 49 |
| 64 | 0.125 | 0.9991 | 0.992 | 0.860 | 23 |
| 64 | 0.1875 | 0.9909 | 0.944 | 0.739 | 2 |
| 64 | 0.25 | 0.9766 | 0.832 | 0.718 | 2 |
| 128 | 0.0625 | 1.0000 | 1.000 | 0.985 | 46 |
| 128 | 0.125 | 0.9977 | 0.982 | 0.739 | 3 |
| 128 | 0.1875 | 0.9900 | 0.910 | 0.677 | 0 |
| 128 | 0.25 | 0.9784 | 0.792 | 0.645 | 0 |
| 256 | 0.0625 | 1.0000 | 1.000 | 0.984 | 48 |
| 256 | 0.125 | 0.9984 | 0.980 | 0.674 | 1 |
| 256 | 0.1875 | 0.9911 | 0.840 | 0.627 | 0 |
| 256 | 0.25 | 0.9777 | 0.705 | 0.621 | 0 |

Reading the mechanism results:

- **Cross-talk is the cause, in the sense of these two tests.** The measured one-step error of an exact stored pattern tracks the classical Gaussian cross-talk prediction in all eight cells where it is measurable (the predictions at P/N = 1/16, and at N = 64 with P/N = 1/8, are too small to measure with 50 trials and are not counted). Replacing the random patterns by exactly orthogonal ones removes the cross-talk term by construction and removes the degradation: Hadamard recall at q = 0 and q = 0.1 is 1.000 up to P/N = 1/4, and at q = 0.3 it still improves on the cue in every cell (0.87 to 1.00 vs 0.70). The residual loss at q = 0.3, P/N = 3/16 and 1/4 with Hadamard patterns (spurious fraction 15/50 and 27/50) comes from the corrupted cue itself, not from stored-pattern interference. This does not show that cross-talk is the only contributor; it shows that removing it is sufficient in this setup.
- **The failure is convergence to spurious states, not mirror states.** In the x-cells 99.3% of final states are farther than 0.10 from every stored pattern and its negation. No final state was near a negated pattern. The spurious states were not characterised further (mixtures of stored patterns, spin-glass-like states and other possibilities are all untested); the 87% figure at tau = 0.20 shows that part of them are within 0.2 of a stored pattern.
- **Dynamics.** At P/N >= 1/8 even an exact cue loses the pattern: at N = 256, P/N = 1/8, accuracy is 0.998 at step 1, 0.980 at step 25 and 0.674 at step 100 (1 of 50 trials still near the target at step 100). The one-step error is only 0.2% to 2%, so the failure is a slow drift (an avalanche of small errors), not an immediate flip. At P/N = 1/16, 48 of 50 exact cues at N = 256 stay near the target at step 100 and 2 do not. The round-2 verdicts at 25 steps are therefore not converged values: with BH correction the random-pattern cells split 12 D / 8 X / 4 ~ at step 25 and 7 D / 15 X / 2 ~ at step 100. Hadamard cells stay 24 D / 0 / 0 at both steps.
- **Exploratory, not preregistered: early steps help.** In the random-pattern cells the accuracy peaks at step 2 to 5, then falls (for example N = 256, P/N = 1/8, q = 0.3: 0.875, 0.898, 0.894, 0.858, 0.778 at steps 1, 2, 5, 10, 25 against a cue of 0.704). Stopping after a few steps would denoise in some cells where 25 steps degrades. No test was run on this; the per-step means are in the JSON.
- **Exploratory: ties.** About 0.0% to 1.1% of local fields are exactly zero at step 1 (the sum of coupling terms cancels). Because the BPSK couplings are real only up to rounding (imaginary parts below 1e-12, tested), such ties are decided by rounding noise, which makes these dynamics synchronous +-1 Hopfield dynamics with noise-broken ties. This was not shown to matter for any verdict.

### Round 3b. Correlated patterns

Patterns with pairwise correlation rho are built from a common random pattern c: each bit equals c with probability (1 + sqrt(rho))/2, so any two patterns have expected correlation exactly rho (realised mean pairwise overlaps 0.001, 0.099, 0.298). rho = 0 is a fresh-seed replication of round 2: for q = 0.1 and 0.3 (36 cells), 35 of 36 raw verdicts equal the round-2 verdicts; the one difference is N = 128, q = 0.3, P/N = 1/4 (round 2: degrades; round 3: inconclusive).

**Correlated patterns: verdicts for q > 0 (36 cells per rho; raw and Benjamini-Hochberg give identical counts)**

| rho | D | X | ~ |
| --- | --- | --- | --- |
| 0.0 | 24 | 7 | 5 |
| 0.1 | 12 | 20 | 4 |
| 0.3 | 20 | 15 | 1 |

**Capacity with an exact cue (q = 0): knee = smallest tested P/N with mean bit accuracy < 0.95**

| N | rho = 0 | rho = 0.1 | rho = 0.3 |
| --- | --- | --- | --- |
| 64 | 0.1875 | 0.125 | 0.0625 |
| 128 | 0.1875 | 0.09375 | 0.03125 |
| 256 | 0.1875 | 0.0625 | 0.03125 |

**Mean over all 18 cells per rho (N x P/N), exact cue and q = 0.1, 0.3**

| rho | q | recall bit accuracy | exact-recall fraction | cue-only accuracy | nearest-stored accuracy |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.0 | 0.940 | 0.671 | 1.000 | 1.000 |
| 0.0 | 0.1 | 0.918 | 0.627 | 0.899 | 1.000 |
| 0.0 | 0.3 | 0.834 | 0.410 | 0.700 | 0.987 |
| 0.1 | 0.0 | 0.804 | 0.367 | 1.000 | 1.000 |
| 0.1 | 0.1 | 0.792 | 0.340 | 0.900 | 1.000 |
| 0.1 | 0.3 | 0.761 | 0.234 | 0.703 | 0.990 |
| 0.3 | 0.0 | 0.807 | 0.107 | 1.000 | 1.000 |
| 0.3 | 0.1 | 0.804 | 0.098 | 0.899 | 1.000 |
| 0.3 | 0.3 | 0.800 | 0.074 | 0.703 | 0.990 |

**Pre-declared hypothesis (e) (correlation shrinks the denoising region: supported if the BH "denoises" count at rho = 0.3 is at least 6 below rho = 0): partial / inconclusive by the rule.** The counts are 24 (rho = 0), 12 (rho = 0.1), 20 (rho = 0.3): non-monotonic, and 20 is only 4 below 24. The gain-over-cue measure is misleading here, as the next paragraph shows, so the capacity numbers are the informative result.

**Correlation strongly reduces capacity.** With an exact cue the knee moves from P/N = 3/16 (rho = 0, every N) to 1/16 to 1/8 (rho = 0.1) and 1/32 to 1/16 (rho = 0.3). The mean exact-recall fraction over the 18 cells falls from 0.671 to 0.367 to 0.107.

**Post hoc explanation (not preregistered; found by reading the table after the run).** For rho = 0.1 and 0.3 the recall accuracy in the failing cells sits at (1 + sqrt(rho))/2 = 0.658 and 0.774, which is the agreement between any stored pattern and the common pattern c. The check script below confirms it: for rho = 0.3 and N >= 128 the final state is within 0.10 of c in 46 to 50 of 50 trials and within 0.10 of the target in 0 or 1 trial; for rho = 0.1 the same happens at larger P/N. So with correlated patterns, recall collapses onto the shared component, which is a spurious attractor close to every stored pattern. That is why "gain over the cue" looks positive at q = 0.3 (cue accuracy 0.70 < 0.774) while the target is not retrieved at all (exact-recall fraction 0.074 at rho = 0.3, q = 0.3). Confirmed in this single post hoc check only for exact cues, 25 steps, these N and P/N.

**Post hoc collapse check** (`experiments/correlated_collapse_check.py`, exact cue, 25 steps, 50 trials per row; written after the round-3 results were seen)

| rho | N | P/N | predicted agreement with c | recall accuracy vs target | final within 0.10 of c | final within 0.10 of target |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 64 | 0.0625 | 0.6581 | 1.0 | 0 / 50 | 50 / 50 |
| 0.1 | 64 | 0.125 | 0.6581 | 0.93 | 0 / 50 | 34 / 50 |
| 0.1 | 64 | 0.25 | 0.6581 | 0.6978 | 20 / 50 | 2 / 50 |
| 0.1 | 128 | 0.0625 | 0.6581 | 0.9464 | 0 / 50 | 38 / 50 |
| 0.1 | 128 | 0.125 | 0.6581 | 0.7314 | 19 / 50 | 6 / 50 |
| 0.1 | 128 | 0.25 | 0.6581 | 0.6617 | 50 / 50 | 0 / 50 |
| 0.1 | 256 | 0.0625 | 0.6581 | 0.6798 | 19 / 50 | 0 / 50 |
| 0.1 | 256 | 0.125 | 0.6581 | 0.6552 | 50 / 50 | 0 / 50 |
| 0.1 | 256 | 0.25 | 0.6581 | 0.658 | 50 / 50 | 0 / 50 |
| 0.3 | 64 | 0.0625 | 0.7739 | 0.9019 | 15 / 50 | 26 / 50 |
| 0.3 | 64 | 0.125 | 0.7739 | 0.7791 | 46 / 50 | 0 / 50 |
| 0.3 | 64 | 0.25 | 0.7739 | 0.7744 | 50 / 50 | 1 / 50 |
| 0.3 | 128 | 0.0625 | 0.7739 | 0.7652 | 49 / 50 | 0 / 50 |
| 0.3 | 128 | 0.125 | 0.7739 | 0.7736 | 50 / 50 | 0 / 50 |
| 0.3 | 128 | 0.25 | 0.7739 | 0.7728 | 50 / 50 | 0 / 50 |
| 0.3 | 256 | 0.0625 | 0.7739 | 0.772 | 50 / 50 | 0 / 50 |
| 0.3 | 256 | 0.125 | 0.7739 | 0.7741 | 50 / 50 | 0 / 50 |
| 0.3 | 256 | 0.25 | 0.7739 | 0.783 | 50 / 50 | 0 / 50 |

### Round 3c. Multiplicity: how many round-2 verdicts survive?

The round-2 denoising profile was rerun with the same seed and per-trial output. **All 126 stored round-2 cells (108 plus 18 controls) reproduced exactly.** For each of the 108 non-control cells a two-sided sign-flip permutation test (100000 permutations, add-one p-values; assumes the per-trial gain is symmetric about 0 under the null) was computed, in two families: gain over the cue and recall minus nearest-stored.

| Family (108 cells) | Raw bootstrap-CI verdicts D / X / ~ | Holm-Bonferroni (alpha 0.05) | Benjamini-Hochberg (alpha 0.05) |
| --- | --- | --- | --- |
| recall vs cue | 61 / 37 / 10 | 57 / 36 / 15 | 61 / 37 / 10 |
| recall vs nearest stored (D = recall better) | 0 / 79 / 29 | 0 / 72 / 36 | 0 / 78 / 30 |

- Against the cue, Benjamini-Hochberg keeps all 98 significant verdicts; Holm keeps 93 (57 of 61 D, 36 of 37 X) and turns 5 into "inconclusive" (raw sign-flip p between 0.0057 and 0.027). No verdict reverses sign and no new one appears.
- Against the nearest-stored lookup, Holm keeps 72 of 79 "recall worse" and BH keeps 78; "recall better" is 0 in every version. The statement "recall is never better than a nearest-stored lookup" survives correction; the counts of "worse" cells shrink slightly.
- The five cells Holm drops (against the cue) are N = 64, q = 0.2, P/N = 1/4 (X); N = 128, q = 0.05, P/N = 1/8 (D); N = 256, q = 0.1 and 0.2, P/N = 1/8 (D); N = 256, sigma = 0.9, P/N = 1/16 (D). All sit at the load boundary, as expected.
- The bootstrap-CI rule and the sign-flip test do not give identical verdicts cell by cell (one "recall worse than nearest" cell has sign-flip p = 0.126); the corrected tables above come from the sign-flip tests.

### Round 3d. Cache microbenchmark for the step above N^2

The VM reports L1d 48 KiB per core, L2 2 MiB per core and one L3 of 320 MiB (`lscpu`, sysfs). Dense matvec bandwidth (matrix bytes per second) was measured against working-set size for complex128 and complex64, together with a plain single-thread streaming read (numpy sum of a float64 array, 16 KiB to 1 GiB), each batch load-gated as in round 2. Two full runs:

**Bandwidth drop events (effective bandwidth falls to <= 0.75 of the previous size); working-set brackets in MiB**

| run | thread mode | series | events (bracket MiB, bandwidth ratio) |
| --- | --- | --- | --- |
| 1 | one thread | complex128 matvec | 1.0 to 2.2 (0.73); 2.2 to 4.0 (0.68); 100.0 to 144.0 (0.51) |
| 1 | one thread | complex64 matvec | 1.1 to 2.0 (0.69); 2.0 to 4.5 (0.64); 128.0 to 200.0 (0.61) |
| 1 | one thread | streaming read (single thread) | 0.2 to 0.3 (0.74); 101.6 to 128.0 (0.27) |
| 1 | default threads | complex128 matvec | 16.0 to 36.0 (0.58); 196.0 to 256.0 (0.70) |
| 1 | default threads | complex64 matvec | 8.0 to 18.0 (0.63); 200.0 to 288.0 (0.69); 288.0 to 512.0 (0.64) |
| 1 | default threads | streaming read (single thread) | 101.6 to 128.0 (0.36) |
| 2 | one thread | complex128 matvec | 1.0 to 2.2 (0.69); 2.2 to 4.0 (0.74); 144.0 to 196.0 (0.74) |
| 2 | one thread | complex64 matvec | 2.0 to 4.5 (0.56); 128.0 to 200.0 (0.73) |
| 2 | one thread | streaming read (single thread) | 0.3 to 0.4 (0.74); 161.3 to 203.2 (0.51) |
| 2 | default threads | complex128 matvec | 4.0 to 9.0 (0.62); 9.0 to 16.0 (0.67); 16.0 to 36.0 (0.56) |
| 2 | default threads | complex64 matvec | 8.0 to 18.0 (0.65); 288.0 to 512.0 (0.65); 800.0 to 1152.0 (0.65) |
| 2 | default threads | streaming read (single thread) | 161.3 to 203.2 (0.63); 203.2 to 256.0 (0.64) |

**complex128 matvec effective bandwidth (GB/s) around the step, one BLAS thread** (matrix size in MiB = 16 N^2 / 2^20)

| N | matrix MiB | run 1 | run 2 |
| --- | --- | --- | --- |
| 2048 | 64 | 28.7 | 28.8 |
| 2560 | 100 | 26.6 | 27.4 |
| 3072 | 144 | 13.6 | 26.5 |
| 3584 | 196 | 12.7 | 19.6 |
| 4096 | 256 | 12.6 | 15.1 |
| 5120 | 400 | 12.4 | 13.1 |
| 6144 | 576 | 12.1 | 12.0 |
| 8192 | 1024 | 11.6 | 12.1 |

| Pre-declared hypothesis | One thread | Default threads |
| --- | --- | --- |
| m1: a matvec drop brackets a reported cache size | rule satisfied in both runs, but only through small events (1 to 4 MiB) that bracket the 2 MiB L2; the large event that corresponds to the step above N^2 (100 to 144 MiB in run 1, 144 to 196 MiB in run 2) brackets no reported cache size, in particular not the 320 MiB L3 | rejected in both runs |
| m2: the matvec drop coincides with a drop in the plain streaming-read bandwidth | **supported in both runs**: matvec 100 to 144 MiB vs streaming cliff 101.6 to 128 MiB (26.5 to 7.2 GB/s, ratio 0.27) in run 1; matvec 144 to 196 MiB vs streaming 161 to 203 MiB (ratio 0.51) in run 2 | rejected in both runs |

**Answer.** With one BLAS thread, the step above N^2 is consistent with a memory-hierarchy cliff that a plain streaming read sees at the same working-set size, and not with the 320 MiB L3 the VM reports: bandwidth falls at about 100 to 200 MiB, well below it. The cliff moved between the two runs (about 100 to 128 MiB in run 1, 144 to 203 MiB in run 2), and in rounds 1 and 2 the step was seen between 144 and 256 MiB. The pre-registered m1 rule was too permissive (it was satisfied by L2 events unrelated to the step), so the m1 "supported" label should not be read as support for the L3 explanation. With default threads (8 threads) the pre-declared criteria are not met in either run; the large drops there bracket 16 to 36 MiB and 196 to 256 MiB in run 1 and 4 to 36 MiB in run 2, so several include the 16 MiB total of the eight L2 caches (exploratory observation, not tested). Not established: why the cliff is below the reported L3 and why it moves (a guest that sees only part of a shared last-level cache, other tenants, and TLB or huge-page effects are all candidates; none was tested here). Page-table and hypervisor effects are not separated by this experiment.

### Round 3e. complex128 timed directly up to N = 14336

The unreachable N = 16384 (4 GiB matrix) was replaced by a chunked complex128 matrix timed at N = 8192, 10240, 12288 and 14336 (peak 3.12 GiB at 14336, under the 4 GiB cap and half of available memory when allowed). N = 12288 was not measured in run 1: after waiting 300 s the memory rule (peak at most half of MemAvailable, 4.74 GB at that moment) still failed. The N = 8192 point cross-checks the chunked implementation against round 2 (`HolographicMemory`): one thread 2.48 s (run 1) and 2.13 s (run 2) versus 2.30 s in round 2; default threads 0.308 and 0.306 s versus 0.304 to 0.321 s.

**complex128 timed directly (chunked 16 N^2-byte matrix, 25 recall steps, load-controlled; median seconds [95% bootstrap CI])**

| thread mode | run | N | median (s) | IQR (s) | GFLOP/s | peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| one thread | 1 | 8192 | 2.4758 [2.3358, 2.6117] | 0.1429 | 5.4 | 1.06 |
| one thread | 1 | 10240 | 3.7606 [3.5005, 5.9931] | 0.2592 | 5.6 | 1.62 |
| one thread | 1 | 12288 | not measured: memory rule (peak 2.31 GiB vs half of MemAvailable at the time, after waiting 300 s) | | | |
| one thread | 1 | 14336 | 8.1640 [7.5491, 9.4013] | 0.5784 | 5.0 | 3.12 |
| one thread | 2 | 8192 | 2.1267 [2.0970, 2.1554] | 0.0268 | 6.3 | 1.06 |
| one thread | 2 | 10240 | 3.3852 [3.3283, 4.1109] | 0.1710 | 6.2 | 1.62 |
| one thread | 2 | 12288 | 5.0188 [4.9622, 5.0573] | 0.0463 | 6.0 | 2.31 |
| one thread | 2 | 14336 | 6.8392 [6.6964, 6.9412] | 0.1428 | 6.0 | 3.12 |
| default threads | 1 | 8192 | 0.3080 [0.3002, 0.3230] | 0.0024 | 43.6 | 1.06 |
| default threads | 1 | 10240 | 0.4827 [0.4711, 0.5378] | 0.0063 | 43.4 | 1.62 |
| default threads | 1 | 12288 | 0.6981 [0.6857, 0.7553] | 0.0112 | 43.3 | 2.31 |
| default threads | 1 | 14336 | 1.0066 [0.9603, 1.1823] | 0.0233 | 40.8 | 3.12 |
| default threads | 2 | 8192 | 0.3056 [0.2968, 0.3198] | 0.0085 | 43.9 | 1.06 |
| default threads | 2 | 10240 | 0.4672 [0.4479, 0.4738] | 0.0113 | 44.9 | 1.62 |
| default threads | 2 | 12288 | 0.6740 [0.6551, 0.8020] | 0.0261 | 44.8 | 2.31 |
| default threads | 2 | 14336 | 0.8952 [0.8726, 0.9254] | 0.0148 | 45.9 | 3.12 |

**Fit on the measured sizes (N >= 8192) and extrapolation to 16384 (EXTRAPOLATED, NOT MEASURED)**

| thread mode | run | points | alpha | 95% CI | extrapolated N = 16384 (s) [interval] |
| --- | --- | --- | --- | --- | --- |
| one thread | 1 | 3 | 2.146 | [1.931, 2.425] | 10.71 [9.65, 12.59] |
| one thread | 2 | 4 | 2.095 | [2.011, 2.124] | 9.09 [8.91, 9.26] |
| default threads | 1 | 4 | 2.101 | [2.015, 2.343] | 1.31 [1.26, 1.50] |
| default threads | 2 | 4 | 1.930 | [1.853, 2.044] | 1.16 [1.13, 1.27] |

- **The original claim (recall at N = 16384 in 2.2 to 3.5 s) is not matched by either thread mode.** Measured at N = 14336: 6.8 to 8.2 s with one thread and 0.90 to 1.01 s with default threads; scaling by (16384/14336)^2 = 1.31 gives about 9 to 11 s and 1.2 to 1.3 s, in line with the fit extrapolations. The claim lies between the two modes; an intermediate thread count (2 or 4) was not tested and could give a number in that range.
- The extrapolation to N = 16384 remains an extrapolation (a 14% extension in N beyond the last measured point), with sampling-noise intervals only. complex128 at 16384 itself was not measured and cannot be under the 4 GiB cap.
- Alpha over N >= 8192 is 2.09 to 2.15 with one thread and 1.93 to 2.10 with default threads; the CIs come from 3 to 4 points and are wide.

### Round 3f. Updated claims versus results

| Claim | Status after round 3 |
| --- | --- |
| "Recall denoises only at low load" (round 2, 25 steps) | Still true at 25 steps, but 25 steps is not a converged state: with random patterns, accuracy keeps falling to step 100 (BH verdicts 12 D / 8 X / 4 ~ at step 25, 7 / 15 / 2 at step 100), and exact stored patterns at P/N >= 1/8 drift away. The cause is cross-talk (Hadamard patterns: 24 of 24 cells denoise at both steps). |
| "Recall never beats a nearest-stored lookup" | Survives Holm and BH correction (0 cells with recall better, in every version). |
| Round-2 verdict counts 61 / 37 / 10 | BH: unchanged. Holm: 57 / 36 / 15. All 126 stored cells reproduce exactly. |
| Mechanism of the high-load failure | Spurious states from cross-talk (supported); sign mirror and period-2 oscillation rejected. Which spurious states they are is unknown. |
| Correlated patterns | Capacity falls sharply (knee P/N 3/16 to 1/32 or 1/16 at rho = 0.3); recall collapses onto the shared component (post hoc); the preregistered denoising-count rule is inconclusive and the gain-over-cue measure is misleading under collapse. |
| The step above N^2 is a cache effect | Consistent with a memory-hierarchy cliff seen by a plain streaming read (one thread, both runs); not consistent with the reported 320 MiB L3; the cliff moves between runs. Default threads: criteria not met. |
| complex128 at N = 16384 in 2.2 to 3.5 s | Not matched: measured 14336 and extrapolated 16384 give about 1.2 to 1.3 s (default) and 9 to 11 s (one thread). Still not measured at 16384. |

## Round 4: early stopping, what the spurious states are, BLAS threads

Round 4 follows a locked plan, `benchmark_protocol_round4.json` and `benchmark_protocol_round4.lock` (hash `0b36a13189b51b91fedacf61336eb42a50d7daf077f00f80a3b9e4561803082f`, plan text in [round4_plan.md](round4_plan.md)), committed in `291dc2e` before `benchmark_round4.py` (`3f1f967`, fix `41c0262`) existed and before any full round-4 run. Results commit: `3175135`. Raw reports: `results/round4/` (`early.json`, `spurious.json`, `threads_run1.json`, `threads_run2.json`, merged `assembled.json`). Seed 20261028, so no round-3 trial is reused. Nothing was uploaded.

```sh
python benchmark_round4.py --quick                                  # CI smoke
python benchmark_round4.py --full --json report_round4.json         # manual, all parts, about 15 minutes
python benchmark_round4.py --full --part early --json early.json    # chunked, as run here (also --part spurious, --part threads --thread-run 1 or 2)
python benchmark_round4.py --assemble early.json spurious.json threads_run1.json threads_run2.json --json assembled.json
```

Early-step and spurious-state hashes reproduced exactly on a second full run (`3a52bc80…7625` and `59341e60…93a8`). The assembled report has `deterministic_sha256` `7e23f21a2695e1d0a75b52d56295bce29e0a3e86eda50ba30a13d9b9b6826281` (the thread timings are not in the hash).

**Disclosures and deviations.** (1) No change to the locked plan. (2) The quick profile (10 to 20 trials) was run, and its spurious-state output looked at, before the full runs; it uses the same seed streams, so its trials are the first trials of three of the full cells. A throwaway simulation with an unrelated seed was run to see how well the mixture detector recovers planted mixtures at N = 64 (about 90%); it did not change any rule. (3) The first full launch crashed at once with a TypeError because the "as listed" entries of the full profile were not resolved; the fix `41c0262` was committed before any full result existed. (4) The helper that merges the thread chunks and analyses both runs together (needed for T2 to T4) was added after the two thread chunks had run; the chunk files contain the per-run analyses and the merged analysis is a pure function of the raw batches in them (tested). (5) N = 8192 was added to the requested 2048 and 4096 as a declared extension. (6) Early-step and spurious runs used one BLAS thread; thread timings used a fresh process per mode and a quiet box (all 30 batches were accepted at the first attempt, external busy cores at most 0.36). (7) The selection scores for steps 1 and 2 were nearly tied; they are stored in the JSON under `selection_only_not_for_inference` and are not used below.

### Round 4a. Early stopping (held-out data only)

Random patterns, bit-flip cues q = 0.1 and 0.3, N = 64, 128, 256, P/N = 1/8, 3/16, 1/4 (18 cells fixed in advance), 300 held-out trials per cell. A separate selection seed set (100 trials per cell) chose one fixed stopping step, **step 2**, before the held-out data were evaluated. Every number below is from the held-out trials.

**Early-step verdicts**

| Comparison (held-out, 18 cells) | cells higher / n.s. / lower | pooled mean difference [95% CI] | Verdict |
| --- | --- | --- | --- |
| E1a: mean(steps 2 to 5) minus step 25 | 18 / 0 / 0 | +0.1004 [+0.0980, +0.1028] | **supported** |
| E1b: mean(steps 2 to 5) minus step 1 | 5 / 2 / 11 | -0.0151 [-0.0163, -0.0139] | **rejected** |
| E2a: step 2 (selected) minus step 25 | 18 / 0 / 0 | +0.1168 [+0.1141, +0.1193] | **supported** |
| E2b: step 2 (selected) minus cue | 18 / 0 / 0 | +0.0988 [+0.0974, +0.1002] | **supported** |

**What this supports.** Stopping after 2 steps instead of 25 beats the 25-step default in all 18 cells, by +0.117 on average (CI [+0.114, +0.119]), and beats the corrupted cue in all 18 cells, by +0.099 (CI [+0.097, +0.100]). The earlier statement that recall is worse than the cue at high load holds for 25 steps (rounds 2 and 3) but not for 2 steps: the failure is, to a large extent, a consequence of running the dynamics too long (round 3 showed the drift away from the stored pattern). Averaged over the 18 cells the held-out accuracy is 0.800 (cue), 0.898 (step 1), 0.899 (step 2), 0.888 (step 3), 0.867 (step 5), 0.829 (step 10) and 0.782 (step 25).

**What it does not support.** The "peak at steps 2 to 5" is not a peak above step 1: step 1 is at least as good in most cells (E1b rejected: 11 cells lower, 5 higher, pooled -0.015). The best step is 1, 2 or 3 per cell and depends on load and noise (hindsight, descriptive: 1 at P/N = 1/4; 1 for q = 0.1 and 2 for q = 0.3 at 3/16; 2 for q = 0.1 and 3 for q = 0.3 at 1/8). One synchronous update already does most of the work; later steps help only a little at P/N = 1/8 and hurt at higher load. The result is for random patterns and these N and P/N; the P/N = 1/16 contrast cells (exploratory) show that where 25 steps already work, stopping early changes little (E2a +0.0004, CI [-0.0011, +0.0020]), while beating the cue by +0.18. Whether stopping early beats the nearest-stored-pattern lookup, correlated patterns, other noise types and larger N were not tested. A stopping step chosen on held-out trials would be optimistic; the one reported here was not.

**Held-out accuracy by step (300 trials per cell; last 6 rows are the exploratory P/N = 1/16 contrast cells)**

| N | q | P/N | cue | step 1 | step 2 | step 3 | step 5 | step 10 | step 25 | selected-per-cell step (exploratory) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.1 | 0.125 | 0.899 | 0.989 | 0.992 | 0.992 | 0.990 | 0.984 | 0.973 | 2 |
| 64 | 0.1 | 0.1875 | 0.898 | 0.970 | 0.969 | 0.962 | 0.946 | 0.920 | 0.888 | 1 |
| 64 | 0.1 | 0.25 | 0.902 | 0.949 | 0.935 | 0.914 | 0.882 | 0.837 | 0.795 | 1 |
| 64 | 0.3 | 0.125 | 0.702 | 0.875 | 0.895 | 0.893 | 0.883 | 0.864 | 0.834 | 3 |
| 64 | 0.3 | 0.1875 | 0.701 | 0.828 | 0.835 | 0.826 | 0.807 | 0.774 | 0.742 | 2 |
| 64 | 0.3 | 0.25 | 0.702 | 0.788 | 0.784 | 0.766 | 0.738 | 0.707 | 0.685 | 1 |
| 128 | 0.1 | 0.125 | 0.901 | 0.989 | 0.993 | 0.993 | 0.991 | 0.983 | 0.958 | 2 |
| 128 | 0.1 | 0.1875 | 0.900 | 0.968 | 0.965 | 0.956 | 0.936 | 0.886 | 0.823 | 1 |
| 128 | 0.1 | 0.25 | 0.899 | 0.946 | 0.928 | 0.905 | 0.865 | 0.798 | 0.737 | 1 |
| 128 | 0.3 | 0.125 | 0.701 | 0.875 | 0.902 | 0.907 | 0.902 | 0.873 | 0.823 | 3 |
| 128 | 0.3 | 0.1875 | 0.703 | 0.822 | 0.825 | 0.809 | 0.781 | 0.735 | 0.693 | 2 |
| 128 | 0.3 | 0.25 | 0.702 | 0.786 | 0.781 | 0.761 | 0.729 | 0.691 | 0.658 | 1 |
| 256 | 0.1 | 0.125 | 0.899 | 0.988 | 0.992 | 0.992 | 0.991 | 0.981 | 0.935 | 2 |
| 256 | 0.1 | 0.1875 | 0.899 | 0.966 | 0.962 | 0.950 | 0.925 | 0.863 | 0.768 | 1 |
| 256 | 0.1 | 0.25 | 0.899 | 0.946 | 0.927 | 0.899 | 0.850 | 0.768 | 0.681 | 1 |
| 256 | 0.3 | 0.125 | 0.697 | 0.866 | 0.889 | 0.893 | 0.885 | 0.847 | 0.781 | 3 |
| 256 | 0.3 | 0.1875 | 0.703 | 0.826 | 0.830 | 0.813 | 0.779 | 0.726 | 0.668 | 2 |
| 256 | 0.3 | 0.25 | 0.700 | 0.787 | 0.781 | 0.759 | 0.724 | 0.682 | 0.642 | 1 |
| 64 | 0.1 | 0.0625 | 0.902 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1 |
| 64 | 0.3 | 0.0625 | 0.700 | 0.941 | 0.956 | 0.959 | 0.960 | 0.957 | 0.956 | 9 |
| 128 | 0.1 | 0.0625 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 2 |
| 128 | 0.3 | 0.0625 | 0.702 | 0.946 | 0.974 | 0.981 | 0.981 | 0.980 | 0.970 | 3 |
| 256 | 0.1 | 0.0625 | 0.901 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
| 256 | 0.3 | 0.0625 | 0.701 | 0.946 | 0.984 | 0.993 | 0.997 | 0.997 | 0.985 | 6 |

**Contrast cells P/N = 1/16 (6 cells, exploratory counts only, no verdict)**

| Comparison | higher / n.s. / lower | pooled mean difference [95% CI] |
| --- | --- | --- |
| E1a_early_vs_step25 | 2 / 4 / 0 | +0.0033 [+0.0020, +0.0046] |
| E1b_early_vs_step1 | 4 / 2 / 0 | +0.0162 [+0.0149, +0.0176] |
| E2a_stop_vs_step25 | 0 / 6 / 0 | +0.0004 [-0.0011, +0.0020] |
| E2b_stop_vs_cue | 6 / 0 / 0 | +0.1845 [+0.1822, +0.1868] |

### Round 4b. What are the spurious final states?

Random patterns, N = 64, 128, 256, q = 0, 0.1, 0.3, P/N = 1/8, 3/16, 1/4, 100 trials per cell, final state at step 100 (2700 states). Categories with precedence: stored (within 0.10 of a stored pattern or its negation), mixture_3 or mixture_5 (within 0.15 of the sign of the sum of the 3 or 5 stored patterns with the largest overlaps), partial_overlap (largest overlap at least 0.5), unstructured (largest overlap below 0.5). Of the 2700 states, 186 (6.9%) are stored (142 the target, 21 another stored pattern, 23 a negated one) and 2514 are not.

**Spurious-state hypotheses**

| Hypothesis | Measure | Result | Verdict |
| --- | --- | --- | --- |
| Sp1_mixture_states | fraction of 2514 non-stored states | 1646 / 2514 = 0.655, Wilson 95% [0.636, 0.673] | **supported** |
| Sp2_unstructured | fraction of 2514 non-stored states | 733 / 2514 = 0.292, Wilson 95% [0.274, 0.310] | **partial or inconclusive** |
| Sp3_partial_overlap | fraction of 2514 non-stored states | 135 / 2514 = 0.054, Wilson 95% [0.046, 0.063] | **rejected** |
| Sp4_fixed_points | fraction of 2514 non-stored states | 2133 / 2514 = 0.848, Wilson 95% [0.834, 0.862] | **supported** |
| Sp5_mean_direction | paired cosine with the mean of the stored patterns, final minus random state | 0.190 vs 0.071; difference +0.118, 95% CI [+0.112, +0.124] | **supported** |

**Pooled, the non-stored states are mostly mixture states (65.5%), fixed points (84.8%), and lean toward the mean of the stored patterns. But the pooled number hides a strong dependence on N** (descriptive, same rules applied per N after the fact): at N = 64, 735 of 738 non-stored states are mixtures (Wilson [0.988, 0.999]); at N = 128, 752 of 876 (0.86); at N = 256, 159 of 900 (0.18, Wilson [0.153, 0.203]) while 650 of 900 (0.72, Wilson [0.692, 0.750]) are unstructured. The unstructured share at N = 256 rises with load (45% at P/N = 1/8, 82% at 3/16, 90% at 1/4). At large N the dynamics ends in states that have no large overlap with any stored pattern (the classical spin-glass picture is consistent with this, but it was not tested here); at small N it ends in mixtures of 3 or 5 stored patterns. The pooled "mixture supported" verdict is therefore a statement about this N mix (a third each of N = 64, 128, 256), not a general one.

**Final-state categories (step 100, 100 trials per cell, q pooled)**

| Cells | Final states | Stored | mixture_3 | mixture_5 | partial_overlap | unstructured | Mixture share of non-stored | Unstructured share of non-stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all 27 cells | 2700 | 186 (7%) | 1042 | 604 | 135 | 733 | 0.65 | 0.29 |
| N = 64 | 900 | 162 (18%) | 629 | 106 | 2 | 1 | 1.00 | 0.00 |
| N = 128 | 900 | 24 (3%) | 396 | 356 | 42 | 82 | 0.86 | 0.09 |
| N = 256 | 900 | 0 (0%) | 17 | 142 | 91 | 650 | 0.18 | 0.72 |
| N = 64, P/N = 0.125 | 300 | 119 (40%) | 171 | 9 | 1 | 0 | 0.99 | 0.00 |
| N = 64, P/N = 0.1875 | 300 | 27 (9%) | 241 | 32 | 0 | 0 | 1.00 | 0.00 |
| N = 64, P/N = 0.25 | 300 | 16 (5%) | 217 | 65 | 1 | 1 | 0.99 | 0.00 |
| N = 128, P/N = 0.125 | 300 | 23 (8%) | 205 | 66 | 5 | 1 | 0.98 | 0.00 |
| N = 128, P/N = 0.1875 | 300 | 1 (0%) | 122 | 135 | 13 | 29 | 0.86 | 0.10 |
| N = 128, P/N = 0.25 | 300 | 0 (0%) | 69 | 155 | 24 | 52 | 0.75 | 0.17 |
| N = 256, P/N = 0.125 | 300 | 0 (0%) | 14 | 106 | 45 | 135 | 0.40 | 0.45 |
| N = 256, P/N = 0.1875 | 300 | 0 (0%) | 2 | 27 | 26 | 245 | 0.10 | 0.82 |
| N = 256, P/N = 0.25 | 300 | 0 (0%) | 1 | 9 | 20 | 270 | 0.03 | 0.90 |

- **The mean-direction effect is small.** The cosine between the final state and the mean of the stored patterns is 0.190 against 0.071 for a random state (+0.118). It is a modest lean, not a collapse onto the mean (compare the correlated-pattern collapse in round 3, where the patterns themselves share a common component).
- **Fixed points.** 84.8% of non-stored states are fixed points (94% at N = 64, 89% at N = 128, 73% at N = 256); none is a period-2 cycle. The still-moving fraction is largest for partial overlaps (41 of 135) and unstructured states (170 of 733).
- **Energy.** With the real coupling matrix, the final non-stored states are deeper energy minima than the stored patterns once P/N exceeds about 1/8 (descriptive; N = 256, P/N = 1/4: -160.9 against -127.7 per bit in the same units). Deeper than the stored patterns is what makes them attractors that win over the target.
- **Detector limits.** The classifier is greedy (top-k overlaps) and a state can be near a mixture of other patterns than the ones it picks. Planted 3-mixtures with 5% noise are recovered 96.3% of the time (89.4% at N = 64, 99.3% at N = 128, 100% at N = 256), planted 5-mixtures only 70.9% (44.8% at N = 64). A random state is never classified as stored or mixture. So the mixture counts at N = 64 and 128 are, if anything, lower bounds, and the "unstructured" count at N = 256 is little affected. The result depends on the thresholds: with tau_mixture = 0.10 the mixture count falls to 891 of 2700, with 0.20 it rises to 2310 (table below), so the shares of mixture and unstructured states are definition-dependent. The harness criteria were met (stored 98.1%, mixture_3 96.3%, null 0%).

**Sensitivity (pooled, 2700 final states)**

| tau_stored / tau_mixture | stored | mixture (3 or 5) | partial_overlap | unstructured |
| --- | --- | --- | --- | --- |
| 0.05/0.15 | 94 | 1734 | 139 | 733 |
| 0.2/0.15 | 852 | 1004 | 111 | 733 |
| 0.1/0.1 | 186 | 891 | 607 | 1016 |
| 0.1/0.2 | 186 | 2310 | 6 | 198 |

**Harness controls (2700 draws; by N descriptive)**

| Control | pooled | N = 64 | N = 128 | N = 256 |
| --- | --- | --- | --- | --- |
| stored_5pct_flips_is_stored | 0.981 | 0.956 | 0.988 | 0.999 |
| mixture3_5pct_flips_is_mixture3 | 0.963 | 0.894 | 0.993 | 1.000 |
| mixture5_5pct_flips_is_mixture5 | 0.709 | 0.448 | 0.717 | 0.963 |
| mixture5_5pct_flips_is_mixture3_or_5 | 0.866 | 0.754 | 0.861 | 0.983 |
| null_random_is_stored | 0.000 | 0.000 | 0.000 | 0.000 |
| null_random_is_mixture | 0.000 | 0.000 | 0.000 | 0.000 |

**Dynamics of non-stored final states**

| Category | fixed point | period 2 | still moving |
| --- | --- | --- | --- |
| mixture_3 | 948 | 0 | 94 |
| mixture_5 | 528 | 0 | 76 |
| partial_overlap | 94 | 0 | 41 |
| unstructured | 563 | 0 | 170 |

**Energy per bit (descriptive; lower is deeper)**

| N | P/N | final non-stored states (mean over q, cells) | stored patterns (first 4 per trial, same trials) |
| --- | --- | --- | --- |
| 64 | 0.125 | -31.5 | -31.7 |
| 64 | 0.1875 | -34.3 | -31.5 |
| 64 | 0.25 | -37.0 | -31.4 |
| 128 | 0.125 | -63.5 | -63.5 |
| 128 | 0.1875 | -71.7 | -63.5 |
| 128 | 0.25 | -77.7 | -63.2 |
| 256 | 0.125 | -132.0 | -127.6 |
| 256 | 0.1875 | -147.7 | -127.4 |
| 256 | 0.25 | -160.9 | -127.7 |

### Round 4c. BLAS threads, complex128, 25-step recall

N = 2048 and 4096 as requested, N = 8192 as a declared extension; 1, 2, 3, 4 and default threads (8 on this VM); two runs in opposite mode order; 7 repeats per batch, load-gated as before (every batch passed at the first attempt). The extra CPU/wall column checks that the thread setting took effect: it is 1.0, 2.0, 3.0 and 4.0 (within 0.14) for 1 to 4 threads and 7.8 to 7.95 for default, so no mode was flagged.

**Median seconds for 25 recall steps, complex128 (7 repeats per batch; 95% CI of the median; speedup = t(1 thread)/t(k) in the same run)**

| N | threads | run 1 seconds | run 1 speedup [CI] | run 2 seconds | run 2 speedup [CI] | cpu/wall (run 1, run 2) |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | 1 | 0.0563 [0.0534, 0.0576] | 1 | 0.0585 [0.0577, 0.0607] | 1 | 1.00, 1.00 |
| 2048 | 2 | 0.0308 [0.0296, 0.0313] | 1.83 [1.71, 1.91] | 0.0312 [0.0300, 0.0324] | 1.88 [1.78, 1.97] | 2.00, 1.99 |
| 2048 | 3 | 0.0207 [0.0201, 0.0212] | 2.72 [2.55, 2.85] | 0.0210 [0.0205, 0.0214] | 2.78 [2.69, 2.95] | 2.99, 3.00 |
| 2048 | 4 | 0.0153 [0.0152, 0.0160] | 3.67 [3.42, 3.76] | 0.0168 [0.0163, 0.0173] | 3.49 [3.34, 3.63] | 3.86, 4.00 |
| 2048 | default | 0.0100 [0.0100, 0.0108] | 5.61 [5.05, 5.75] | 0.0105 [0.0104, 0.0122] | 5.57 [4.74, 5.80] | 7.77, 7.90 |
| 4096 | 1 | 0.3718 [0.3602, 0.4097] | 1 | 0.3191 [0.3166, 0.3276] | 1 | 1.00, 1.00 |
| 4096 | 2 | 0.1602 [0.1575, 0.1777] | 2.32 [2.05, 2.57] | 0.1556 [0.1527, 0.1647] | 2.05 [1.92, 2.12] | 2.00, 2.00 |
| 4096 | 3 | 0.1117 [0.0994, 0.1174] | 3.33 [3.10, 4.08] | 0.1033 [0.1019, 0.1066] | 3.09 [2.97, 3.20] | 3.01, 3.01 |
| 4096 | 4 | 0.0764 [0.0756, 0.0801] | 4.87 [4.55, 5.42] | 0.0741 [0.0732, 0.0754] | 4.31 [4.20, 4.44] | 4.01, 4.05 |
| 4096 | default | 0.0445 [0.0418, 0.0454] | 8.36 [7.98, 9.70] | 0.0411 [0.0397, 0.0450] | 7.76 [7.03, 8.14] | 7.95, 7.85 |
| 8192 | 1 | 2.1699 [2.1653, 2.2710] | 1 | 2.1316 [2.1174, 2.1659] | 1 | 1.00, 1.00 |
| 8192 | 2 | 1.0634 [1.0401, 1.0830] | 2.04 [2.00, 2.17] | 1.0410 [1.0260, 1.0491] | 2.05 [2.02, 2.10] | 2.00, 2.00 |
| 8192 | 3 | 0.6905 [0.6879, 0.7358] | 3.14 [2.95, 3.29] | 0.6881 [0.6854, 0.7066] | 3.10 [3.00, 3.15] | 3.00, 3.00 |
| 8192 | 4 | 0.5252 [0.5210, 0.5272] | 4.13 [4.11, 4.35] | 0.5135 [0.5042, 0.5266] | 4.15 [4.03, 4.27] | 4.00, 4.00 |
| 8192 | default | 0.2881 [0.2811, 0.2964] | 7.53 [7.32, 8.05] | 0.2853 [0.2783, 0.2913] | 7.47 [7.28, 7.78] | 7.85, 7.93 |

**Thread hypotheses and extrapolation to N = 16384 (EXTRAPOLATED, NOT MEASURED)**

| Hypothesis | Units satisfying | Verdict |
| --- | --- | --- |
| T1_speedup | 18 of 18 | **supported** |
| T2_monotone | 6 of 6 | **supported** |
| T3_repeatability | 15 of 15 | **supported** |
| T4_original_claim | see table | **rejected** |

| Threads | declared estimate run 1 / run 2 (t8192 squared over t4096) | exploratory: 4 x t8192 run 1 / run 2 |
| --- | --- | --- |
| 1 | 12.66 s / 14.24 s | 8.68 s / 8.53 s |
| 2 | 7.06 s / 6.97 s | 4.25 s / 4.16 s |
| 3 | 4.27 s / 4.59 s | 2.76 s / 2.75 s |
| 4 | 3.61 s / 3.56 s | 2.10 s / 2.05 s |
| default | 1.87 s / 1.98 s | 1.15 s / 1.14 s |


- **Time falls monotonically from 1 to 4 threads and the two runs agree** (T1, T2, T3 supported: 18 of 18, 6 of 6 and 15 of 15 units). Efficiency (speedup divided by threads) is 0.87 to 0.94 at N = 2048, 1.02 to 1.05 at N = 8192 and 1.03 to 1.22 at N = 4096 (the single-thread time at N = 4096 is 5.5 to 6.6 times the time at N = 2048, more than N^2 predicts (4), which inflates the speedups there; this fits with the round-3 cache-size cliff for one thread, but this experiment did not test that). Default threads (8) give 5.6 to 8.4 times the single-thread speed.
- **Original claim (recall at N = 16384 in 2.2 to 3.5 s).** T4 is **rejected** by the pre-declared estimator: the extrapolation t(8192)^2 / t(4096) gives 3.61 s and 3.56 s with 4 threads (3 threads: 4.3 to 4.6 s), just above the 3.5 s limit, 1.9 to 2.0 s with default threads. That estimator is crude because the 8192 / 4096 time ratio (5.8 to 6.9) is larger than the N^2 value 4, which the one-doubling extrapolation compounds. An exploratory estimate that scales the measured 8192 time by 4 gives 2.05 to 2.10 s (4 threads), 2.75 s (3 threads) and 4.2 s (2 threads), which would be inside the claimed range for 3 to 4 threads; the round-3 measurement at N = 14336 with one thread (6.8 to 8.2 s against 2.1 to 2.2 s at 8192 here) supports about N^2 scaling above 8192. Both estimates are extrapolations; **complex128 at N = 16384 has still not been measured** (4 GiB matrix, over the cap), and neither estimate is a measurement of the claim. The accurate statement is: with 4 threads the original claim is neither confirmed nor excluded; with the pre-declared estimator it is slightly outside, with a plain N^2 scaling it is inside. Measuring N = 14336 with 3 and 4 threads (3.1 GiB, fits) would settle it and was not part of the plan.

### Round 4d. Updated claims versus results

| Claim | Status after round 4 |
| --- | --- |
| "Recall is worse than the cue at high load" (rounds 2 and 3) | True for 25 steps. False for 2 steps: stopping after 2 steps (chosen on separate seeds) beats the cue in 18 of 18 high-load cells (+0.099) and beats 25 steps in 18 of 18 (+0.117), random patterns. |
| "Accuracy peaks at steps 2 to 5" (round 3, exploratory) | Not confirmed as stated: steps 2 to 5 are far above step 25 (18 of 18) but not above step 1 (rejected). The best step is 1 to 3. |
| Spurious states at high load | Mostly not stored patterns (93%). At N = 64 and 128 they are mixtures of 3 or 5 stored patterns (99.6% and 86%); at N = 256 they are mostly unstructured (72%, rising with load). 85% are fixed points; they are deeper energy minima than the stored patterns at P/N of 3/16 and above; the lean toward the mean pattern is small. Definition-dependent (thresholds), classifier limits stated above. |
| More BLAS threads make the 25-step complex128 recall faster | Supported: near-linear from 1 to 4 threads, repeatable across two runs (efficiency 0.87 to 1.22); default (8) threads 5.6 to 8.4 times. |
| complex128 at N = 16384 in 2.2 to 3.5 s | Still not measured. Pre-declared estimate for 4 threads: 3.6 s (rejected by a hair); plain N^2 scaling from 8192: 2.05 to 2.10 s. Unresolved. |

## Preregistration record

The locked plans of rounds 1 to 4 (`benchmark_protocol*.json` and `.lock`, `docs/round2_plan.md` to `docs/round4_plan.md`, the protocol hashes and the commit ids at which each plan was committed before its experiments) are archived on Zenodo as a bundle with plain-text result summaries for rounds 3 and 4: [10.5281/zenodo.23675451](https://doi.org/10.5281/zenodo.23675451) (concept DOI [10.5281/zenodo.23675450](https://doi.org/10.5281/zenodo.23675450), CC-BY-4.0). The plans in the bundle contain no results; the summaries were added after the experiments and are marked as such. The archive and the git history are the author's own records, not an independent timestamp. The full reports are in `results/` in this repository.

## Limitations

- **Shared machine.** Timings were taken on a shared 8-core box while other jobs were running. Load averages are stored per mode in the report; high load inflates and destabilises timings, most visibly with default BLAS threads. Rerun on an idle machine before quoting numbers.
- **Machine-dependent.** Absolute times depend on the CPU, BLAS build and thread count; only the shapes (overhead, N^2 regime, memory step) are expected to transfer.
- **Extrapolation.** Intervals cover sampling noise of the fit only, not model error; the hold-out check shows large model error for one thread. complex128 at N = 16384 is unmeasured (round 2 measured complex64 only, a different dtype).
- **Capacity scope.** Random BPSK patterns, one target per memory, 25 steps, Gaussian phase noise only, 40 trials per cell, a coarse P/N grid. The round-1 noise levels do not flip bits, so round 1 does not test error correction; round 2 does (see above).
- **Round 4 scope.** The early-stopping result is for random patterns, bit-flip cues, N = 64 to 256 and P/N = 1/8 to 1/4 (and not compared with the nearest-stored lookup); the spurious-state categories are heuristic and threshold-dependent, N dominates their composition, and the planted-mixture recovery is imperfect at N = 64; thread timings are for one shared 8-CPU VM and one BLAS build; N = 16384 complex128 is still unmeasured.
- **Round 3 scope.** Mechanism tests use random and Hadamard patterns, bit-flip cues, N = 64, 128, 256, four loads, 50 trials per cell; spurious states are defined by distance only and not characterised; the sign-flip p-values assume a symmetric null distribution of the gain; the cache microbenchmark runs on one KVM guest whose cache report may not describe the physical machine, and its pre-declared m1 rule was too permissive; two full runs only; no thread count between 1 and 8 was tested.
- **Round 2 scope.** Random uncorrelated patterns, 3 values of N, 6 loads, 50 trials per cell, one target (the first stored pattern), 25 steps, no multiplicity correction across 108 cells, global-phase mirror not folded. Load control excludes CPU contention from other jobs but not cache, memory-bandwidth or VM effects, and timings still varied between runs by up to about 1.7x at some N. Two full runs only; one-thread complex128 N = 4096 and N = 6144 each have a usable batch in only one run.
- **The lock** is a recorded plan in git, not an independent timestamp.

---

# Original tool: `benchmark.py`

The sections below describe `benchmark.py` as first supplied and the earlier three-run comparison. Its scaling fit (N >= 128, three points) is superseded by the rigorous benchmark above. The text is kept for the record.

`benchmark.py` is unchanged from the version supplied. The `cseq_wave` imports it uses (`wave.WaveState`, `encode.bits_to_wave/wave_to_bits/text_to_bits/bits_to_text`, `memory.HolographicMemory.store/finalize/recall`) exist in the current package and needed no adapter.

## Deterministic results (original tool) (full run, seed 42)

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
