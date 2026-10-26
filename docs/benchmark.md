# Benchmark tools: reverse, scaling and capacity (SYNTHETIC)

Two tools live in the repo root:

| File | Role |
| --- | --- |
| `benchmark.py` | The first benchmark, supplied unchanged. Quick to run; its scaling fit is unstable (see "Original tool" below). |
| `benchmark_rigorous.py` | The upgraded benchmark: locked analysis plan, direct timings with confidence intervals, de-confounded capacity grid, null baselines, adversarial reverse tests. Settings live in `benchmark_protocol.json`. |
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

## Limitations

- **Shared machine.** Timings were taken on a shared 8-core box while other jobs were running. Load averages are stored per mode in the report; high load inflates and destabilises timings, most visibly with default BLAS threads. Rerun on an idle machine before quoting numbers.
- **Machine-dependent.** Absolute times depend on the CPU, BLAS build and thread count; only the shapes (overhead, N^2 regime, memory step) are expected to transfer.
- **Extrapolation.** Intervals cover sampling noise of the fit only, not model error; the hold-out check shows large model error for one thread. complex128 at N = 16384 is unmeasured (round 2 measured complex64 only, a different dtype).
- **Capacity scope.** Random BPSK patterns, one target per memory, 25 steps, Gaussian phase noise only, 40 trials per cell, a coarse P/N grid. The round-1 noise levels do not flip bits, so round 1 does not test error correction; round 2 does (see above).
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
