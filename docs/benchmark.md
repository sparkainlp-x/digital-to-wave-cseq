# Benchmark tools: reverse, scaling and capacity (SYNTHETIC)

Two tools live in the repo root:

| File | Role |
| --- | --- |
| `benchmark.py` | The first benchmark, supplied unchanged. Quick to run; its scaling fit is unstable (see "Original tool" below). |
| `benchmark_rigorous.py` | The upgraded benchmark: locked analysis plan, direct timings with confidence intervals, de-confounded capacity grid, null baselines, adversarial reverse tests. Settings live in `benchmark_protocol.json`. |

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
- **The noise levels tested do not test denoising.** A Gaussian phase error of 0.25 rad almost never pushes a BPSK phase past the pi/2 decision boundary, so the cue-only baseline (and the nearest-stored-pattern baseline) are 1.000 in every cell, and the recalled bit accuracy is **below** both whenever it is below 1. Here the memory adds crosstalk rather than error correction. Sigma = 0.1 and 0.25 give nearly the same curves for that reason. A benchmark that adds bit-flip noise to the cue is the natural follow-up; it was not added here because it was not in the locked protocol.
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

## Limitations

- **Shared machine.** Timings were taken on a shared 8-core box while other jobs were running. Load averages are stored per mode in the report; high load inflates and destabilises timings, most visibly with default BLAS threads. Rerun on an idle machine before quoting numbers.
- **Machine-dependent.** Absolute times depend on the CPU, BLAS build and thread count; only the shapes (overhead, N^2 regime, memory step) are expected to transfer.
- **Extrapolation.** Intervals cover sampling noise of the fit only, not model error; the hold-out check shows large model error. N = 16384 is unmeasured.
- **Capacity scope.** Random BPSK patterns, one target per memory, 25 steps, Gaussian phase noise only, 40 trials per cell, a coarse P/N grid. The noise levels do not flip bits, so the benchmark does not test error correction.
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
