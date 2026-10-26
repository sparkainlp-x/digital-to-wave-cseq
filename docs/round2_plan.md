# Round-2 plan (written before any round-2 experiment was run)

Status: **plan only**. This file, `benchmark_protocol_round2.json` and `benchmark_protocol_round2.lock` are committed before `benchmark_round2.py` exists. The git history shows the order. The round-1 protocol and lock are unchanged (hash `826e3e6870f9e3b95b605e2e70701067b970909f196b739d8e662dd06f094c9f`, commit `3e067401e6617f3537aa888aad3f14851c980813`).

Round-2 protocol hash: `a06cd2e65d18d2eac28b57a1a74e8291ab5b81b7f31f28f7637685d8f8bcc27d`.

What the author knew when writing this: the round-1 results, including that Gaussian phase noise of at most 0.25 rad never flips a bit and therefore did not test denoising. Nothing about bit-flip cues or complex64 timings had been measured.

## Questions

1. **Does recall denoise?** Corrupt the cue of a stored pattern (independent bit flips with q in {0.05, 0.10, 0.20, 0.30}, control q = 0; and Gaussian phase noise sigma in {0.9, 1.2} rad, which flips bits) for N in {64, 128, 256} and P/N in {1/32, 1/16, 3/32, 1/8, 3/16, 1/4}, 50 trials per cell. The primary endpoint is the per-trial gain of recalled bit accuracy over the corrupted cue's own bit accuracy, with a paired percentile bootstrap 95% CI. A cell is "denoises" if the CI lies above 0, "degrades" if below 0, otherwise "inconclusive"; all cells are reported, there is no multiplicity correction and no single whole-system claim is made. Baselines: cue only, nearest stored pattern (matched filter), random pattern. Secondary: exact-recall fraction and nearest-stored retrieval fraction with Wilson intervals.
2. **Can N = 16384 be timed directly?** Only with a complex64 coupling matrix (2 GiB) built in row chunks, if peak <= 4 GiB and <= half of MemAvailable. That measures a different dtype than the original complex128 claim. A complex128 figure at 16384 is given only as a labelled estimate from the measured complex128/complex64 time ratio at N = 8192.
3. **Load control.** Each timing batch records `os.getloadavg()` before and after, and the external CPU load (system busy cores minus the worker's own). A batch above 0.5 external busy cores is discarded and rerun, up to 4 attempts; batches that never pass are flagged and excluded from the fit.

## Stated prior expectation

Classical Hopfield theory suggests recall should denoise at low load and lose that advantage near or above the capacity knee. The experiment can return "no denoising anywhere" and that would be reported as such.

## Not part of this plan

No quantum-hardware, QEC or quantum-advantage claim. No claim about the Coherent Sequence theory. Timings are machine-dependent.
