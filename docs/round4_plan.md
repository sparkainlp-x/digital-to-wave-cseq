# Round-4 plan (written before any round-4 code or experiment)

Status: **plan only**. This file, `benchmark_protocol_round4.json` and `benchmark_protocol_round4.lock` are committed before `benchmark_round4.py` exists. Earlier locks are unchanged (round 1 `826e3e68…94c9f`, round 2 `a06cd2e6…8cc27d`, round 3 `b6365f50…5637d4`; round-3 results commit `3cc0d9c01ee30ccdf28006b97614c36bcc7b5760`).

Round-4 protocol hash: `0b36a13189b51b91fedacf61336eb42a50d7daf077f00f80a3b9e4561803082f`. Seed 20261028 (new; no round-3 trial is reused).

What the author knew when writing this: rounds 1 to 3, including the round-3 exploratory observation that accuracy at steps 2 to 5 exceeded step 25 (never tested), the round-3 spurious-state fraction (397 of 400 far from every stored pattern), 344 fixed points and 56 still-moving trials, and the complex128 timings at one and eight threads. No round-4 measurement existed.

## Questions and pre-declared rules

1. **Early-step peak and early stopping** (random patterns, bit-flip cues q = 0.1 and 0.3, N = 64, 128, 256, P/N = 1/8, 3/16, 1/4: 18 primary cells fixed in advance). Selection seeds (100 trials per cell) are used only to choose one fixed stopping step; every reported number comes from separate held-out seeds (300 trials per cell). Four comparisons, each with its own verdict (paired sign-flip test, BH inside the comparison, pooled equal-weight mean with bootstrap CI):
   - E1a: mean accuracy over steps 2 to 5 minus accuracy at step 25;
   - E1b: the same minus accuracy at step 1;
   - E2a: accuracy at the selected step s* minus step 25;
   - E2b: accuracy at s* minus the cue.
   Supported: at least 75% of cells significantly higher, pooled CI lower bound > 0 and pooled mean >= 0.01. Rejected: at most 25% of cells higher, or pooled CI upper bound <= 0. Otherwise partial/inconclusive.
2. **Spurious states** (random patterns, N = 64, 128, 256, q = 0, 0.1, 0.3, P/N = 1/8, 3/16, 1/4, 100 trials per cell, final state at step 100). Pre-declared categories with precedence: stored (within 0.10 of a stored pattern or its negation), mixture_3, mixture_5 (sign of the sum of the 3 or 5 most-overlapping stored patterns, within 0.15), partial_overlap (max overlap >= 0.5), unstructured. Hypotheses Sp1 to Sp5 with Wilson or bootstrap rules in the JSON (mixture states, unstructured, partial overlap, fixed points, lean toward the mean pattern). Positive controls and a random-state null check the classifier; if they fail the verdicts are reported "harness_invalid".
3. **Threads** (complex128, N = 2048 and 4096 as requested, plus 8192 as a declared extension; 1, 2, 3, 4 and default BLAS threads; 2 runs in opposite order; load-gated as before). Hypotheses T1 (speed-up), T2 (monotone), T3 (repeatability), T4 (is the original 2.2 to 3.5 s claim for N = 16384 reachable with at most 4 threads; an extrapolation from 4096 to 8192 to 16384, labelled as such).

## Not part of this plan

No quantum-hardware, QEC or quantum-advantage claim. No upload anywhere. complex128 at N = 16384 is not measured. Timings are machine- and load-dependent.
