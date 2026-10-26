# Round-3 plan (written before any round-3 experiment was run)

Status: **plan only**. This file, `benchmark_protocol_round3.json` and `benchmark_protocol_round3.lock` are committed before `benchmark_round3.py` exists. Round-1 and round-2 protocols and locks are unchanged (hashes `826e3e6870f9e3b95b605e2e70701067b970909f196b739d8e662dd06f094c9f` and `a06cd2e65d18d2eac28b57a1a74e8291ab5b81b7f31f28f7637685d8f8bcc27d`; round-2 results at commit `64406dd792925b4307f8193d6a6ea4d88cddb89b`).

Round-3 protocol hash: `b6365f500ea5a6c1e60c8eb4db1200c6c5980e027dd409e3895a1676ff5637d4`.

What the author knew when writing this: the round-1 and round-2 results, the cache sizes the VM reports (L1d 48 KiB and L2 2 MiB per core, L3 320 MiB shared), the algebra that BPSK patterns make the couplings real so the dynamics are synchronous +-1 Hopfield dynamics, the classical cross-talk prediction Q(sqrt((N-1)/(P-1))) for one-step errors, and that exactly orthogonal patterns have zero cross-talk on an exact cue. No round-3 measurement existed.

## Questions and what would support or reject each answer

1. **Why does recall get worse than the cue at high load?** On cells where recall is significantly worse than the cue (Benjamini-Hochberg adjusted, random patterns, bit flips), pool the trials and test four hypotheses, each with a stated rule in the JSON:
   - (a) cross-talk: supported if the measured one-step error of an exact stored pattern matches the Gaussian cross-talk prediction in at least 75% of testable cells AND orthogonal (Hadamard) patterns remove the degradation in at least 80% of those cells with at least +0.05 accuracy; rejected if Hadamard patterns degrade in at least half of them.
   - (b) spurious states: fraction of final states (step 100) farther than tau = 0.10 from every stored pattern and its negation; supported if the Wilson lower bound is at least 0.50, rejected if the upper bound is at most 0.10.
   - (c) sign mirror: fraction of final states near a negated stored pattern; supported if the Wilson lower bound is at least 0.10, rejected if the upper bound is at most 0.05.
   - (d) dynamics: accuracy at steps 1, 2, 5, 10, 25, 50, 100; truncation supported if accuracy changes by at least 0.02 between steps 25 and 100 and at least a quarter of trials change bits; rejected if the change is below 0.01 and at most 10% of trials change; oscillation (period-2 cycles at step 100) supported/rejected by Wilson bounds 0.10 / 0.05.
   Intermediate outcomes are reported as partial/inconclusive.
2. **Correlated patterns** (pairwise correlation rho in 0, 0.1, 0.3, with rho = 0 an independent replication of round 2): capacity knee and denoising verdicts per rho. Supported: the number of BH-adjusted "denoises" cells at rho = 0.3 is at least 6 lower than at rho = 0 (of 36); rejected if it is not lower.
3. **Multiplicity**: rerun the round-2 denoising profile with per-trial output (same seed; the stored round-2 cells must reproduce exactly), sign-flip permutation p-values, Holm and Benjamini-Hochberg at 0.05; report which of the 61 / 37 / 10 raw verdicts survive.
4. **Cache microbenchmark** for the step above N^2: matvec bandwidth versus working set (complex128 and complex64) and a plain streaming-read curve, load-gated like round 2. Reported as "consistent with" a reported cache size (bracketing rule) or with the streaming-read drop, or inconclusive.
5. **complex128 timed directly at 8192, 10240, 12288, 14336** with a chunked 16 N^2-byte matrix (peak at most 4 GiB and half of available memory), then a labelled extrapolation to 16384.

## Not part of this plan

No quantum-hardware, QEC or quantum-advantage claim. No upload anywhere. Timings are machine-dependent.
