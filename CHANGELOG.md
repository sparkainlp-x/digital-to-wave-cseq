# Changelog

## Unreleased

- Added round 2 (`benchmark_round2.py`, `benchmark_protocol_round2.json`/`.lock`, plan in `docs/round2_plan.md`, raw reports in `results/round2/`): bit-flip and phase-noise denoising test with paired bootstrap CIs and nearest-stored baselines (recall denoises only at low load P/N <= 1/16 and is never better than a nearest-stored lookup), load-controlled timing batches (external-CPU gate, discarded attempts recorded), and a memory-safe complex64 recall timed at N = 16384 (complex128 at 16384 remains an estimate). Manual full workflow and a CI quick-mode determinism check cover it.
- Added `benchmark_rigorous.py` with a locked protocol (`benchmark_protocol.json` / `.lock`): direct scaling timings (N = 64 to 8192) with bootstrap CIs in two thread modes, an a-priori fit range with a CI for alpha, local slopes, a hold-out check of the extrapolation, a P/N x N x sigma capacity grid with null baselines, adversarial reverse tests, a deterministic hash that excludes timings, tests and a manual full-run workflow. `docs/benchmark.md` now reports the measured numbers and compares them with the original claims.
- Added `benchmark.py` (reverse round trips, scaling, capacity knee; supplied unchanged), `docs/benchmark.md` (measured vs extrapolated, determinism, two bugs found), `experiments/measure_large_n.py` (direct timing to N = 4096) and `tests/test_benchmark.py`. Timings at N = 4096 and 16384 from `benchmark.py` are power-law extrapolations, not measurements; see the doc for how they compare with direct measurements.
- Documentation only: replaced the statement that the code is "this project's own choices" with a Provenance section in the README. It states which files are byte-identical to, edited from, or added to the package received under Mun Seok Lee's name, that he is not an author of this release and has not endorsed it, and that the C_seq preprint has no known DOI.

## 0.3.0 (2026-10-21)

- Archived on Zenodo: version DOI [10.5281/zenodo.23529123](https://doi.org/10.5281/zenodo.23529123), concept DOI [10.5281/zenodo.23529122](https://doi.org/10.5281/zenodo.23529122).

All results remain SYNTHETIC (classical simulation only).

- `verify_claims` is fail-closed: empty dicts, missing required metrics (either side), unknown reported metrics, bool/str/NaN/inf values, out-of-range values, invalid winner indices and invalid tolerances all fail with an explicit `reason`. Return shape `(ok, checks)` is unchanged; new optional `n_patterns` argument bounds the winner index.
- New `cseq_wave/validation.py` (phase and binary vector validation, no broadcasting, binary values checked before the integer cast), wired into `encode.py`, `memory.py`, `verify.py` and `verdict.py`.
- `assess_recall` gates on BOTH the reported and the recomputed cue overlap, fidelity and bit accuracy; agreement within tolerance never rescues a value below its threshold. `target_index` must be an in-range integer (no truncation).
- Calibration is bound to the hologram (`schema_version` `cseq-cal/1`, `memory_digest`, mode/pattern counts, null protocol, noise, sample count, quantile, seed, threshold). New `HolographicMemory.digest()` and `calibrate.validate_calibration`.
- Corrected the docs: slicing may raise or lower fidelity.

### Compatibility changes
- Legacy calibration dicts such as `{"tau_cue_overlap": 0.03}` now give `WITHHELD`; re-run `calibrate_cue_threshold` on the finalized memory. Calibrations go stale when patterns are added.
- `bits_to_wave` / `bits_to_text` reject multidimensional input (previously flattened silently) and fractional values (previously truncated by the int cast). Empty bit vectors are still accepted by the text codec.
- Phase inputs with NaN/inf, complex, boolean or wrongly shaped values raise `ValueError`/`TypeError` in the memory and verifier, and give `WITHHELD` in `assess_recall`.

## 0.2.0 (unreleased)

Verification layer, adapted from a v0.2.0 handoff package and corrected.

- `calibrate.py`: cue-overlap threshold tau = q0.99 of never-stored cues on the hologram in use.
- `verify.py`: separate-path re-computation of reported metrics. A mismatch withholds the verdict. Fixed: target bits are now read from wrapped phases.
- `claims.py`: regular-expression screen for report text, with evidence tags and a boundary statement. Fixed: negation now only counts inside its own clause ("not slow: a quantum computer" was passing), and the org's banned terms were added.
- `verdict.py`: fail-closed verdicts VERIFIED_RECALL / UNVERIFIED_RECALL / CONFABULATION_SUSPECTED / WITHHELD. Fixed: a bit-accuracy gate was added, because the handoff version returned VERIFIED_RECALL for a fully inverted recall (bit accuracy 0.0). New: a target-free mode, so the gate can be used when the true answer is unknown.
- `demo_verified.py`: verified demo for seeds 7 and 42. The never-stored control is assessed without a target. (The handoff demo used the recalled spectrum's winner as the "target", which is circular.)
- `experiments/cue_gate_sweep.py` and `results/cue_gate.md`: SYNTHETIC separation, decoherence and capacity results, plus a mixture-cue blind spot.
- Tests: 30 -> 52. All v0.1.0 fixes kept.

## 0.1.0 (unreleased)

- Initial code exactly as given in the agent-instructions spec (first commit, kept in `spec/`).
- Removed the bogus guarded import block from `tests.py`.
- Corrected the `recall()` docstring: the trajectory is step-to-step self-consistency, a convergence measure.
- Documented that the fidelity readout is sign-blind (equals |2 x bit accuracy - 1| for BPSK) and how slicing affects it (corrected in Unreleased: slicing may raise or lower it).
- Documented that `overlaps()` on a decohered cue gives about (1 - 2p)^2. The demo now also prints the recalled-state overlaps.
- Relabelled the OES-32 trajectory result as a convergence flag. The demo adds never-stored and uniform-noise controls.
- Evidence passports convert numpy types, reject NaN, and carry an `evidence_label: SYNTHETIC`.
- The demo exits non-zero if exact recall fails (the earlier variant always printed PASS).
- Added seeded capacity and control sweeps (`experiments/`, `results/`).
- Added a pytest suite (30 tests, warnings are errors) and CI on Python 3.10 to 3.13.
- Added README, CITATION.cff, .zenodo.json and the authorship options in `authorship/`.
