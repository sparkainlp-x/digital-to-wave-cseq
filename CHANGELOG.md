# Changelog

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
- Documented that the fidelity readout is sign-blind (equals |2 x bit accuracy - 1| for BPSK) and that slicing can only raise it.
- Documented that `overlaps()` on a decohered cue gives about (1 - 2p)^2. The demo now also prints the recalled-state overlaps.
- Relabelled the OES-32 trajectory result as a convergence flag. The demo adds never-stored and uniform-noise controls.
- Evidence passports convert numpy types, reject NaN, and carry an `evidence_label: SYNTHETIC`.
- The demo exits non-zero if exact recall fails (the earlier variant always printed PASS).
- Added seeded capacity and control sweeps (`experiments/`, `results/`).
- Added a pytest suite (30 tests, warnings are errors) and CI on Python 3.10 to 3.13.
- Added README, CITATION.cff, .zenodo.json and the authorship options in `authorship/`.
