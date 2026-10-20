# Changelog

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
