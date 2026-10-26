# Coherent Sequence Wave Computer (`cseq_wave`)

Software DOI for v0.3.0: [10.5281/zenodo.23529123](https://doi.org/10.5281/zenodo.23529123) (all versions: [10.5281/zenodo.23529122](https://doi.org/10.5281/zenodo.23529122)).

A small, classical Python reference implementation of phase-based associative memory. It uses the phase-coherence function written as `C_seq = |<exp(i*theta)>|` in Mun Seok Lee's *Coherent Sequence Unified Coupled-Field Theory* (2026 preprint) as its recall-fidelity readout.

**Evidence label: SYNTHETIC.** Every number below comes from seeded software runs on random bit patterns. Nothing here is measured on hardware.

```
digital bits --encode--> wave-states psi = A * exp(i*theta)   (BPSK)
            --store----> one complex Hopfield coupling matrix (interference)
            --recall---> s <- exp(i * arg(W s))
            --readout--> fidelity = |<exp(i(recalled - target))>|  (the C_seq formula)
            --flag-----> OES-32 block score on the recall trajectory (convergence flag)
            --record---> evidence passport (seed, parameters, metrics, SHA-256)
```

## v0.2.0: recall verdicts (SYNTHETIC)

`assess_recall` returns a fail-closed verdict. VERIFIED_RECALL needs three things: the raw cue's matched-filter overlap must clear a threshold calibrated on never-stored cues, the recall must reach fidelity and bit accuracy of at least 0.99, and a separate re-computation must agree with the reported numbers. It also works without the true answer, by checking the recall against the stored pattern the cue pointed at. Since the verification-contract fix, every gate must pass on both the reported and the independently recomputed numbers, and the calibration must be a `cseq-cal/1` record bound to the exact stored patterns (`memory_digest`); legacy `{"tau_cue_overlap": x}` dicts give WITHHELD. See CHANGELOG for compatibility changes.

On 240 trials (4 patterns, 20% flips, 0.3 rad jitter), the cue overlap averaged 0.324 for stored cues (minimum 0.170) and 0.0087 for never-stored cues (q0.99 0.029). After recall, the winner margins overlap (stored down to 0.395, never-stored up to 0.973), so the OES-32 flag and the post-recall spectrum cannot do this job. With tau at the null q0.99, about 1.3% of never-stored cues still pass.

```bash
python demo_verified.py --seed 7          # stored cue VERIFIED_RECALL, never-stored cue CONFABULATION_SUSPECTED
python experiments/cue_gate_sweep.py      # regenerates results/cue_gate.{json,md}
```

Limits: stored cues with 40% or more flips are mostly rejected (56% verified at 40%, 9% at 45%). A cue mixing two stored patterns passes the cue gate, and is only caught because recall then fails. With 20 patterns and 0.3 rad jitter, recall itself fails (2% verified), and the verdict reports that. The claims screen is a pattern list, not a guarantee. See [results/cue_gate.md](results/cue_gate.md).

## Boundaries

- **Classical simulation** of interference-based computing, in the spirit of optical or holographic associative memories. Not quantum hardware.
- **Not a validation of the Coherent Sequence theory**, and not a cosmological or physical result. The software shares a formula with the theory, nothing more.
- **Not a claim of advantage** over any other memory or computing method.
- The theory's author has not reviewed or endorsed this software.

## Quick start

```bash
pip install -r requirements.txt        # numpy is the only dependency
python tests.py                        # original script-style checks
pip install -r requirements-dev.txt && python -m pytest   # 52 tests, warnings are errors
python demo.py --seed 7
python experiments/capacity_sweep.py   # regenerates results/sweeps.{json,md}
```

## Demo result (SYNTHETIC, N = 256 modes, 5 stored patterns, 20% phase flips)

| seed | bit accuracy | fidelity (sliced / unsliced) | cue overlap with source | recalled overlap | OES-32 score |
|---|---:|---:|---:|---:|---:|
| 7 | 1.0000 | 1.0000 / 1.0000 | 0.27 | 1.00 | 0.9911 |
| 42 | 1.0000 | 1.0000 / 1.0000 | 0.32 | 1.00 | 0.9917 |

A cue with 20% of its phases flipped is pulled back to the exact stored pattern. The cue's own overlap with its source is about (1 - 2p)^2 = 0.36, and it rises to 1.00 after recall. The original spec expected the cue overlap to read 0.99 and the OES-32 score 0.9817. Neither matches: 0.99 is the *recalled* state's overlap, and 0.98 is what the flag scores for any converged run.

## Capacity sweep (SYNTHETIC, 100 trials per row)

| patterns | load P/N | bit accuracy | exact recall |
|---:|---:|---:|---:|
| 1 | 0.004 | 1.0000 | 1.00 |
| 5 | 0.019 | 1.0000 | 1.00 |
| 10 | 0.039 | 0.9998 | 0.99 |
| 15 | 0.059 | 1.0000 | 1.00 |
| 20 | 0.078 | 0.9816 | 0.77 |
| 25 | 0.098 | 0.9922 | 0.83 |
| 30 | 0.117 | 0.8793 | 0.29 |
| 40 | 0.156 | 0.7867 | 0.06 |
| 50 | 0.195 | 0.7065 | 0.00 |
| 60 | 0.234 | 0.6570 | 0.00 |

Exact recall holds up to about 15 patterns (load 0.06) and breaks down past about 25 (load 0.10). Small reversals between rows are sampling noise. The full table, including fidelity, is in [results/sweeps.md](results/sweeps.md).

## Benchmarks

- `benchmark.py` (supplied unchanged): quick reverse / scaling / capacity checks.
- `benchmark_rigorous.py`: locked analysis plan (`benchmark_protocol.json`), direct timings with bootstrap CIs in two BLAS-thread modes, a log-log fit on a range fixed in advance, capacity at the same P/N across N = 32 to 256 with null baselines, and adversarial reverse tests. `python benchmark_rigorous.py --quick` (CI) or `--full --json report_rigorous.json` (manual).

N = 16384 was never measured; times there are labelled extrapolations. Timings are machine- and load-dependent. See [docs/benchmark.md](docs/benchmark.md) for the numbers, the comparison with the original claims, and the limitations. SYNTHETIC classical software, no quantum-hardware claim.

## Limitations

1. **The OES-32 flag does not detect correct recall** (v0.2.0 adds the calibrated cue gate for that). On the recall trajectory it fired in 200 of 200 runs for stored cues, never-stored cues, uniform-noise cues *and an empty memory* (mean scores 0.98 to 0.99). For a signal in [0, 1] the score is at least the signal's mean, and the trajectory approaches 1 whenever the dynamics settle. It is kept as a convergence flag.
2. **Confabulation.** A cue for a pattern that was never stored gets a confident wrong answer. In 200 runs, 63% converged onto a stored pattern (overlap > 0.9), with mean fidelity 0.10 against the cue's own bits. Nothing available *without* the true target exposes this, so the fidelity readout cannot catch it in a real query. v0.2.0 flags most such cues with the calibrated cue gate (above).
3. **Fidelity is sign-blind.** For BPSK the sliced readout equals |2 x bit accuracy - 1|, so a fully inverted pattern also scores 1.0. Slicing may raise or lower the score (recalled [1.5, 1.7] vs target [0, 0]: unsliced about 0.995, sliced 0.0). Sliced fidelity, unsliced fidelity and bit accuracy are three distinct numbers. Always report bit accuracy too.
4. Uniform random codebooks only. Correlated codewords (for example raw text) create spurious attractors.
5. Results depend on the seed. Passport digests cover protocol, seed, parameters and metrics, not the timestamp.

## Related repositories

- [digital-to-wave-testbench](https://github.com/sparkainlp-x/digital-to-wave-testbench): synthetic testbench for the signed-amplitude sine encoding in *Digital to Wave* Part I.
- [digital-to-wave-2-thz-validity](https://github.com/sparkainlp-x/digital-to-wave-2-thz-validity): engineering-validity study of the THz equations in *Digital to Wave* Part II.

This repository is different from both. It models phase-based associative recall, not the sine encoding or the THz interface.

## Sources and acknowledgements

The `C_seq` formula and the wave-state framing come from Mun Seok Lee's work, as supplied to this project. The verification layer, experiments, results and tests are ours; the base package is derived from the one received under his name (see Provenance below).

- Lee, Mun Seok (2026). *Coherent Sequence Unified Coupled-Field Theory.* Preprint (October 2026, no DOI known to this project).
- Lee, Mun Seok (2026). *Digital to Wave: Paradigm Shift in Quantum Wave-Based Computing and Lab-Grown Diamond Memory.* Zenodo. https://doi.org/10.5281/zenodo.21947036
- Lee, Mun Seok (2026). *Digital to Wave: LGD In Situ Wave Memory-Human Wave Integrated Interface-Metaverse Hologram- Wave Affective AI.* Zenodo. https://doi.org/10.5281/zenodo.21947108. Cited for context only, none of it is modelled here.

The original agent-instructions spec is kept unchanged in [spec/](spec/), and the first commit reproduces it exactly.

## Provenance

The first commit of this repository is a package received under Mun Seok Lee's name, kept unchanged in [spec/](spec/). Its metadata lists him as sole author under AGPL-3.0-only. We could not independently verify who wrote it. This repository is derived from it. Compared with that first commit (line similarity from a diff):

| file | status |
|---|---|
| `cseq_wave/wave.py` (includes `coherence` and `alignment`, the C_seq readout) | byte-identical |
| `tests.py` | lightly edited (about 94% line similarity) |
| `cseq_wave/encode.py`, `cseq_wave/oes.py` | edited (about 83% and 81%) |
| `cseq_wave/memory.py`, `cseq_wave/evidence.py`, `cseq_wave/__init__.py` | edited and extended (about 71%, 60%, and new exports) |
| `demo.py` | largely rewritten (about 28% line similarity) |
| `cseq_wave/calibrate.py`, `claims.py`, `validation.py`, `verdict.py`, `verify.py`, `experiments/`, `results/`, `tests/`, CI | added by this project (the verification layer was adapted from a later handoff package and corrected) |

[CHANGELOG.md](CHANGELOG.md) lists every change. Mun Seok Lee is not an author of this release, and he has not reviewed or endorsed this software. The *Coherent Sequence Unified Coupled-Field Theory* preprint has no DOI known to this project, so it is unpublished as far as we can tell; we implemented the C_seq formula from the supplied material and have not checked it against the preprint itself.

## How to cite

See [CITATION.cff](CITATION.cff). Please also cite Lee (2026) when you refer to the `C_seq` formula.

## License

Copyright (c) 2026 Jean-François Brisson, Spark AI NLP.

This software is available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).
