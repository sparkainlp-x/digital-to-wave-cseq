# Coherent Sequence Wave Computer (`cseq_wave`)

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

## Boundaries

- **Classical simulation** of interference-based computing, in the spirit of optical or holographic associative memories. Not quantum hardware.
- **Not a validation of the Coherent Sequence theory**, and not a cosmological or physical result. The software shares a formula with the theory, nothing more.
- **Not a claim of advantage** over any other memory or computing method.
- The theory's author has not reviewed or endorsed this software.

## Quick start

```bash
pip install -r requirements.txt        # numpy is the only dependency
python tests.py                        # original script-style checks
pip install -r requirements-dev.txt && python -m pytest   # 30 tests, warnings are errors
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

## Limitations

1. **The OES-32 flag does not detect correct recall.** On the recall trajectory it fired in 200 of 200 runs for stored cues, never-stored cues, uniform-noise cues *and an empty memory* (mean scores 0.98 to 0.99). For a signal in [0, 1] the score is at least the signal's mean, and the trajectory approaches 1 whenever the dynamics settle. It is kept as a convergence flag.
2. **Confabulation.** A cue for a pattern that was never stored gets a confident wrong answer. In 200 runs, 63% converged onto a stored pattern (overlap > 0.9), with mean fidelity 0.10 against the cue's own bits. Nothing available *without* the true target exposes this, so the fidelity readout cannot catch it in a real query. Margin-based confidence is planned for v0.2.
3. **Fidelity is sign-blind.** For BPSK the sliced readout equals |2 x bit accuracy - 1|, so a fully inverted pattern also scores 1.0. Slicing can only raise the score. Always report bit accuracy too.
4. Uniform random codebooks only. Correlated codewords (for example raw text) create spurious attractors.
5. Results depend on the seed. Passport digests cover protocol, seed, parameters and metrics, not the timestamp.

## Related repositories

- [digital-to-wave-testbench](https://github.com/sparkainlp-x/digital-to-wave-testbench): synthetic testbench for the signed-amplitude sine encoding in *Digital to Wave* Part I.
- [digital-to-wave-2-thz-validity](https://github.com/sparkainlp-x/digital-to-wave-2-thz-validity): engineering-validity study of the THz equations in *Digital to Wave* Part II.

This repository is different from both. It models phase-based associative recall, not the sine encoding or the THz interface.

## Sources and acknowledgements

The `C_seq` formula and the wave-state framing come from Mun Seok Lee's work. The memory model, codec, OES-32 use, experiments and code are this project's own choices.

- Lee, Mun Seok (2026). *Coherent Sequence Unified Coupled-Field Theory.* Preprint (October 2026, no DOI known to this project).
- Lee, Mun Seok (2026). *Digital to Wave: Paradigm Shift in Quantum Wave-Based Computing and Lab-Grown Diamond Memory.* Zenodo. https://doi.org/10.5281/zenodo.21947036
- Lee, Mun Seok (2026). *Digital to Wave: LGD In Situ Wave Memory-Human Wave Integrated Interface-Metaverse Hologram- Wave Affective AI.* Zenodo. https://doi.org/10.5281/zenodo.21947108. Cited for context only, none of it is modelled here.

The original agent-instructions spec is kept unchanged in [spec/](spec/), and the first commit reproduces it exactly.

## How to cite

See [CITATION.cff](CITATION.cff). Please also cite Lee (2026) when you refer to the `C_seq` formula.

## License

Copyright (c) 2026 Jean-François Brisson, Spark AI NLP.

This software is available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).
