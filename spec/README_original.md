# Coherent Sequence Wave Computer

**A reference implementation of wave-state computing built from the central
object of Coherent Sequence theory: the phase-coherence function `C_seq`.**

```
digital bits --encode--> wave-states psi = A * exp(i*theta)
            --interfere--> holographic associative memory
            --readout---> fidelity measured AS C_seq
            --detect----> OES-32 block score declares lock-in events
            --record----> evidence passport (seeded, SHA-256 hashed)
```

## The breakthrough

Lee (2026), *Coherent Sequence Unified Coupled-Field Theory*, proposes
`C_seq = |<exp(i*theta)>|` as the order parameter of a macroscopic coherent
state — and the *Digital-to-Wave* paradigm (Zenodo 21947036) proposes
continuous wave-states, not bits, as the unit of information, with
interference as the natural interaction.

This code makes those two ideas **executable together**:

- **Encode** data as phase patterns (BPSK: bit -> wave phase).
- **Compute** by interference: many patterns superposed in one hologram
  (complex-valued Hopfield network); recall is interference dynamics.
- **Read out** fidelity *as* `C_seq` — the theory's order parameter becomes
  the computer's native observable. The same mathematics that describes a
  coherent state describes a working memory.
- **Detect** lock-in events with the frozen OES-32 block score
  (`S = 0.45*max|x| + 0.35*RMS + 0.20*mean|x|`, threshold `0.50`),
  the same detection layer used across the OES telemetry work.
- **Record** everything in an evidence passport: protocol version, seed,
  parameters, metrics, SHA-256 digest, and explicit limitations.

### What the demo shows (seed 7, reproducible)

- Exact recall of a 256-mode wave-state from **20% phase decoherence**:
  bit accuracy `1.0000`, fidelity `C_seq = 1.0000`.
- Wave-filtering: matched-filter spectrum shows the constructive winner
  (`0.99`) against destructively cancelled rest (`~0.01`).
- OES-32 lock-in event detected (`S = 0.67 >= 0.50`).
- **Disclosed limitation:** a cue for a pattern *never stored* makes the
  memory confabulate — it locks onto the nearest stored pattern, and
  OES-32 fires on energy alone. Energy detection cannot distinguish true
  recall from confident confabulation; the `C_seq` fidelity readout is the
  disambiguator. This is reported, not hidden, and motivates margin-based
  confidence scoring as next work.

## What this is not

- **Not quantum hardware.** A classical simulator of interference-based
  computing, in the spirit of optical/holographic computing.
- **Not a cosmological validation.** The macro-micro bridge is demonstrated
  as a *computational* correspondence, not a physical derivation.
- **Not a claim of advantage.** All signals are synthetic; results are
  seed-dependent. See `evidence_passport.json` limitations.

## Quick start

```bash
pip install -r requirements.txt   # numpy only
python tests.py                   # 6 sanity tests
python demo.py --seed 7           # full pipeline + evidence passport
```

## Layout

```
cseq_wave/
  wave.py      wave-states, C_seq coherence, coherent-receiver slicer
  encode.py    digital<->wave codec (BPSK phase-shift keying)
  memory.py    holographic associative memory (complex Hopfield)
  oes.py       OES-32 block score + event detection (threshold 0.50)
  evidence.py  evidence passport (seeded, SHA-256)
demo.py        end-to-end experiment
tests.py       sanity tests
```

## Reproduce the headline result

```bash
python demo.py --seed 7
# expect: bit accuracy 1.0000, fidelity (C_seq) 1.0000, lock-in detected: True
```

## Related work

- Lee, Mun Seok (2026). *Coherent Sequence Unified Coupled-Field Theory.*
  Preprint, 16 Oct 2026.
- Lee, Mun Seok (2026). *Digital to Wave: Paradigm Shift in Quantum
  Wave-Based Computing and Lab-Grown Diamond Memory.* Zenodo 21947036.
- Lee, Mun Seok (2026). *Digital to Wave: LGD In Situ Wave Memory…*
  Zenodo 21947108.
- Orbit–Wave–OES Lab concept brief (13 Oct 2026): ground-first satellite
  simulator; OES-32 detection layer; evidence passport pattern.

## License

GNU Affero General Public License v3.0 — see `LICENSE`.
If you publish results built on this code, cite it (see `CITATION.cff`).
