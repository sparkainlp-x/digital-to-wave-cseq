"""Coherent Sequence Wave Computer (cseq_wave).

A reference implementation of wave-state computing built from the central
object of Lee (2026), "Coherent Sequence Unified Coupled-Field Theory":
the phase-coherence function C_seq.

Pipeline
--------
digital bits --encode--> wave-states psi = A * exp(i*theta)
            --interfere--> holographic associative memory (complex Hopfield)
            --readout---> fidelity measured AS C_seq
            --detect----> OES-32 block score on the recall transient
            --record----> evidence passport (seeded, hashed, reproducible)

What this is
-----------
A *classical* simulator of interference-based computing (in the spirit of
optical/holographic computing). It is NOT quantum hardware, NOT a
cosmological simulation, and NOT a validation of the Coherent Sequence
theory. All signals are synthetic.

What it demonstrates
--------------------
That C_seq -- the theory's order parameter for a macroscopic coherent
state -- is computationally meaningful: encode data as phase patterns,
compute by interference, and read the answer's fidelity back as C_seq.
The same mathematics that describes a coherent state describes a working
memory. That is the macro-micro bridge, made executable.
"""

from .wave import WaveState, coherence
from .encode import bits_to_wave, wave_to_bits, text_to_bits, bits_to_text
from .memory import HolographicMemory
from .oes import oes32_score, detect_events
from .evidence import evidence_passport

__all__ = [
    "WaveState",
    "coherence",
    "bits_to_wave",
    "wave_to_bits",
    "text_to_bits",
    "bits_to_text",
    "HolographicMemory",
    "oes32_score",
    "detect_events",
    "evidence_passport",
]

__version__ = "0.1.0"
