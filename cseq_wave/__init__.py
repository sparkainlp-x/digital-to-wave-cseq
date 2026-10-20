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

What it shows
-------------
That the same function used as C_seq in the theory, |<exp(i*theta)>|, can
be used as a recall-fidelity readout in a classical phase-based associative
memory. This is a shared formula, not a physical link between the theory
and this software. When the target is known, the readout equals
|2 * bit_accuracy - 1| for BPSK patterns; it cannot score a recall when the
target is unknown. Results are SYNTHETIC.
"""

from .wave import WaveState, coherence
from .encode import bits_to_wave, wave_to_bits, text_to_bits, bits_to_text
from .memory import HolographicMemory
from .oes import oes32_score, detect_events
from .evidence import evidence_passport
from .calibrate import calibrate_cue_threshold
from .verify import recompute_all, verify_claims
from .claims import BOUNDARY_STATEMENT, screen_text, tag
from .verdict import assess_recall

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
    "calibrate_cue_threshold",
    "recompute_all",
    "verify_claims",
    "screen_text",
    "tag",
    "BOUNDARY_STATEMENT",
    "assess_recall",
]

__version__ = "0.2.0"
