"""Digital <-> wave codec ("digital to wave", literally).

Bits are mapped to wave-states by binary phase-shift keying (BPSK):

    bit 0 -> phase 0
    bit 1 -> phase pi

so an N-bit message becomes N wave modes of unit amplitude. Decoding
recovers bits from phases: phase near pi -> 1, else 0. Phase is the
information carrier; amplitude is the confidence carrier.
"""

import numpy as np


def bits_to_wave(bits):
    """Encode a bit array as a WaveState (BPSK: 0->0, 1->pi)."""
    from .wave import WaveState

    from .validation import validate_bits

    bits = validate_bits(bits)
    phase = bits.astype(float) * np.pi
    return WaveState(np.ones_like(phase), phase)


def wave_to_bits(wavestate):
    """Decode a WaveState back to bits (hard decision on phase)."""
    from .validation import validate_phase

    phase = np.angle(np.exp(1j * validate_phase(wavestate.phase, name="wave phase")))  # wrap to [-pi, pi]
    return (np.abs(phase) > np.pi / 2).astype(int)


def slice_bpsk(phase):
    """Coherent-receiver slicer: quantize phases to the BPSK constellation.

    Real coherent receivers slice the received wave to the nearest
    constellation point before readout. Without this step the
    continuous-phase dynamics can rest slightly off-constellation and
    the raw-phase fidelity can differ from what the receiver sees. Slicing
    can RAISE or LOWER fidelity (e.g. recalled [1.5, 1.7] vs target [0, 0]:
    unsliced ~0.995, sliced 0.0), so report which one you use.
    """
    from .validation import validate_phase

    phase = validate_phase(phase)
    wrapped = np.angle(np.exp(1j * phase))
    return np.where(np.abs(wrapped) > np.pi / 2, np.pi, 0.0)


def text_to_bits(text):
    """UTF-8 text -> bit array (MSB first per byte)."""
    raw = text.encode("utf-8")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits


def bits_to_text(bits):
    """Bit array -> UTF-8 text (truncates to whole bytes)."""
    from .validation import validate_bits

    bits = validate_bits(bits)
    n = (len(bits) // 8) * 8
    if n == 0:
        return ""
    raw = np.packbits(bits[:n]).tobytes()
    return raw.decode("utf-8", errors="replace")
