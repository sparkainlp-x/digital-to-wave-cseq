import numpy as np
import pytest

from cseq_wave import WaveState, bits_to_text, bits_to_wave, coherence, text_to_bits, wave_to_bits
from cseq_wave.encode import slice_bpsk
from cseq_wave.wave import alignment


def test_uniform_phase_coherence_is_one():
    assert coherence(np.ones(64), np.zeros(64)) == pytest.approx(1.0, abs=1e-12)


def test_random_phase_coherence_is_small():
    rng = np.random.default_rng(0)
    assert coherence(np.ones(4096), rng.uniform(-np.pi, np.pi, 4096)) < 0.05


def test_zero_amplitude_coherence_is_zero():
    assert coherence(np.zeros(4), np.zeros(4)) == 0.0


def test_amplitude_weighting_ignores_dark_modes():
    amp = np.array([1.0, 1.0, 0.0, 0.0])
    ph = np.array([0.0, 0.0, np.pi, np.pi])
    assert coherence(amp, ph) == pytest.approx(1.0)


def test_wavestate_validation():
    with pytest.raises(ValueError):
        WaveState(np.ones(3), np.zeros(4))
    with pytest.raises(ValueError):
        WaveState(-np.ones(3), np.zeros(3))
    with pytest.raises(ValueError):
        WaveState(np.ones(3), np.zeros(3)).interfere(WaveState(np.ones(4), np.zeros(4)))


@pytest.mark.parametrize("text", ["Coherent Sequence", "", "é ψ ∞"])
def test_text_roundtrip(text):
    assert bits_to_text(text_to_bits(text)) == text


def test_bpsk_roundtrip_and_validation():
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    assert np.array_equal(wave_to_bits(bits_to_wave(bits)), bits)
    with pytest.raises(ValueError):
        bits_to_wave([0, 2])


def test_interference_constructive_and_destructive():
    w1 = WaveState(np.ones(8), np.zeros(8))
    assert np.allclose(w1.interfere(w1).amplitude, 2.0)
    w3 = WaveState(np.ones(8), np.full(8, np.pi))
    assert np.allclose(w1.interfere(w3).amplitude, 0.0, atol=1e-12)


def test_decohere_flip_fraction_and_noop():
    rng = np.random.default_rng(1)
    w = bits_to_wave(np.zeros(20000, dtype=int))
    assert np.array_equal(w.decohere(rng).phase, w.phase)
    flipped = wave_to_bits(w.decohere(rng, flip_prob=0.2))
    assert abs(flipped.mean() - 0.2) < 0.01


def test_slicer_maps_to_constellation():
    out = slice_bpsk([0.1, -0.1, 3.0, -3.0, 2 * np.pi + 0.2])
    assert np.allclose(out, [0, 0, np.pi, np.pi, 0])


def test_alignment_is_relative_phase_coherence():
    a = np.array([0.0, np.pi, 0.0, np.pi])
    assert alignment(a, a) == pytest.approx(1.0)
    assert alignment(a + 0.7, a) == pytest.approx(1.0)  # blind to global phase
