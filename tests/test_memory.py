import numpy as np
import pytest

from cseq_wave import HolographicMemory, WaveState, bits_to_wave, detect_events, wave_to_bits

N = 256


def build(seed, p=5):
    rng = np.random.default_rng(seed)
    cb = rng.integers(0, 2, size=(p, N))
    mem = HolographicMemory(N)
    for b in cb:
        mem.store(bits_to_wave(b))
    mem.finalize()
    return rng, cb, mem


@pytest.mark.parametrize("seed", [7, 42])
def test_recall_from_20pct_decoherence(seed):
    rng, cb, mem = build(seed)
    tgt = seed % 5
    tw = bits_to_wave(cb[tgt])
    rec, traj = mem.recall(tw.decohere(rng, flip_prob=0.20), steps=25)
    acc = float(np.mean(wave_to_bits(WaveState(np.ones(N), rec)) == cb[tgt]))
    assert acc == 1.0
    assert mem.fidelity(rec, tw.phase) == 1.0
    assert detect_events(traj)[1]
    assert int(np.argmax(mem.overlaps(rec))) == tgt
    assert np.max(mem.overlaps(rec)) == pytest.approx(1.0)


def test_store_rejects_wrong_size_and_diagonal_is_zero():
    mem = HolographicMemory(8)
    with pytest.raises(ValueError):
        mem.store(bits_to_wave(np.zeros(4, dtype=int)))
    mem.store(bits_to_wave(np.array([0, 1] * 4)))
    mem.finalize()
    assert np.allclose(np.diag(mem.weights), 0)


def test_decohered_cue_overlap_is_about_one_minus_2p_squared():
    rng, cb, mem = build(3)
    cue = bits_to_wave(cb[0]).decohere(rng, flip_prob=0.20)
    assert mem.overlaps(cue.phase)[0] == pytest.approx(0.36, abs=0.1)


def test_fidelity_is_blind_to_global_flip():
    mem = HolographicMemory(4)
    target = np.array([0, np.pi, np.pi, 0])
    inverted = target + np.pi
    assert mem.fidelity(inverted, target) == pytest.approx(1.0)
    assert np.all(wave_to_bits(WaveState(np.ones(4), inverted)) != wave_to_bits(WaveState(np.ones(4), target)))


def test_fidelity_equals_abs_2acc_minus_1_for_bpsk():
    rng = np.random.default_rng(5)
    target = rng.integers(0, 2, N)
    rec = target.copy()
    rec[:32] ^= 1
    acc = np.mean(rec == target)
    mem = HolographicMemory(N)
    assert mem.fidelity(rec * np.pi, target * np.pi) == pytest.approx(abs(2 * acc - 1))


def test_slicing_never_lowers_fidelity():
    rng = np.random.default_rng(9)
    target = rng.integers(0, 2, 64) * np.pi
    rec = target + rng.normal(0, 0.4, 64)
    mem = HolographicMemory(64)
    assert mem.fidelity(rec, target) >= mem.fidelity(rec, target, slice_output=False)


def test_recall_without_tracking_returns_empty_trajectory():
    rng, cb, mem = build(1)
    _, traj = mem.recall(bits_to_wave(cb[0]), steps=5, track_coherence=False)
    assert traj.size == 0


def test_confabulation_never_stored_cue_is_pulled_to_a_stored_pattern():
    """Honest limitation: a never-stored cue gets a confident wrong answer.

    The OES-32 flag fires, the recalled state matches a stored pattern
    almost exactly, and nothing available without the true target says the
    answer is wrong. Only fidelity against the cue's own (unstored) bits
    reveals it, and that target is not known in a real query.
    """
    hits = 0
    for seed in range(20):
        rng, cb, mem = build(100 + seed)
        ghost = rng.integers(0, 2, N)
        rec, traj = mem.recall(bits_to_wave(ghost), steps=25)
        assert detect_events(traj)[1]  # flag fires regardless
        assert mem.fidelity(rec, ghost * np.pi) < 0.35  # wrong vs its own bits
        if np.max(mem.overlaps(rec)) > 0.9:
            hits += 1
    assert hits >= 8  # most converge confidently onto a stored pattern
