"""Test suite for cseq_wave. Run:  python tests.py"""

import sys
sys.path.insert(0, ".")

import numpy as np

from cseq_wave import (
    HolographicMemory,
    WaveState,
    bits_to_wave,
    coherence,
    detect_events,
    evidence_passport,
    oes32_score,
    text_to_bits,
    bits_to_text,
    wave_to_bits,
)


def t(name):
    print(f"  [ok] {name}")


# 1. coherence of a uniform-phase field is 1
a = np.ones(64)
assert abs(coherence(a, np.zeros(64)) - 1.0) < 1e-12
t("C_seq(uniform phase) == 1")

# 2. coherence of random phases is ~0
rng = np.random.default_rng(0)
c = coherence(np.ones(4096), rng.uniform(-np.pi, np.pi, 4096))
assert c < 0.05, c
t(f"C_seq(random phases) ~= 0 ({c:.4f})")

# 3. text <-> bits roundtrip
assert bits_to_text(text_to_bits("Coherent Sequence")) == "Coherent Sequence"
t("text codec roundtrip")

# 4. bits <-> wave roundtrip
bits = np.array([0, 1, 1, 0, 1, 0, 0, 1])
assert np.array_equal(wave_to_bits(bits_to_wave(bits)), bits)
t("BPSK wave codec roundtrip")

# 5. interference: aligned waves amplify, opposed waves cancel
w1 = WaveState(np.ones(8), np.zeros(8))
w2 = WaveState(np.ones(8), np.zeros(8))
assert np.allclose(w1.interfere(w2).amplitude, 2.0)
w3 = WaveState(np.ones(8), np.full(8, np.pi))
assert np.allclose(w1.interfere(w3).amplitude, 0.0, atol=1e-12)
t("interference constructive/destructive")

# 6. holographic recall from 20% decoherence (seeds 7 and 42)
for seed in (7, 42):
    r = np.random.default_rng(seed)
    mem = HolographicMemory(256)
    codebook = r.integers(0, 2, size=(5, 256))
    for m in range(5):
        mem.store(bits_to_wave(codebook[m]))
    mem.finalize()
    tgt = seed % 5
    tw = bits_to_wave(codebook[tgt])
    rec, traj = mem.recall(tw.decohere(r, flip_prob=0.20), steps=25)
    bits_hat = wave_to_bits(WaveState(np.ones(256), rec))
    acc = float(np.mean(bits_hat == codebook[tgt]))
    fid = mem.fidelity(rec, tw.phase)
    score, locked = detect_events(traj)
    assert acc == 1.0, (seed, acc)
    assert fid == 1.0, (seed, fid)
    assert locked, (seed, score)
    ov = mem.overlaps(tw.decohere(r, flip_prob=0.0).phase)
    assert np.argmax(ov) == tgt
    t(f"recall from 20% decoherence (seed {seed}): acc={acc:.4f} fid={fid:.4f} lock-in={locked}")

# 7. evidence passport hashes are deterministic per payload
p1 = evidence_passport(7, {"a": 1}, {"m": 0.5})
p2 = evidence_passport(7, {"a": 1}, {"m": 0.5})
assert p1["sha256"] == p2["sha256"] and len(p1["sha256"]) == 64
t("evidence passport deterministic hash")

print("\nAll tests passed.")
