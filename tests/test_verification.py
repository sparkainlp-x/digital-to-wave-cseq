import json

import numpy as np
import pytest

import demo_verified
from cseq_wave import (
    BOUNDARY_STATEMENT,
    HolographicMemory,
    assess_recall,
    bits_to_wave,
    calibrate_cue_threshold,
    recompute_all,
    screen_text,
    tag,
    verify_claims,
)
from cseq_wave.claims import ClaimViolation, assert_clean

N = 256


def build(seed=1, p=4):
    rng = np.random.default_rng(seed)
    cb = rng.integers(0, 2, (p, N))
    mem = HolographicMemory(N)
    for b in cb:
        mem.store(bits_to_wave(b))
    mem.finalize()
    return rng, cb, mem


def test_calibration_deterministic_and_floor():
    _, _, mem = build()
    a = calibrate_cue_threshold(mem, seed=3, n_null=64)
    b = calibrate_cue_threshold(mem, seed=3, n_null=64)
    assert a == b and a["evidence_tag"] == "SYNTHETIC"
    assert 0.01 <= a["tau_cue_overlap"] < 0.1
    assert calibrate_cue_threshold(mem, seed=3, n_null=8, floor=0.5)["tau_cue_overlap"] == 0.5


def test_cue_gate_separates_stored_from_never_stored():
    rng, cb, mem = build(2)
    tau = calibrate_cue_threshold(mem, seed=11)["tau_cue_overlap"]
    stored = [mem.overlaps(bits_to_wave(cb[i % 4]).decohere(rng, flip_prob=0.2, jitter=0.3).phase).max() for i in range(40)]
    ghosts = [mem.overlaps(bits_to_wave(rng.integers(0, 2, N)).decohere(rng, flip_prob=0.2, jitter=0.3).phase).max() for _ in range(200)]
    assert min(stored) > tau
    assert np.mean(np.asarray(ghosts) >= tau) < 0.05


@pytest.mark.parametrize("seed", [7, 42])
def test_verified_demo_passes_and_reproduces(seed, tmp_path, capsys):
    out = tmp_path / "v.json"
    assert demo_verified.main(["--seed", str(seed), "--out", str(out)]) == 0
    pp = json.loads(out.read_text())
    assert pp["metrics"]["control_verdict"] == "CONFABULATION_SUSPECTED"
    assert pp["metrics"]["control_oes32_spectrum_flag"] is True  # OES fires anyway
    assert demo_verified.run(seed)[0]["sha256"] == pp["sha256"]
    assert "PASS" in capsys.readouterr().out


def test_seed7_matches_handoff_numbers():
    pp, cal = demo_verified.run(7)
    assert cal["tau_cue_overlap"] == pytest.approx(0.0315, abs=1e-4)
    assert pp["metrics"]["main_cue_overlap"] == pytest.approx(0.2576, abs=1e-4)
    assert pp["metrics"]["control_cue_overlap"] == pytest.approx(0.0082, abs=1e-4)


def test_inverted_recall_is_not_verified():
    """v0.2.0 handoff bug: fidelity is sign-blind, so an inverted recall passed."""
    rng, cb, mem = build()
    cal = calibrate_cue_threshold(mem, seed=1)
    cue = bits_to_wave(cb[0]).decohere(rng, flip_prob=0.2)
    inverted = cb[0] * np.pi + np.pi
    v = assess_recall(mem, cue, inverted, cal, target_phase=cb[0] * np.pi, target_index=0)
    assert v["metrics"]["fidelity_cseq"]["value"] == pytest.approx(1.0)
    assert v["verdict"] == "UNVERIFIED_RECALL"
    assert assess_recall(mem, cue, inverted, cal)["verdict"] == "UNVERIFIED_RECALL"


def test_wrong_target_index_is_suspected_confabulation():
    rng, cb, mem = build()
    cue = bits_to_wave(cb[0]).decohere(rng, flip_prob=0.2)
    rec, _ = mem.recall(cue)
    v = assess_recall(mem, cue, rec, calibrate_cue_threshold(mem, seed=1), target_phase=cb[1] * np.pi, target_index=1)
    assert v["verdict"] == "CONFABULATION_SUSPECTED"


def test_mixture_cue_passes_cue_gate_but_not_recall_gate():
    _, cb, mem = build(3)
    cal = calibrate_cue_threshold(mem, seed=1)
    mix = cb[0].copy()
    mix[N // 2:] = cb[1][N // 2:]
    cue = bits_to_wave(mix)
    rec, _ = mem.recall(cue, steps=40)
    v = assess_recall(mem, cue, rec, cal)
    assert v["gates"]["cue_gate"] and v["verdict"] == "UNVERIFIED_RECALL"


def test_fail_closed_on_missing_calibration_and_errors():
    rng, cb, mem = build()
    cue = bits_to_wave(cb[0])
    assert assess_recall(mem, cue, cb[0] * np.pi, None)["verdict"] == "WITHHELD"
    v = assess_recall(mem, cue, np.zeros(3), calibrate_cue_threshold(mem, seed=1))
    assert v["verdict"] == "WITHHELD" and "error" in v


def test_verifier_catches_tampering():
    _, cb, mem = build()
    rec = recompute_all(mem._patterns, cb[0] * np.pi, cb[0] * np.pi, cb[0] * np.pi)
    ok, _ = verify_claims(dict(rec), rec)
    assert ok
    bad = dict(rec, fidelity_cseq=rec["fidelity_cseq"] - 0.1, cue_overlap_winner=3)
    ok, checks = verify_claims(bad, rec)
    assert not ok and sum(not c["ok"] for c in checks) == 2


def test_verifier_bit_accuracy_handles_wrapped_target_phases():
    _, cb, mem = build()
    target = cb[0] * np.pi
    assert recompute_all(mem._patterns, target, target, -target)["bit_accuracy"] == 1.0
    assert recompute_all(mem._patterns, target, target, target + 2 * np.pi)["bit_accuracy"] == 1.0


@pytest.mark.parametrize("text", [
    BOUNDARY_STATEMENT,
    "Not quantum hardware.",
    "This is not a validation of the Coherent Sequence theory.",
])
def test_claims_screen_passes_disclaimers(text):
    assert screen_text(text) == []
    assert assert_clean(text) == text


@pytest.mark.parametrize("text", [
    "This is a quantum computer.",
    "It is not slow at all: a quantum computer.",  # negation in another clause
    "Results are proven.",
    "A validated, production-ready system.",
    "Strong real-world performance.",
    "This proves the theory.",
])
def test_claims_screen_refuses_claims(text):
    assert screen_text(text)
    with pytest.raises(ClaimViolation):
        assert_clean(text)


def test_tag_rejects_unknown_tags():
    assert tag(1.0, "SYNTHETIC") == {"value": 1.0, "tag": "SYNTHETIC"}
    with pytest.raises(ValueError):
        tag(1.0, "MEASURED")


def test_cue_gate_sweep_is_deterministic(tmp_path):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "cue_gate_sweep.py"
    spec = importlib.util.spec_from_file_location("cue_gate_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.mixture() == mod.mixture()
    assert mod.separation(seeds=1, trials=5) == mod.separation(seeds=1, trials=5)
