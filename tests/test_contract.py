"""Regression tests for the v0.3.0 verification contract (SYNTHETIC)."""

import numpy as np
import pytest

from cseq_wave import HolographicMemory, WaveState, assess_recall, bits_to_wave, calibrate_cue_threshold, recompute_all, verify_claims
from cseq_wave.calibrate import validate_calibration
from cseq_wave.encode import bits_to_text, text_to_bits
from cseq_wave.validation import validate_bits, validate_phase
from cseq_wave.verdict import _decide

N = 64


def build(seed=1, p=3):
    rng = np.random.default_rng(seed)
    cb = rng.integers(0, 2, (p, N))
    mem = HolographicMemory(N)
    for b in cb:
        mem.store(bits_to_wave(b))
    mem.finalize()
    return rng, cb, mem


def good():
    _, cb, mem = build()
    t = cb[0] * np.pi
    rec = recompute_all(mem._patterns, t, t, t)
    return mem, cb, rec


# --- verify_claims -------------------------------------------------------

def test_valid_report_passes():
    mem, _, rec = good()
    ok, checks = verify_claims(dict(rec), rec, n_patterns=mem.n_patterns)
    assert ok and len(checks) == 5


@pytest.mark.parametrize("rep,recd", [({}, {"bit_accuracy": 1.0}), ({"bit_accuracy": 1.0}, {}), ({}, {}), (None, None)])
def test_empty_dicts_fail(rep, recd):
    ok, checks = verify_claims(rep, recd)
    assert not ok and all("reason" in c for c in checks if not c["ok"])


@pytest.mark.parametrize("key", ["cue_overlap_max", "cue_overlap_winner", "fidelity_cseq", "bit_accuracy"])
@pytest.mark.parametrize("side", ["reported", "recomputed"])
def test_missing_required_metric_fails(key, side):
    _, _, rec = good()
    rep, recd = dict(rec), dict(rec)
    (rep if side == "reported" else recd).pop(key)
    ok, checks = verify_claims(rep, recd)
    assert not ok
    assert any(c["claim"] == key and "missing from " + side in c["reason"] for c in checks)


def test_unknown_reported_metric_rejected_and_extra_recomputed_ok():
    _, _, rec = good()
    ok, checks = verify_claims(dict(rec, quantum_advantage=1.0), rec)
    assert not ok and any(c["reason"] == "unknown reported metric" for c in checks if not c["ok"])
    rep = {k: rec[k] for k in ("cue_overlap_max", "cue_overlap_winner", "fidelity_cseq", "bit_accuracy")}
    assert verify_claims(rep, rec)[0]  # cue_margin optional


@pytest.mark.parametrize("key,bad", [
    ("fidelity_cseq", True), ("fidelity_cseq", "1.0"), ("fidelity_cseq", float("nan")),
    ("fidelity_cseq", float("inf")), ("fidelity_cseq", 1.5), ("bit_accuracy", -0.1),
    ("cue_overlap_max", 1.01), ("cue_margin", -0.5), ("cue_margin", float("nan")),
    ("cue_overlap_winner", -1), ("cue_overlap_winner", 0.0), ("cue_overlap_winner", True),
    ("cue_overlap_winner", 3), ("fidelity_cseq", 1 + 0j),
])
def test_invalid_types_and_ranges_fail(key, bad):
    mem, _, rec = good()
    ok, checks = verify_claims(dict(rec, **{key: bad}), rec, n_patterns=mem.n_patterns)
    assert not ok and any(c["claim"] == key and not c["ok"] for c in checks)


def test_invalid_recomputed_value_fails_even_if_equal():
    _, _, rec = good()
    bad = dict(rec, fidelity_cseq=float("nan"))
    assert not verify_claims(bad, dict(bad))[0]


@pytest.mark.parametrize("tol", [-1e-3, float("nan"), float("inf"), "0.01", True, None])
def test_invalid_tolerance_fails(tol):
    _, _, rec = good()
    ok, checks = verify_claims(dict(rec), rec, tol=tol)
    assert not ok and checks[0]["claim"] == "tolerance"


def test_tampering_reasons_are_explicit():
    _, _, rec = good()
    ok, checks = verify_claims(dict(rec, fidelity_cseq=rec["fidelity_cseq"] - 0.1), rec)
    assert not ok and "tol" in [c for c in checks if not c["ok"]][0]["reason"]


# --- validation helpers ---------------------------------------------------

@pytest.mark.parametrize("bad", [
    [], "0101", [np.nan, 0.0], [np.inf, 0.0], [1 + 1j, 0], [True, False],
    np.array(["a", "b"]), np.array([object(), 1], dtype=object), [[0.0, 1.0]], 0.5,
])
def test_phase_validation_rejects(bad):
    with pytest.raises((TypeError, ValueError)):
        validate_phase(bad)


def test_phase_validation_shape_no_broadcast():
    assert validate_phase([0, 1], 2).dtype == np.float64
    for bad in ([0.0], [0.0, 1.0, 2.0]):
        with pytest.raises(ValueError):
            validate_phase(bad, 2)


@pytest.mark.parametrize("bad", [[0, 0.5], [0, 0.9999], [np.nan, 1], [np.inf], "01", [1j, 0], [0, 2], [-1, 0], [[0, 1]]])
def test_binary_validation_before_cast(bad):
    with pytest.raises((TypeError, ValueError)):
        validate_bits(bad)
    with pytest.raises((TypeError, ValueError)):
        bits_to_wave(bad)


def test_binary_accepts_exact_values_and_empty_text_roundtrip():
    assert validate_bits([0.0, 1.0, True]).tolist() == [0, 1, 1]
    assert bits_to_text(text_to_bits("")) == ""
    with pytest.raises(ValueError):
        validate_bits([], allow_empty=False)


def test_memory_rejects_bad_phases_and_broadcast():
    _, cb, mem = build()
    with pytest.raises(ValueError):
        mem.overlaps(np.zeros(1))  # would broadcast against N modes
    with pytest.raises(ValueError):
        mem.recall(WaveState(np.ones(N), np.full(N, np.nan)))
    with pytest.raises(ValueError):
        mem.fidelity(np.zeros(N), np.zeros(N - 1))
    with pytest.raises(ValueError):
        recompute_all(mem._patterns, np.zeros(N), np.zeros(N), np.zeros(1))


# --- slicing ---------------------------------------------------------------

def test_slicing_can_lower_fidelity():
    mem = HolographicMemory(2)
    rec, tgt = np.array([1.5, 1.7]), np.zeros(2)
    unsliced = mem.fidelity(rec, tgt, slice_output=False)
    sliced = mem.fidelity(rec, tgt, slice_output=True)
    assert unsliced > 0.99 and sliced == pytest.approx(0.0, abs=1e-12)


# --- calibration binding ----------------------------------------------------

def test_calibration_schema_and_binding():
    _, cb, mem = build()
    cal = calibrate_cue_threshold(mem, seed=1, n_null=32)
    assert cal["schema_version"] == "cseq-cal/1" and cal["memory_digest"] == mem.digest()
    assert validate_calibration(cal, mem) == cal["tau_cue_overlap"]


def test_digest_depends_on_order_and_content():
    _, cb, a = build()
    b = HolographicMemory(N)
    for bits in cb[::-1]:
        b.store(bits_to_wave(bits))
    assert a.digest() != b.digest()


def test_stale_calibration_withheld():
    rng, cb, mem = build()
    cal = calibrate_cue_threshold(mem, seed=1, n_null=32)
    mem.store(bits_to_wave(rng.integers(0, 2, N)))
    cue = bits_to_wave(cb[0])
    v = assess_recall(mem, cue, cb[0] * np.pi, cal)
    assert v["verdict"] == "WITHHELD" and "pattern_count" in v["error"]


def test_mismatched_calibration_withheld():
    _, cb, mem = build(1)
    _, _, other = build(2)
    cal = calibrate_cue_threshold(other, seed=1, n_null=32)
    v = assess_recall(mem, bits_to_wave(cb[0]), cb[0] * np.pi, cal)
    assert v["verdict"] == "WITHHELD" and "digest" in v["error"]


@pytest.mark.parametrize("mutate", [
    lambda c: {"tau_cue_overlap": 0.03},
    lambda c: dict(c, schema_version="cseq-cal/0"),
    lambda c: dict(c, tau_cue_overlap=float("nan")),
    lambda c: dict(c, tau_cue_overlap=1.5),
    lambda c: dict(c, mode_count=N + 1),
    lambda c: dict(c, seed=1.5),
    lambda c: dict(c, null_quantile=1.0),
    lambda c: {k: v for k, v in c.items() if k != "noise"},
])
def test_legacy_or_invalid_calibration_withheld(mutate):
    _, cb, mem = build()
    cal = mutate(calibrate_cue_threshold(mem, seed=1, n_null=32))
    assert assess_recall(mem, bits_to_wave(cb[0]), cb[0] * np.pi, cal)["verdict"] == "WITHHELD"


# --- assess_recall ------------------------------------------------------------

def test_valid_examples_still_verified_in_both_modes():
    _, cb, mem = build()
    cal = calibrate_cue_threshold(mem, seed=1, n_null=64)
    cue, tgt = bits_to_wave(cb[1]), cb[1] * np.pi
    rec, _ = mem.recall(cue)
    blind = assess_recall(mem, cue, rec, cal)
    truth = assess_recall(mem, cue, rec, cal, target_phase=tgt, target_index=1)
    assert blind["verdict"] == truth["verdict"] == "VERIFIED_RECALL"
    assert blind["mode"] == "self_consistency" and truth["mode"] == "ground_truth"


@pytest.mark.parametrize("idx", [-1, 3, 1.0, 1.9, True, "1", np.float64(1.0)])
def test_bad_target_index_withheld(idx):
    _, cb, mem = build()
    cal = calibrate_cue_threshold(mem, seed=1, n_null=32)
    v = assess_recall(mem, bits_to_wave(cb[1]), cb[1] * np.pi, cal, target_phase=cb[1] * np.pi, target_index=idx)
    assert v["verdict"] == "WITHHELD"


def test_numpy_integer_target_index_ok():
    _, cb, mem = build()
    cal = calibrate_cue_threshold(mem, seed=1, n_null=32)
    v = assess_recall(mem, bits_to_wave(cb[1]), cb[1] * np.pi, cal, target_phase=cb[1] * np.pi, target_index=np.int64(1))
    assert v["verdict"] == "VERIFIED_RECALL"


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_recalled_or_cue_never_verified(bad):
    _, cb, mem = build()
    cal = calibrate_cue_threshold(mem, seed=1, n_null=32)
    rec = cb[0] * np.pi
    rec = rec.astype(float); rec[3] = bad
    assert assess_recall(mem, bits_to_wave(cb[0]), rec, cal)["verdict"] == "WITHHELD"
    cue = WaveState(np.ones(N), np.full(N, bad))
    assert assess_recall(mem, cue, cb[0] * np.pi, cal)["verdict"] == "WITHHELD"


@pytest.mark.parametrize("bad_target", [np.zeros(N - 1), np.zeros(1), np.full(N, np.nan)])
def test_bad_target_phase_withheld(bad_target):
    _, cb, mem = build()
    cal = calibrate_cue_threshold(mem, seed=1, n_null=32)
    assert assess_recall(mem, bits_to_wave(cb[0]), cb[0] * np.pi, cal, target_phase=bad_target)["verdict"] == "WITHHELD"


def _metrics(acc_rep, acc_rec, fid=1.0, ov=0.5):
    rep = {"cue_overlap_max": ov, "cue_overlap_winner": 0, "fidelity_cseq": fid, "bit_accuracy": acc_rep}
    rec = dict(rep, bit_accuracy=acc_rec)
    return rep, rec


def test_agreement_never_overrides_failed_gate_bit_accuracy():
    """Reported 0.992 vs recomputed 0.988: within tol, but recomputed < 0.99."""
    mem = build()[2]
    rep, rec = _metrics(0.992, 0.988)
    ok, checks = verify_claims(rep, rec, n_patterns=mem.n_patterns)
    assert ok  # agreement holds ...
    v = _decide(rep, rec, ok, checks, 0.03, None, "ground_truth", mem, np.zeros(N))
    assert v["verdict"] != "VERIFIED_RECALL" and not v["gates"]["recall_gate"]


def test_agreement_never_overrides_failed_gate_cue_and_fidelity():
    mem = build()[2]
    rep = {"cue_overlap_max": 0.032, "cue_overlap_winner": 0, "fidelity_cseq": 0.992, "bit_accuracy": 1.0}
    rec = dict(rep, cue_overlap_max=0.029)
    v = _decide(rep, rec, True, [], 0.03, None, "self_consistency", mem, np.zeros(N))
    assert v["verdict"] == "CONFABULATION_SUSPECTED"
    rec = dict(rep, fidelity_cseq=0.988)
    v = _decide(rep, rec, True, [], 0.03, None, "self_consistency", mem, np.zeros(N))
    assert v["verdict"] == "UNVERIFIED_RECALL"
