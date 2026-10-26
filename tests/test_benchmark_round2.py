"""Tests for benchmark_round2.py: plan lock, denoising harness, complex64 implementation, load gate."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_round2 as b2  # noqa: E402
from cseq_wave.encode import bits_to_wave  # noqa: E402
from cseq_wave.evidence import canonical_digest  # noqa: E402
from cseq_wave.memory import HolographicMemory  # noqa: E402

TINY = {"n_modes": [64], "p_over_n": [0.0625, 0.125], "q": [0.0, 0.2], "sigma": [0.9], "trials": 6}


def test_round2_lock_matches_and_refuses_changes(tmp_path, monkeypatch):
    protocol, digest, locked = b2.load_protocol2()
    assert locked is True
    assert digest == canonical_digest(json.loads((ROOT / "benchmark_protocol_round2.json").read_text()))
    changed = json.loads((ROOT / "benchmark_protocol_round2.json").read_text())
    changed["denoising"]["profiles"]["full"]["trials"] = 5
    path = tmp_path / "p.json"
    path.write_text(json.dumps(changed))
    monkeypatch.setattr(b2, "PROTOCOL_PATH", path)
    with pytest.raises(SystemExit):
        b2.load_protocol2()
    assert b2.load_protocol2(allow_change=True)[2] is False


def test_round1_protocol_untouched():
    # round 2 must not edit the round-1 plan
    assert (ROOT / "benchmark_protocol.lock").read_text().strip() == \
        "826e3e6870f9e3b95b605e2e70701067b970909f196b739d8e662dd06f094c9f"
    p2 = json.loads((ROOT / "benchmark_protocol_round2.json").read_text())
    assert p2["round1_protocol_sha256"] == (ROOT / "benchmark_protocol.lock").read_text().strip()


def test_full_profile_meets_stated_minimums():
    p = json.loads((ROOT / "benchmark_protocol_round2.json").read_text())
    full = p["denoising"]["profiles"]["full"]
    assert full["trials"] >= 40
    assert set(full["n_modes"]) == {64, 128, 256}
    assert {0.05, 0.10, 0.20, 0.30} <= set(full["q"])
    assert p["scaling_round2"]["complex64"]["n_list"][-1] == 16384


def test_denoising_is_deterministic_and_control_is_exact():
    a = b2.bench_denoising(TINY, 7, 25, 200, 0.95)
    b = b2.bench_denoising(TINY, 7, 25, 200, 0.95)
    assert canonical_digest(a) == canonical_digest(b)
    assert a["control_cue_accuracy_is_exactly_1"] is True
    for r in a["rows"]:
        lo, hi = r["gain_over_cue_ci_paired_bootstrap"]
        assert lo <= r["gain_over_cue"] <= hi or r["is_control"]
        assert 0 <= r["exact_recall_ci_wilson"][0] <= r["exact_recall_fraction"] <= r["exact_recall_ci_wilson"][1] <= 1


def test_realised_flip_fraction_tracks_q():
    prof = {**TINY, "n_modes": [128], "p_over_n": [0.0625], "q": [0.3], "sigma": [], "trials": 40}
    row = b2.bench_denoising(prof, 3, 5, 100, 0.95)["rows"][0]
    assert abs(row["realised_cue_flip_fraction"] - 0.3) < 0.03


def test_verdict_rule():
    assert b2.verdict([0.01, 0.2]) == "denoises"
    assert b2.verdict([-0.2, -0.01]) == "degrades"
    assert b2.verdict([-0.01, 0.2]) == "inconclusive"
    assert b2.verdict([0.0, 0.2]) == "inconclusive"  # lower bound must be strictly above 0


def test_known_regimes():
    # tiny load, moderate flips: recall should repair the cue; huge load: it should not help
    easy = {**TINY, "n_modes": [128], "p_over_n": [0.03125], "q": [0.2], "sigma": [], "trials": 20}
    row = b2.bench_denoising(easy, 5, 25, 300, 0.95)["rows"][0]
    assert row["gain_over_cue"] > 0.1 and row["verdict_vs_cue"] == "denoises"


def test_c64_matches_complex128_weights_and_recall():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=(5, 70))
    mem = HolographicMemory(70)
    for b in bits:
        mem.store(bits_to_wave(b))
    mem.finalize()
    w = b2.build_c64(bits)
    assert w.dtype == np.complex64 and w.shape == (70, 70)
    assert np.allclose(w, mem.weights, atol=1e-5)
    cue = bits_to_wave(bits[0]).phase
    out = b2.recall_c64(w, cue, 25)
    assert out.dtype == np.float32 and np.array_equal(np.cos(out) < 0, bits[0] == 1)


def test_c64_chunked_build_equals_unchunked(monkeypatch):
    bits = np.random.default_rng(2).integers(0, 2, size=(3, 50))
    full = b2.build_c64(bits)
    monkeypatch.setattr(b2, "CHUNK_BYTES", 8 * 50 * 7)  # 7 rows per chunk, 50 is not a multiple
    assert np.array_equal(full, b2.build_c64(bits))


def test_agreement_check_reports_counts():
    spec = {"n_modes": [128], "n_patterns": [4], "seeds": 5, "q": 0.1}
    out = b2.agreement_check(spec, 9, 25)[0]
    assert out["seeds"] == 5 and 0 <= out["identical_bit_vectors"] <= 5 and 0.9 <= out["mean_bit_agreement"] <= 1.0


def test_load_gate_retries_then_flags(monkeypatch):
    lc = {"threshold_external_busy_cores": 0.5, "max_attempts": 3, "retry_wait_seconds": 0}
    t = {"v": 0.0}

    def fake_busy():
        return t["v"]

    monkeypatch.setattr(b2, "_busy_seconds", fake_busy)
    monkeypatch.setattr(b2, "_own_cpu", lambda: 0.0)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        t["v"] += 1000.0  # external load enormous relative to wall time -> always rejected

    res = b2.controlled_batch(fn, 2, lc)
    assert res["load_controlled"] is False and len(res["attempts"]) == 3 and calls["n"] == 6
    assert all(not a["accepted"] for a in res["attempts"])
    # idle: no busy increase -> accepted on the first attempt
    monkeypatch.setattr(b2, "_busy_seconds", lambda: 5.0)
    res = b2.controlled_batch(lambda: None, 2, lc)
    assert res["load_controlled"] is True and res["accepted_attempt"] == 0 and len(res["attempts"]) == 1


def _quick(tmp_path, name):
    out = tmp_path / name
    subprocess.run([sys.executable, "-W", "error", str(ROOT / "benchmark_round2.py"), "--quick", "--json", str(out)],
                   check=True, capture_output=True, text=True, cwd=ROOT)
    return json.loads(out.read_text())


def test_quick_run_hash_excludes_timings_and_load(tmp_path):
    a = _quick(tmp_path, "a.json")
    b = _quick(tmp_path, "b.json")
    assert a["deterministic_sha256"] == b["deterministic_sha256"]
    assert a["protocol_locked"] is True
    assert a["denoising"]["control_cue_accuracy_is_exactly_1"] is True
    batch = a["scaling"]["modes"]["blas_threads_1"]["complex64"]["batches"][0]
    assert batch["measured"] and "attempts" in batch and "external_busy_cores" in batch["attempts"][0]
    assert "loadavg1_before" in batch["attempts"][0]
