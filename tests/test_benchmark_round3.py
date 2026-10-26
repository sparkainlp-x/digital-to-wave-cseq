"""Tests for benchmark_round3.py: plan lock, statistics, pattern generators, mechanism harness, cache analysis."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_round2 as b2  # noqa: E402
import benchmark_round3 as b3  # noqa: E402
from cseq_wave.encode import bits_to_wave, wave_to_bits  # noqa: E402
from cseq_wave.evidence import canonical_digest  # noqa: E402
from cseq_wave.memory import HolographicMemory  # noqa: E402
from cseq_wave.wave import WaveState  # noqa: E402


def test_round3_lock_and_older_locks_untouched(tmp_path, monkeypatch):
    protocol, digest, locked = b3.load_protocol3()
    assert locked is True and digest == canonical_digest(json.loads((ROOT / "benchmark_protocol_round3.json").read_text()))
    assert (ROOT / "benchmark_protocol.lock").read_text().strip() == "826e3e6870f9e3b95b605e2e70701067b970909f196b739d8e662dd06f094c9f"
    assert (ROOT / "benchmark_protocol_round2.lock").read_text().strip() == "a06cd2e65d18d2eac28b57a1a74e8291ab5b81b7f31f28f7637685d8f8bcc27d"
    assert protocol["round2_protocol_sha256"] == (ROOT / "benchmark_protocol_round2.lock").read_text().strip()
    changed = json.loads((ROOT / "benchmark_protocol_round3.json").read_text())
    changed["mechanism"]["profiles"]["full"]["trials"] = 5
    path = tmp_path / "p.json"
    path.write_text(json.dumps(changed))
    monkeypatch.setattr(b3, "PROTOCOL_PATH", path)
    with pytest.raises(SystemExit):
        b3.load_protocol3()
    assert b3.load_protocol3(allow_change=True)[2] is False


def test_full_profiles_meet_stated_requirements():
    p = json.loads((ROOT / "benchmark_protocol_round3.json").read_text())
    assert p["mechanism"]["step_snapshots"] == [1, 2, 5, 10, 25, 50, 100]
    assert p["correlated"]["rho"] == [0.0, 0.1, 0.3]
    assert p["mechanism"]["profiles"]["full"]["trials"] >= 40
    assert p["tests_for_multiplicity"]["permutations"]["full"] >= 100000
    assert max(p["complex128_large"]["n_list"]) * max(p["complex128_large"]["n_list"]) * 16 + 64 * 2**20 <= 4 * 2**30


def test_holm_and_bh_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(b3.holm_adjust(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(b3.bh_adjust(p), [0.02, 0.04, 0.04, 0.02])
    assert b3.holm_adjust([1e-9, 0.9])[0] < 1e-8 and (b3.holm_adjust([0.2, 0.9]) <= 1).all()


def test_signflip_p_behaviour():
    rng = np.random.default_rng(0)
    assert b3.signflip_p(np.full(50, 0.05), 5000, rng) < 1e-3  # all positive: hopelessly unlikely under symmetry
    assert b3.signflip_p(np.zeros(50), 5000, rng) == 1.0
    g = np.array([1, -1] * 25, dtype=float) * 0.1
    assert b3.signflip_p(g, 5000, np.random.default_rng(1)) > 0.5
    a = b3.signflip_p(np.linspace(-0.1, 0.2, 50), 3000, np.random.default_rng(2))
    assert a == b3.signflip_p(np.linspace(-0.1, 0.2, 50), 3000, np.random.default_rng(2))  # deterministic
    assert b3.adjusted_verdict(0.01, 0.1, 0.05) == "denoises" and b3.adjusted_verdict(0.01, -0.1, 0.05) == "degrades"
    assert b3.adjusted_verdict(0.2, 0.1, 0.05) == "inconclusive"


def test_hadamard_is_orthogonal_and_has_zero_one_step_error():
    h = b3.hadamard(64)
    assert np.array_equal(h @ h.T, 64 * np.eye(64, dtype=int))
    rng = np.random.default_rng(3)
    bits = b3.gen_patterns("hadamard", 0.0, 16, 64, rng, {})
    xi = 1 - 2 * bits
    g = xi @ xi.T
    assert np.array_equal(g, 64 * np.eye(16, dtype=int))
    mem = b3.build_memory(bits)
    traj, _ = b3.trajectory(mem.weights, bits[0].astype(np.int8), 3)
    assert np.array_equal(traj[1], bits[0])  # exact cue is a fixed point with zero cross-talk


def test_correlated_generator_hits_target_rho():
    rng = np.random.default_rng(4)
    for rho in (0.1, 0.3):
        vals = []
        for _ in range(40):
            bits = b3.gen_patterns("random", rho, 20, 256, rng, {})
            xi = 1 - 2 * bits
            g = xi @ xi.T / 256
            vals.append((g.sum() - np.trace(g)) / (20 * 19))
        assert abs(np.mean(vals) - rho) < 0.02


def test_trajectory_matches_recall_and_weights_are_nearly_real():
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2, size=(12, 64))
    mem = b3.build_memory(bits)
    assert np.max(np.abs(mem.weights.imag)) < 1e-12  # real up to rounding: +-1 Hopfield dynamics, ties decided by rounding
    cue = bits[0].copy()
    cue[:10] = 1 - cue[:10]
    traj, _ = b3.trajectory(mem.weights, cue.astype(np.int8), 25)
    ref = wave_to_bits(WaveState(np.ones(64), mem.recall(WaveState(np.ones(64), cue * np.pi), steps=25)[0]))
    assert np.array_equal(traj[25], ref)


def test_classification_and_dynamics_helpers():
    pats = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1] * 5])
    assert b3.classify_final(pats[0], pats, 0.1) == "retrieved_target"
    assert b3.classify_final(pats[2], pats, 0.1) == "retrieved_other_stored"
    assert b3.classify_final(np.array([1, 0] * 5), pats, 0.1) == "mirror_other"
    mixed = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1])
    assert b3.classify_final(mixed, pats, 0.1) == "spurious"
    still = np.zeros((101, 4), dtype=np.int8)
    assert b3.dynamics_class(still) == "fixed_point" and b3.convergence_step(still) == 1
    osc = np.zeros((101, 4), dtype=np.int8)
    osc[1::2] = 1
    assert b3.dynamics_class(osc) == "period_2" and b3.convergence_step(osc) is None


def test_crosstalk_prediction_values():
    assert abs(b3.crosstalk_prediction(256, 65) - 0.0228) < 0.003
    assert b3.crosstalk_prediction(64, 1) == 0.0
    assert b3.gauss_q(0) == 0.5


def test_wilson_verdict_thresholds():
    assert b3.wilson_verdict(90, 100, 0.5, 0.1) == "supported"
    assert b3.wilson_verdict(0, 200, 0.5, 0.1) == "rejected"
    assert b3.wilson_verdict(30, 100, 0.5, 0.1) == "partial_or_inconclusive"
    assert b3.wilson_verdict(0, 0, 0.5, 0.1) == "not_testable"


def test_drop_events_and_cache_analysis():
    series = [(1000, 10.0), (2000, 9.0), (4000, 5.0), (8000, 4.9)]
    ev = b3.drop_events(series, "bytes")
    assert len(ev) == 1 and ev[0]["bytes_lo"] == 2000 and ev[0]["bytes_hi"] == 4000
    mk = lambda n, t: {"dtype": "complex128", "n_modes": n, "matrix_bytes": 16 * n * n, "measured": True, "load_controlled": True, "recall_seconds": [t]}
    raw = {"matvec": [mk(100, 1e-5), mk(200, 4e-5), mk(400, 5e-4)],  # bandwidth collapses between 200 and 400
           "streaming": [{"bytes": 16 * 200 * 200, "measured": True, "load_controlled": True, "recall_seconds": [1e-4]},
                         {"bytes": 16 * 400 * 400, "measured": True, "load_controlled": True, "recall_seconds": [1e-3]}]}
    cinfo = {"sysfs": [{"level": 2, "type": "Unified", "size_bytes": 16 * 300 * 300, "shared_cpu_list": "0"}], "lscpu": []}
    out = b3.analyse_cache(raw, cinfo)
    assert out["m1_reported_cache"]["verdict"] == "supported" and out["m2_streaming_drop"]["verdict"] == "supported"
    cinfo["sysfs"][0]["size_bytes"] = 10**9
    assert b3.analyse_cache(raw, cinfo)["m1_reported_cache"]["verdict"] == "rejected"


def test_trial_sink_does_not_change_round2_output():
    prof = json.loads((ROOT / "benchmark_protocol_round2.json").read_text())["denoising"]["profiles"]["quick"]
    a = b2.bench_denoising(prof, 5, 25, 100, 0.95)
    sink = []
    c = b2.bench_denoising(prof, 5, 25, 100, 0.95, trial_sink=sink)
    assert canonical_digest(a) == canonical_digest(c) and len(sink) == len(a["rows"])
    assert all(len(s["gain"]) == prof["trials"] for s in sink)


def test_dense_builders_match_holographic_memory():
    bits = np.random.default_rng(6).integers(0, 2, size=(4, 40))
    mem = HolographicMemory(40)
    for b in bits:
        mem.store(bits_to_wave(b))
    mem.finalize()
    w = b3.build_dense(bits, np.complex128)
    assert w.dtype == np.complex128 and np.allclose(w, mem.weights, atol=1e-12)
    cue = bits_to_wave(bits[0]).phase
    assert np.allclose(b3.dense_recall(w, cue, 25, np.complex128), mem.recall(WaveState(np.ones(40), cue), steps=25)[0])


def test_mechanism_small_run_is_deterministic_and_checks_trajectory():
    mp = json.loads((ROOT / "benchmark_protocol_round3.json").read_text())["mechanism"]
    prof = {"n_modes": [64], "p_over_n": [0.0625, 0.25], "q": [0.0, 0.3], "trials": 6}
    c1, t1, m1 = b3.bench_mechanism(prof, mp, 11)
    c2, t2, m2 = b3.bench_mechanism(prof, mp, 11)
    assert canonical_digest(c1) == canonical_digest(c2) and m1 == 0
    had_q0 = [c for c in c1 if c["pattern_kind"] == "hadamard" and c["q"] == 0]
    assert had_q0 and all(c["one_step_error"] == 0.0 for c in had_q0)
    for c in c1:
        for tau, counts in c["final_category_counts_by_tau"].items():
            assert sum(counts.values()) == 6
        assert sum(c["dynamics_counts_step100"].values()) == 6


def _quick(tmp_path, name):
    out = tmp_path / name
    subprocess.run([sys.executable, "-W", "error", str(ROOT / "benchmark_round3.py"), "--quick", "--json", str(out)],
                   check=True, capture_output=True, text=True, cwd=ROOT)
    return json.loads(out.read_text())


def test_quick_run_hash_excludes_timings_and_cache_info(tmp_path):
    a, b = _quick(tmp_path, "a.json"), _quick(tmp_path, "b.json")
    assert a["deterministic_sha256"] == b["deterministic_sha256"] and a["protocol_locked"] is True
    assert a["mechanism"]["trajectory_vs_recall_mismatches"] == 0
    batch = a["timing"]["modes"]["blas_threads_1"]["complex128_large"]["batches"][0]
    assert batch["measured"] and "external_busy_cores" in batch["attempts"][0]
