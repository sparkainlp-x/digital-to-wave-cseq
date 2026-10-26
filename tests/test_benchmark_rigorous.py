"""Tests for benchmark_rigorous.py: statistics, reverse adversarial cases, protocol lock, determinism."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_rigorous as br  # noqa: E402
from cseq_wave.encode import bits_to_text, bits_to_wave, text_to_bits, wave_to_bits  # noqa: E402
from cseq_wave.evidence import canonical_digest  # noqa: E402


# ------------------------------------------------------------ protocol lock

def test_protocol_lock_matches():
    protocol, digest, locked = br.load_protocol()
    assert locked is True
    assert digest == (ROOT / "benchmark_protocol.lock").read_text().strip()
    assert digest == canonical_digest(json.loads((ROOT / "benchmark_protocol.json").read_text()))


def test_modified_protocol_is_refused_unless_allowed(tmp_path, monkeypatch):
    protocol = json.loads((ROOT / "benchmark_protocol.json").read_text())
    protocol["scaling"]["fit_min_n"] = 64  # post-hoc change of the pre-declared fit range
    changed = tmp_path / "benchmark_protocol.json"
    changed.write_text(json.dumps(protocol))
    monkeypatch.setattr(br, "PROTOCOL_PATH", changed)
    with pytest.raises(SystemExit):
        br.load_protocol()
    _, digest, locked = br.load_protocol(allow_change=True)
    assert locked is False and digest != (ROOT / "benchmark_protocol.lock").read_text().strip()


def test_protocol_meets_stated_minimums():
    p = json.loads((ROOT / "benchmark_protocol.json").read_text())
    full = p["scaling"]["profiles"]["full"]
    assert min(full["repeats"], full["repeats_large"]) >= 7
    assert p["scaling"]["fit_min_n"] == 512
    assert p["capacity"]["profiles"]["full"]["trials"] >= 30
    assert set(p["capacity"]["profiles"]["full"]["n_modes"]) == {32, 64, 128, 256}
    assert p["scaling"]["profiles"]["quick"]["repeats"] >= 7
    assert 16384 not in full["n_list"] and max(full["n_list"]) * max(full["n_list"]) * 16 * p["scaling"]["peak_bytes_factor"] <= p["scaling"]["max_peak_bytes"]


# --------------------------------------------------------------- statistics

def test_wilson_known_values():
    lo, hi = br.wilson(0, 40)
    assert lo == 0.0 and abs(hi - 0.087622) < 1e-5
    lo, hi = br.wilson(20, 40)
    assert lo < 0.5 < hi
    assert all(np.isnan(br.wilson(0, 0)))


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    vals = np.random.default_rng(0).normal(0.7, 0.05, size=40)
    a = br.boot_ci(vals, np.random.default_rng(1), 500, 0.95)
    b = br.boot_ci(vals, np.random.default_rng(1), 500, 0.95)
    assert a == b and a[0] < vals.mean() < a[1]


def _synthetic(alpha, n_list, repeats=9, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for n in n_list:
        t = 1e-9 * n ** alpha
        out.append({"n_modes": n, "measured": True, "repeats": repeats,
                    "recall_seconds": list(t * (1 + noise * rng.standard_normal(repeats))),
                    "store_seconds": [t] * 3, "weights_bytes": 16 * n * n})
    return out


SC = json.loads((ROOT / "benchmark_protocol.json").read_text())["scaling"]


def test_analyze_scaling_recovers_known_exponent():
    meas = _synthetic(2.0, [64, 128, 256, 512, 1024, 2048, 4096])
    res = br.analyze_scaling(meas, SC, seed=1)
    fit = res["fit"]
    assert fit["fit_n"] == [512, 1024, 2048, 4096]
    assert abs(fit["alpha"] - 2.0) < 0.05
    assert fit["alpha_ci"][0] <= 2.0 <= fit["alpha_ci"][1]
    assert fit["ci_contains_theoretical"] and not fit["ci_overlaps_original_claim_1.4_1.5"]
    ex = fit["extrapolations"][0]
    assert ex["n_modes"] == 16384 and "NOT MEASURED" in ex["status"]
    assert ex["interval_seconds"][0] <= ex["point_seconds"] <= ex["interval_seconds"][1]
    assert all(abs(s["slope"] - 2.0) < 0.2 for s in res["local_slopes"])


def test_analyze_scaling_exposes_overhead_regime():
    # constant overhead below N=256, N^2 above: early local slopes must be visibly smaller than late ones
    meas = []
    for n in [64, 128, 256, 512, 1024, 2048]:
        t = 1e-4 + 1e-9 * n ** 2 * (n >= 256)
        meas.append({"n_modes": n, "measured": True, "repeats": 9, "recall_seconds": [t * (1 + 0.001 * i) for i in range(9)],
                     "store_seconds": [t], "weights_bytes": 16 * n * n})
    res = br.analyze_scaling(meas, SC, seed=1)
    assert res["local_slopes"][0]["slope"] < 0.2 < res["local_slopes"][-1]["slope"]


def test_fit_not_run_with_too_few_points_and_unmeasured_are_listed():
    meas = _synthetic(2.0, [64, 128, 512]) + [{"n_modes": 99999, "measured": False, "reason": "cap"}]
    res = br.analyze_scaling(meas, SC, seed=1)
    assert "alpha" not in res["fit"] and "not run" in res["fit"]["status"]
    assert res["unmeasured"][0]["n_modes"] == 99999


def test_scaling_worker_respects_memory_cap_without_allocating():
    args = {"seed": 1, "n_list": [32, 60000], "repeats": 2, "repeats_large": 2, "large_from_n": 10 ** 9,
            "warmup": 1, "steps": 3, "max_peak_bytes": SC["max_peak_bytes"], "peak_bytes_factor": SC["peak_bytes_factor"]}
    out = br.scaling_worker(json.dumps(args))
    small, big = out["measurements"]
    assert small["measured"] and len(small["recall_seconds"]) == 2
    assert big["measured"] is False and "cap" in big["reason"]


# ------------------------------------------------------- reverse / adversarial

@pytest.mark.parametrize("bits", [[0], [1], [0] * 7, [1] * 7, [1, 0, 1], [0, 1] * 33, [1] * 255, [0] * 4097])
def test_codec_exact_on_adversarial_bits(bits):
    b = np.array(bits)
    w = bits_to_wave(b)
    assert np.array_equal(wave_to_bits(w), b)
    back = bits_to_wave(wave_to_bits(w))
    assert np.max(np.abs(np.angle(np.exp(1j * (back.phase - w.phase))))) == 0.0


def test_empty_input_behaviour_is_documented():
    # bits_to_wave accepts empty input; wave_to_bits rejects it (asymmetry, recorded in the report)
    w = bits_to_wave(np.array([], dtype=int))
    assert len(w.phase) == 0
    with pytest.raises(ValueError, match="nonempty"):
        wave_to_bits(w)
    assert bits_to_text(text_to_bits("")) == ""
    with pytest.raises(ValueError):
        bits_to_wave(np.array([[0, 1]]))
    with pytest.raises(ValueError):
        bits_to_wave(np.array([0, 2]))


def test_single_mode_memory_cannot_hold_a_pattern():
    from cseq_wave.memory import HolographicMemory
    from cseq_wave.wave import WaveState
    mem = HolographicMemory(1)
    mem.store(bits_to_wave(np.array([1])))
    mem.finalize()
    phases, _ = mem.recall(bits_to_wave(np.array([1])), steps=3)
    assert wave_to_bits(WaveState(np.ones(1), phases)).tolist() == [0]  # information lost: documented limit


def test_property_loop_many_seeds_exact_at_sigma_zero():
    for seed in range(300):
        rng = np.random.default_rng([99, seed])
        n = int(rng.integers(1, 400))
        bits = rng.integers(0, 2, size=n)
        assert np.array_equal(wave_to_bits(bits_to_wave(bits)), bits), (seed, n)
        text = "".join(chr(int(c)) for c in rng.integers(0x20, 0x7E, size=int(rng.integers(1, 20))))
        assert bits_to_text(wave_to_bits(bits_to_wave(text_to_bits(text)))) == text


def test_bench_reverse_reports_no_failures():
    p = json.loads((ROOT / "benchmark_protocol.json").read_text())
    rep = br.bench_reverse({"seeds": 120}, 5, p["reverse"]["max_length"])
    assert rep["failures"] == []
    assert all(v == 120 for v in rep["counts"].values())
    assert {a["case"] for a in rep["adversarial"]} >= {"empty_bits_encode", "empty_bits_round_trip", "length_1_zero", "length_1_one"}


# ------------------------------------------------------------------ capacity

def _cap(profile, seed=3):
    p = json.loads((ROOT / "benchmark_protocol.json").read_text())["capacity"]
    return br.bench_capacity(profile, [0.03125, 0.5], seed, p["recall_steps"], p["knee_accuracy_threshold"], 200, 0.95)


def test_capacity_null_baselines_and_ordering():
    rep = _cap({"n_modes": [32, 64], "sigmas": [0.1], "trials": 12})
    for r in rep["rows"]:
        assert abs(r["random_pattern_bit_accuracy"] - 0.5) < 0.1
        assert r["recall_ci_bootstrap"][0] <= r["recall_bit_accuracy"] <= r["recall_ci_bootstrap"][1]
        assert r["exact_recall_ci_wilson"][0] <= r["exact_recall_fraction"] <= r["exact_recall_ci_wilson"][1]
    for n in (32, 64):
        low = [r for r in rep["rows"] if r["n_modes"] == n and r["p_over_n"] == 0.03125][0]
        high = [r for r in rep["rows"] if r["n_modes"] == n and r["p_over_n"] == 0.5][0]
        assert low["recall_bit_accuracy"] == 1.0
        assert high["recall_bit_accuracy"] < 0.9 and high["recall_bit_accuracy"] > 0.5


def test_capacity_is_deterministic():
    prof = {"n_modes": [32], "sigmas": [0.0, 0.1], "trials": 6}
    assert _cap(prof) == _cap(prof)


# ----------------------------------------------------------- determinism (CLI)

def _quick(tmp_path, name):
    out = tmp_path / name
    proc = subprocess.run([sys.executable, "-W", "error", str(ROOT / "benchmark_rigorous.py"), "--quick", "--json", str(out)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def test_quick_run_hash_is_deterministic_and_excludes_timings(tmp_path):
    a, b = _quick(tmp_path, "a.json"), _quick(tmp_path, "b.json")
    assert a["deterministic_sha256"] == b["deterministic_sha256"]
    assert a["protocol_locked"] is True and a["protocol_sha256"] == b["protocol_sha256"]
    assert a["reverse"]["failures"] == []
    # timings differ between the two runs (otherwise the exclusion would be untestable)
    ta = a["scaling"]["modes"]["blas_threads_1"]["analysis"]["per_n"][0]["median_seconds"]
    tb = b["scaling"]["modes"]["blas_threads_1"]["analysis"]["per_n"][0]["median_seconds"]
    assert ta > 0 and tb > 0
    env = a["scaling"]["modes"]["blas_threads_1"]["environment"]
    assert env["thread_env"]["OPENBLAS_NUM_THREADS"] == "1" and env["numpy"] == np.__version__
