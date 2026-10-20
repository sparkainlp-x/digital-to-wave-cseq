import json

import numpy as np
import pytest

import demo
from cseq_wave import HolographicMemory, WaveState, detect_events, evidence_passport, oes32_score


def test_oes32_formula():
    x = np.array([0.0, 1.0, -0.5])
    expected = 0.45 * 1.0 + 0.35 * np.sqrt(np.mean(x**2)) + 0.20 * np.mean(np.abs(x))
    assert oes32_score(x) == pytest.approx(expected)
    assert oes32_score([]) == 0.0
    assert detect_events([0.5]) == (pytest.approx(0.5), True)
    assert detect_events([0.49])[1] is False


def test_oes32_on_unit_interval_is_at_least_the_mean():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.random(25)
        assert oes32_score(x) >= x.mean() - 1e-12


def test_oes32_flag_fires_even_for_empty_memory_and_noise():
    """Honest limitation: the trajectory flag marks convergence only."""
    rng = np.random.default_rng(4)
    mem = HolographicMemory(256)
    mem.finalize()
    _, traj = mem.recall(WaveState(np.ones(256), rng.uniform(-np.pi, np.pi, 256)))
    assert detect_events(traj)[1]


def test_passport_hash_deterministic_and_numpy_safe():
    p1 = evidence_passport(7, {"a": np.int64(1)}, {"m": np.float32(0.5), "v": np.array([1, 2]), "b": np.bool_(True)})
    p2 = evidence_passport(7, {"a": 1}, {"m": 0.5, "v": [1, 2], "b": True})
    assert p1["sha256"] == p2["sha256"] and len(p1["sha256"]) == 64
    json.dumps(p1)
    assert p1["evidence_label"] == "SYNTHETIC"
    assert evidence_passport(8, {"a": 1}, {"m": 0.5})["sha256"] != p1["sha256"]


def test_passport_rejects_nan():
    with pytest.raises(ValueError):
        evidence_passport(1, {}, {"m": float("nan")})


@pytest.mark.parametrize("seed", [7, 42])
def test_demo_acceptance_and_reproducible_digest(seed, tmp_path, capsys):
    out = tmp_path / "p.json"
    assert demo.main(["--seed", str(seed), "--out", str(out)]) == 0
    pp = json.loads(out.read_text())
    assert pp["metrics"]["bit_accuracy"] == 1.0
    assert pp["metrics"]["fidelity_cseq"] == 1.0
    assert pp["metrics"]["control_unstored_oes32_flag"] is True
    assert demo.run(seed)["sha256"] == pp["sha256"]
    assert "PASS" in capsys.readouterr().out


def test_sweep_runs_and_is_deterministic(tmp_path):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "capacity_sweep.py"
    spec = importlib.util.spec_from_file_location("capacity_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--trials", "2", "--outdir", str(tmp_path / "a")])
    mod.main(["--trials", "2", "--outdir", str(tmp_path / "b")])
    a = json.loads((tmp_path / "a" / "sweeps.json").read_text())
    b = json.loads((tmp_path / "b" / "sweeps.json").read_text())
    assert a["sha256"] == b["sha256"] and a["evidence_label"] == "SYNTHETIC"
