"""Tests for benchmark_round4.py: plan lock, verdict rules, state classifier, early-step harness, thread analysis, chunk assembly."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_round3 as b3  # noqa: E402
import benchmark_round4 as b4  # noqa: E402
from cseq_wave.evidence import canonical_digest  # noqa: E402


def test_round4_lock_and_older_locks_untouched(tmp_path, monkeypatch):
    protocol, digest, locked = b4.load_protocol4()
    assert locked is True and digest == canonical_digest(json.loads((ROOT / "benchmark_protocol_round4.json").read_text()))
    assert (ROOT / "benchmark_protocol_round3.lock").read_text().strip() == "b6365f500ea5a6c1e60c8eb4db1200c6c5980e027dd409e3895a1676ff5637d4"
    assert protocol["round3_protocol_sha256"] == (ROOT / "benchmark_protocol_round3.lock").read_text().strip()
    changed = json.loads((ROOT / "benchmark_protocol_round4.json").read_text())
    changed["early_step"]["early_window_steps"] = [1]
    path = tmp_path / "p.json"
    path.write_text(json.dumps(changed))
    monkeypatch.setattr(b4, "PROTOCOL_PATH", path)
    with pytest.raises(SystemExit):
        b4.load_protocol4()
    assert b4.load_protocol4(allow_change=True)[2] is False


def test_protocol_matches_stated_requirements():
    p = json.loads((ROOT / "benchmark_protocol_round4.json").read_text())
    e = p["early_step"]
    assert e["n_modes"] == [64, 128, 256] and e["early_window_steps"] == list(b4.EARLY_WINDOW)
    assert e["seed_sets"]["held_out"]["trials_per_cell"]["full"] >= 50 and e["seed_sets"]["selection"]["trials_per_cell"]["full"] >= 50
    assert len(e["n_modes"]) * len(e["q"]) * len(e["p_over_n_primary"]) == 18
    t = p["threads"]
    assert t["n_modes_requested"] == [2048, 4096] and t["thread_counts"] == [1, 2, 3, 4] and t["runs"]["full"] == 2
    assert t["profiles"]["full"]["include_default"] is True


def _pats(rng, p, n):
    return rng.integers(0, 2, size=(p, n))


def test_classify_state_categories():
    rng = np.random.default_rng(0)
    pat = _pats(rng, 12, 256)
    xi = 1 - 2 * pat
    bits = lambda v: ((1 - v) // 2).astype(int)
    assert b4.classify_state(pat[0], pat)[0] == "stored_target_pos"
    assert b4.classify_state(1 - pat[0], pat)[0] == "stored_target_neg"
    assert b4.classify_state(pat[3], pat)[0] == "stored_other_pos"
    assert b4.classify_state(1 - pat[3], pat)[0] == "stored_other_neg"
    flipped = pat[0].copy()
    flipped[:20] = 1 - flipped[:20]  # 7.8% flips: still stored at tau 0.10
    assert b4.classify_state(flipped, pat)[0] == "stored_target_pos"
    assert b4.classify_state(bits(np.sign(xi[[2, 5, 7]].sum(axis=0))), pat)[0] == "mixture_3"
    assert b4.classify_state(bits(np.sign(xi[[1, 4, 6, 8, 9]].sum(axis=0))), pat)[0] == "mixture_5"
    # unstructured: random state
    cats = {b4.classify_state(rng.integers(0, 2, size=256), pat)[0] for _ in range(20)}
    assert cats <= {"unstructured", "partial_overlap"}
    # partial overlap: 70% of a stored pattern, rest random, not a mixture
    x = pat[0].copy()
    cut = rng.permutation(256)[:80]
    x[cut] = rng.integers(0, 2, size=80)
    cat, amax = b4.classify_state(x, pat)
    assert cat == "partial_overlap" and 0.5 <= amax < 0.8


def test_mean_direction_cosine_and_energy():
    rng = np.random.default_rng(1)
    pat = _pats(rng, 9, 128)
    xi = 1 - 2 * pat
    s = xi.sum(axis=0)
    aligned = ((1 - np.where(s >= 0, 1, -1)) // 2).astype(int)
    assert b4.mean_direction_cosine(aligned, pat) > 0.7
    assert b4.mean_direction_cosine(rng.integers(0, 2, size=128), pat) < 0.4
    mem = b3.build_memory(pat)
    w = mem.weights.real
    assert b4.energy_per_bit(w, pat[0]) < b4.energy_per_bit(w, rng.integers(0, 2, size=128))


def test_early_verdict_rules():
    assert b4.early_verdict(14, 18, 0.02, 0.01, 0.03) == "supported"
    assert b4.early_verdict(13, 18, 0.02, 0.01, 0.03) == "partial_or_inconclusive"
    assert b4.early_verdict(14, 18, 0.005, 0.001, 0.009) == "partial_or_inconclusive"  # mean below floor
    assert b4.early_verdict(4, 18, 0.02, 0.01, 0.03) == "rejected"
    assert b4.early_verdict(10, 18, -0.01, -0.02, 0.0) == "rejected"
    assert b4.early_verdict(2, 2, 0.02, 0.01, 0.03) == "supported"  # quick profile: ceil(0.75 * 2) = 2
    assert b4.early_verdict(0, 0, 0, 0, 0) == "not_testable"


def test_early_trials_step0_is_cue_and_matches_recall():
    acc, mism = b4.early_cell_trials(64, 0.3, 1, 0.125, 5, 20261028, 72, check=True)
    assert acc.shape == (5, 26) and mism == 0
    assert np.all((acc[:, 0] > 0.5) & (acc[:, 0] < 0.9))  # q = 0.3 cue accuracy about 0.7
    a2, _ = b4.early_cell_trials(64, 0.3, 1, 0.125, 5, 20261028, 72)
    assert np.array_equal(acc, a2)
    sel, _ = b4.early_cell_trials(64, 0.3, 1, 0.125, 5, 20261028, 71)
    assert not np.array_equal(acc, sel)  # selection and held-out seed sets differ


def test_pooled_bootstrap_and_family():
    rng = np.random.default_rng(3)
    g = [rng.normal(0.05, 0.05, 40), rng.normal(0.0, 0.05, 40)]
    lo, hi = b4.pooled_bootstrap(g, np.random.default_rng(1), 2000, 0.95)
    assert lo < np.mean([x.mean() for x in g]) < hi
    ps, bh, means, verd = b4.compare_family(g, 0, 1, 2000, 0.05)
    assert verd[0] == "higher" and verd[1] == "not_significant"


def test_wilson_helpers_and_frac_verdict():
    assert b4._frac_verdict(8, 10, 0.8, 0.2) == "supported"
    assert b4._frac_verdict(2, 10, 0.8, 0.2) == "rejected"
    assert b4._frac_verdict(5, 10, 0.8, 0.2) == "partial_or_inconclusive"
    assert b4._frac_verdict(0, 0, 0.8, 0.2) == "not_testable"


def _fake_entry(n, secs, cpu):
    return {"n_modes": n, "measured": True, "load_controlled": True, "recall_seconds": [secs * (1 + 0.01 * i) for i in range(7)],
            "attempts": [{"external_busy_cores": 0.1}], "accepted_attempt": 0, "cpu_over_wall": cpu}


def test_analyse_threads_verdicts():
    st = json.loads((ROOT / "benchmark_protocol_round4.json").read_text())["statistics"]
    base = {2048: 0.2, 4096: 0.8, 8192: 3.2}
    speed = {"1": 1.0, "2": 1.8, "3": 2.5, "4": 3.0, "default": 3.8}
    raw = {}
    for run in (1, 2):
        raw[run] = {m: {"large": [_fake_entry(n, t / s * (1.02 if run == 2 else 1.0), 1.0 if m == "1" else float(m.replace("default", "8")))
                                  for n, t in base.items()]} for m, s in speed.items()}
    an = b4.analyse_threads(raw, {}, st, 1, "full")
    h = an["hypotheses"]
    assert h["T1_speedup"]["verdict"] == "supported" and h["T1_speedup"]["units"] == 18
    assert h["T2_monotone"]["verdict"] == "supported" and h["T2_monotone"]["units"] == 6
    assert h["T3_repeatability"]["verdict"] == "supported"
    # estimate k=4: t8192^2 / t4096 = (3.2/3)^2 / (0.8/3) = 4.27 s > 3.5; k=default is not in the 2..4 rule, so not reachable
    assert h["T4_original_claim"]["verdict"] == "rejected"
    assert an["thread_effect_flags"] == []
    # flag a mode that did not use threads
    raw[1]["4"]["large"][1]["cpu_over_wall"] = 1.0
    assert b4.analyse_threads(raw, {}, st, 1, "full")["thread_effect_flags"][0]["mode"] == "4"


def test_chunked_quick_equals_single_quick_hash(tmp_path):
    full = b4.run4("quick")
    parts = []
    for part in ("early", "spurious", "threads"):
        rep = b4.run4("quick", part=part)
        path = tmp_path / f"{part}.json"
        path.write_text(json.dumps(rep, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        parts.append(str(path))
    assert b4.assemble(parts)["deterministic_sha256"] == full["deterministic_sha256"]
    assert full["early_step"]["trajectory_vs_recall_mismatches"] == 0
    assert full["protocol_locked"] is True


def test_quick_cli_prints_lock_and_hash():
    out = subprocess.run([sys.executable, str(ROOT / "benchmark_round4.py"), "--quick", "--part", "early"], capture_output=True, text=True, check=True).stdout
    assert "protocol_locked=True" in out and "deterministic_sha256:" in out


def test_full_profiles_resolve_to_listed_values():
    p = json.loads((ROOT / "benchmark_protocol_round4.json").read_text())
    e = b4.resolve_profile(p["early_step"], "full")
    assert e["n_modes"] == [64, 128, 256] and e["q"] == [0.1, 0.3] and e["p_over_n_primary"] == [0.125, 0.1875, 0.25]
    s = b4.resolve_profile(p["spurious_states"], "full")
    assert s["q"] == [0.0, 0.1, 0.3] and s["p_over_n"] == [0.125, 0.1875, 0.25]
    th = p["threads"]["profiles"]["full"]
    assert th["n_modes"] == [2048, 4096, 8192] and th["thread_counts"] == [1, 2, 3, 4]


def test_merge_thread_analysis_from_chunk_reports():
    st = json.loads((ROOT / "benchmark_protocol_round4.json").read_text())["statistics"]
    base = {2048: 0.2, 4096: 0.8}
    speed = {"1": 1.0, "2": 1.9, "4": 3.6}
    chunks = []
    for run in (1, 2):
        raw = {run: {m: {"large": [_fake_entry(n, t / s, 1.0 if m == "1" else float(m)) for n, t in base.items()]} for m, s in speed.items()}}
        an = b4.analyse_threads(raw, {}, st, 1, "full")
        batches = [{"run": run, "mode": m, "n_modes": e["n_modes"], "measured": True, "load_controlled": True,
                    "attempts": [{"recall_seconds": e["recall_seconds"], "accepted": True, "external_busy_cores": 0.1}]}
                   for m, d in raw[run].items() for e in d["large"]]
        chunks.append({"analysis": an, "batches": batches, "environments": {}, "runs_included": [run]})
    merged = b4.merge_thread_analysis(chunks, "full", 1)
    h = merged["analysis"]["hypotheses"]
    assert merged["runs_included"] == [1, 2]
    assert h["T3_repeatability"]["units"] == 6 and h["T3_repeatability"]["verdict"] == "supported"
    assert h["T1_speedup"]["units"] == 8
