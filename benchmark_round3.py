"""Round-3 benchmark for cseq_wave (SYNTHETIC classical software).

Plan: benchmark_protocol_round3.json (hash in benchmark_protocol_round3.lock),
committed before this file existed and before any round-3 experiment was run.

  1. MECHANISM: why does recall become worse than the cue at high load?
     Cross-talk, spurious states, sign mirror and truncated/oscillating
     dynamics, each with a pre-declared support / reject rule.
  2. CORRELATED patterns (pairwise correlation rho): capacity and denoising.
  3. MULTIPLICITY: re-analysis of the 108 round-2 cells with sign-flip
     permutation p-values, Holm and Benjamini-Hochberg.
  4. CACHE microbenchmark for the step above N^2 (matvec and streaming read
     bandwidth versus working set, against the caches the VM reports).
  5. complex128 timed directly at 8192..14336 with a chunked matrix.

Usage (repo root):

    python benchmark_round3.py --quick                           # CI smoke
    python benchmark_round3.py --full --json report_round3.json  # manual

No quantum-hardware claim. Timings, load, cache info and memory decisions are
not part of the deterministic hash.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import benchmark_rigorous as r1  # noqa: E402
import benchmark_round2 as b2  # noqa: E402
from benchmark_rigorous import boot_ci, r6, wilson  # noqa: E402
from cseq_wave.encode import bits_to_wave  # noqa: E402
from cseq_wave.evidence import canonical_digest  # noqa: E402
from cseq_wave.memory import HolographicMemory  # noqa: E402

PROTOCOL_PATH = ROOT / "benchmark_protocol_round3.json"
LOCK_PATH = ROOT / "benchmark_protocol_round3.lock"
ROUND2_STORED = ROOT / "results" / "round2" / "full_run1.json"
CHUNK_BYTES = 64 * 1024 * 1024


def load_protocol3(allow_change=False):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    digest = canonical_digest(protocol)
    locked = LOCK_PATH.read_text().strip() if LOCK_PATH.exists() else None
    ok = digest == locked
    if not ok and not allow_change:
        raise SystemExit(
            f"round-3 protocol hash {digest} does not match benchmark_protocol_round3.lock ({locked}). "
            "Use --allow-protocol-change to run anyway (recorded as protocol_locked=false)."
        )
    return protocol, digest, ok


# --------------------------------------------------------------- multiplicity

def signflip_p(gains, B, rng):
    """Two-sided sign-flip permutation p-value for mean(gain) = 0, add-one rule."""
    g = np.asarray(gains, dtype=float)
    if not np.any(g != 0):
        return 1.0
    obs = abs(g.sum())
    count, done = 0, 0
    while done < B:
        m = min(5000, B - done)
        signs = rng.integers(0, 2, size=(m, len(g))) * 2 - 1
        count += int(np.sum(np.abs(signs @ g) >= obs - 1e-12))
        done += m
    return (1 + count) / (1 + B)


def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(-p, kind="stable")  # descending
    adj = np.empty(m)
    running = 1.0
    for k, i in zip(range(m, 0, -1), order):
        running = min(running, p[i] * m / k)
        adj[i] = running
    return adj


def adjusted_verdict(adj_p, mean_gain, alpha, pos="denoises", neg="degrades"):
    if adj_p < alpha and mean_gain > 0:
        return pos
    if adj_p < alpha and mean_gain < 0:
        return neg
    return "inconclusive"


def family_tests(gain_list, B, seed, family_id, alpha):
    """gain_list: list of per-trial gain arrays. Returns raw p, Holm p, BH p, verdicts."""
    ps = [signflip_p(g, B, np.random.default_rng([seed, 61, family_id, i])) for i, g in enumerate(gain_list)]
    holm, bh = holm_adjust(ps), bh_adjust(ps)
    means = [float(np.mean(g)) for g in gain_list]
    return {"p": ps, "p_holm": list(holm), "p_bh": list(bh),
            "verdict_holm": [adjusted_verdict(h, m, alpha) for h, m in zip(holm, means)],
            "verdict_bh": [adjusted_verdict(h, m, alpha) for h, m in zip(bh, means)]}


# ----------------------------------------------------------------- patterns

def hadamard(n):
    h = np.array([[1]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    if h.shape[0] != n:
        raise ValueError("Hadamard order must be a power of two")
    return h


def gen_patterns(kind, rho, p, n, rng, hcache):
    """Return (p, n) 0/1 bit patterns."""
    if kind == "hadamard":
        if n not in hcache:
            hcache[n] = hadamard(n)
        rows = rng.choice(np.arange(1, n), size=p, replace=False)
        return (hcache[n][rows] == -1).astype(int)
    if rho == 0:
        return rng.integers(0, 2, size=(p, n))
    c = rng.integers(0, 2, size=n)
    agree = rng.random((p, n)) < (1 + math.sqrt(rho)) / 2
    return np.where(agree, c, 1 - c)


def build_memory(patterns):
    n = patterns.shape[1]
    mem = HolographicMemory(n)
    for bits in patterns:
        mem.store(bits_to_wave(bits))
    mem.finalize()
    return mem


def _bits_of(s):
    return (np.abs(np.angle(np.exp(1j * np.angle(s)))) > np.pi / 2).astype(np.int8)


def trajectory(weights, cue_bits, steps_max):
    """Bits at steps 0..steps_max; same arithmetic as HolographicMemory.recall."""
    s = np.exp(1j * (cue_bits.astype(float) * np.pi))
    out = np.empty((steps_max + 1, len(cue_bits)), dtype=np.int8)
    out[0] = cue_bits
    tie_frac = None
    for t in range(1, steps_max + 1):
        field = weights @ s
        if t == 1:
            tie_frac = float(np.mean(np.abs(field.real) < 1e-9))
        s = np.exp(1j * np.angle(field))
        out[t] = _bits_of(s)
    return out, tie_frac


def classify_final(bits, patterns, tau):
    """Category of a final bit vector relative to stored patterns (target = row 0)."""
    d = np.mean(patterns != bits[None, :], axis=1)
    if d[0] <= tau:
        return "retrieved_target"
    if len(d) > 1 and d[1:].min() <= tau:
        return "retrieved_other_stored"
    if 1 - d[0] <= tau:
        return "mirror_target"
    if len(d) > 1 and (1 - d[1:]).min() <= tau:
        return "mirror_other"
    return "spurious"


def dynamics_class(traj):
    b98, b99, b100 = traj[-3], traj[-2], traj[-1]
    if np.array_equal(b98, b99) and np.array_equal(b99, b100):
        return "fixed_point"
    if np.array_equal(b100, b98) and not np.array_equal(b100, b99):
        return "period_2"
    return "still_moving"


def convergence_step(traj):
    for t in range(1, len(traj)):
        if np.array_equal(traj[t], traj[t - 1]):
            return t
    return None


def gauss_q(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def crosstalk_prediction(n, p):
    return gauss_q(math.sqrt((n - 1) / (p - 1))) if p > 1 else 0.0


# ---------------------------------------------------------------- mechanism

CATS = ["retrieved_target", "retrieved_other_stored", "mirror_target", "mirror_other", "spurious"]
DYN = ["fixed_point", "period_2", "still_moving"]


def bench_mechanism(prof, mp, seed):
    snaps = mp["step_snapshots"]
    smax = max(snaps)
    taus = [mp["classification"]["tau_primary"]] + list(mp["classification"]["tau_sensitivity"])
    res, trials, mismatches, hcache = [], {}, 0, {}
    kinds = mp["pattern_kinds"]
    for n in prof["n_modes"]:
        for ki, kind in enumerate(kinds):
            for qi, q in enumerate(prof["q"]):
                for ratio in prof["p_over_n"]:
                    p = int(round(n * ratio))
                    assert abs(p / n - ratio) < 1e-12
                    rng = np.random.default_rng([seed, 51, n, ki, qi, p])
                    acc = {t: [] for t in snaps}
                    cue_acc, ovl_t, ovl_o, ties = [], [], [], []
                    cat = {tau: {c: 0 for c in CATS} for tau in taus}
                    cat25 = {c: 0 for c in CATS}
                    dyn = {c: 0 for c in DYN}
                    conv, changed = [], 0
                    for _ in range(prof["trials"]):
                        patterns = gen_patterns(kind, 0.0, p, n, rng, hcache)
                        mem = build_memory(patterns)
                        target = patterns[0]
                        flips = (rng.random(n) < q) if q > 0 else np.zeros(n, dtype=bool)
                        cue_bits = np.where(flips, 1 - target, target).astype(np.int8)
                        traj, tf = trajectory(mem.weights, cue_bits, smax)
                        ref = b2.wave_to_bits(b2.WaveState(np.ones(n), mem.recall(
                            b2.WaveState(np.ones(n), cue_bits * np.pi), steps=25)[0]))
                        mismatches += int(not np.array_equal(ref, traj[25]))
                        for t in snaps:
                            acc[t].append(float(np.mean(traj[t] == target)))
                        cue_acc.append(float(np.mean(cue_bits == target)))
                        xi_cue = 1 - 2 * cue_bits
                        xi = 1 - 2 * patterns
                        ov = xi @ xi_cue / n
                        ovl_t.append(float(ov[0]))
                        ovl_o.append(float(np.max(np.abs(ov[1:]))) if p > 1 else 0.0)
                        ties.append(tf)
                        for tau in taus:
                            cat[tau][classify_final(traj[smax], patterns, tau)] += 1
                        cat25[classify_final(traj[25], patterns, taus[0])] += 1
                        dyn[dynamics_class(traj)] += 1
                        conv.append(convergence_step(traj))
                        changed += int(not np.array_equal(traj[25], traj[smax]))
                    cell_id = (n, kind, q, ratio)
                    trials[cell_id] = {"acc": {t: np.asarray(v) for t, v in acc.items()}, "cue": np.asarray(cue_acc)}
                    brng = np.random.default_rng([seed, 53, n, ki, qi, p])
                    ci = {t: boot_ci(acc[t], brng, mp["bootstrap_resamples"], mp["ci_level"]) for t in (1, 25, smax)}
                    conv_ok = [c for c in conv if c is not None]
                    pred = crosstalk_prediction(n, p)
                    res.append({
                        "n_modes": n, "pattern_kind": kind, "q": q, "n_patterns": p, "p_over_n": ratio, "trials": prof["trials"],
                        "cue_bit_accuracy": r6(np.mean(cue_acc)),
                        "accuracy_by_step": {str(t): r6(np.mean(acc[t])) for t in snaps},
                        "accuracy_ci_bootstrap": {str(t): [r6(ci[t][0]), r6(ci[t][1])] for t in ci},
                        "one_step_error": r6(1 - np.mean(acc[1])),
                        "one_step_error_ci": [r6(1 - ci[1][1]), r6(1 - ci[1][0])],
                        "crosstalk_prediction_one_step_error": r6(pred),
                        "cue_overlap_with_target_mean": r6(np.mean(ovl_t)),
                        "cue_overlap_max_other_mean": r6(np.mean(ovl_o)),
                        "tie_fraction_step1_exploratory_not_preregistered": r6(np.mean(ties)),
                        "final_category_counts_by_tau": {str(tau): cat[tau] for tau in taus},
                        "step25_category_counts_tau_primary": cat25,
                        "dynamics_counts_step100": dyn,
                        "converged_by_step_25": int(sum(1 for c in conv if c is not None and c <= 25)),
                        "converged_by_step_100": len(conv_ok),
                        "median_convergence_step_among_converged": None if not conv_ok else float(np.median(conv_ok)),
                        "trials_with_bits25_ne_bits100": changed,
                    })
    return res, trials, mismatches


def _pooled_wilson(cells, key_fn):
    k = sum(key_fn(c) for c in cells)
    n = sum(c["trials"] for c in cells)
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return k, n, lo, hi


def wilson_verdict(k, n, sup_lower, rej_upper):
    if n == 0:
        return "not_testable"
    lo, hi = wilson(k, n)
    if lo >= sup_lower:
        return "supported"
    if hi <= rej_upper:
        return "rejected"
    return "partial_or_inconclusive"


def analyse_mechanism(cells, trials, prof, mp, seed, B, alpha):
    snaps = mp["step_snapshots"]
    smax = max(snaps)
    tau0 = mp["classification"]["tau_primary"]
    resamples = mp["bootstrap_resamples"]
    out = {}
    # per-cell gains and families
    def gains(c, t):
        tr = trials[(c["n_modes"], c["pattern_kind"], c["q"], c["p_over_n"])]
        return tr["acc"][t] - tr["cue"]
    fams = {}
    for kind in mp["pattern_kinds"]:
        cs = [c for c in cells if c["pattern_kind"] == kind and c["q"] > 0]
        for t in (25, smax):
            fid = 100 + 10 * mp["pattern_kinds"].index(kind) + (0 if t == 25 else 1)
            ft = family_tests([gains(c, t) for c in cs], B, seed, fid, alpha)
            fams[(kind, t)] = (cs, ft)
            for i, c in enumerate(cs):
                c.setdefault("gain_tests", {})[str(t)] = {
                    "gain": r6(np.mean(gains(c, t))), "p_signflip": r6(ft["p"][i]), "p_bh": r6(ft["p_bh"][i]),
                    "p_holm": r6(ft["p_holm"][i]), "verdict_bh": ft["verdict_bh"][i], "verdict_holm": ft["verdict_holm"][i]}
    xcells = [c for c in fams[("random", 25)][0] if c["gain_tests"]["25"]["verdict_bh"] == "degrades"]
    out["x_cells"] = [{"n_modes": c["n_modes"], "q": c["q"], "p_over_n": c["p_over_n"]} for c in xcells]
    out["n_x_cells"] = len(xcells)
    hyp = {}
    if len(xcells) < 3:
        for h in ("a_crosstalk", "b_spurious", "c_mirror", "d_truncation", "d_oscillation"):
            hyp[h] = {"verdict": "not_testable", "reason": f"only {len(xcells)} x-cells"}
        out["hypotheses"] = hyp
        return out
    # (a) cross-talk
    a1_rows = []
    for c in cells:
        if c["pattern_kind"] == "random" and c["q"] == 0:
            pred = c["crosstalk_prediction_one_step_error"]
            expected_errors = pred * c["trials"] * c["n_modes"]
            if expected_errors >= 5:
                lo, hi = c["one_step_error_ci"]
                meas = c["one_step_error"]
                agree = (lo <= pred <= hi) or (pred > 0 and 1 / 1.5 <= (meas / pred if pred else 0) <= 1.5)
                a1_rows.append({"n_modes": c["n_modes"], "p_over_n": c["p_over_n"], "measured": meas, "ci": [lo, hi],
                                "predicted": pred, "agrees": bool(agree)})
    agree_n = sum(r["agrees"] for r in a1_rows)
    a1_pass = len(a1_rows) >= 3 and agree_n / len(a1_rows) >= 0.75
    a1_fail = len(a1_rows) >= 1 and agree_n / len(a1_rows) < 0.5
    had = {(c["n_modes"], c["q"], c["p_over_n"]): c for c in fams[("hadamard", 25)][0]}
    had_q0 = [c for c in cells if c["pattern_kind"] == "hadamard" and c["q"] == 0]
    control_ok = all(c["one_step_error"] == 0.0 for c in had_q0) if had_q0 else None
    not_deg, deg, acc_diff = 0, 0, []
    rand_by_key = {(c["n_modes"], c["q"], c["p_over_n"]): c for c in cells if c["pattern_kind"] == "random"}
    for x in xcells:
        key = (x["n_modes"], x["q"], x["p_over_n"])
        h = had.get(key)
        if h is None:
            continue
        if h["gain_tests"]["25"]["verdict_bh"] == "degrades":
            deg += 1
        else:
            not_deg += 1
        acc_diff.append(h["accuracy_by_step"]["25"] - rand_by_key[key]["accuracy_by_step"]["25"])
    nx = max(1, deg + not_deg)
    a2_pass = (not_deg / nx >= 0.8) and (np.mean(acc_diff) >= 0.05 if acc_diff else False)
    a2_fail = deg / nx >= 0.5
    if a2_fail or a1_fail:
        va = "rejected"
    elif a1_pass and a2_pass:
        va = "supported"
    else:
        va = "partial_or_inconclusive"
    hyp["a_crosstalk"] = {
        "verdict": va, "a1_testable_cells": len(a1_rows), "a1_agreeing_cells": agree_n, "a1_rows": a1_rows,
        "a2_hadamard_control_exact_zero_one_step_error_at_q0": control_ok,
        "a2_xcells_with_hadamard_counterpart": deg + not_deg, "a2_hadamard_also_degrades": deg,
        "a2_hadamard_not_degrades": not_deg,
        "a2_mean_accuracy_step25_hadamard_minus_random": r6(np.mean(acc_diff)) if acc_diff else None}
    # (b), (c) pooled over x-cells, step 100, tau primary
    key_tau = str(tau0)
    def cnt(c, names):
        return sum(c["final_category_counts_by_tau"][key_tau][n] for n in names)
    kb, nb, lob, hib = _pooled_wilson(xcells_full(cells, xcells), lambda c: cnt(c, ["spurious"]))
    hyp["b_spurious"] = {"verdict": wilson_verdict(kb, nb, 0.50, 0.10), "k": kb, "n": nb, "wilson_95": [r6(lob), r6(hib)],
                         "sensitivity_pooled_spurious_fraction_by_tau": {
                             str(t): r6(sum(c["final_category_counts_by_tau"][str(t)]["spurious"] for c in xcells_full(cells, xcells)) / nb)
                             for t in [tau0] + list(mp["classification"]["tau_sensitivity"])}}
    kc, nc, loc, hic = _pooled_wilson(xcells_full(cells, xcells), lambda c: cnt(c, ["mirror_target", "mirror_other"]))
    hyp["c_mirror"] = {"verdict": wilson_verdict(kc, nc, 0.10, 0.05), "k": kc, "n": nc, "wilson_95": [r6(loc), r6(hic)],
                       "mirror_target_only_k": sum(cnt(c, ["mirror_target"]) for c in xcells_full(cells, xcells))}
    # (d) dynamics
    xs = xcells_full(cells, xcells)
    diffs = np.concatenate([trials[(c["n_modes"], c["pattern_kind"], c["q"], c["p_over_n"])]["acc"][smax]
                            - trials[(c["n_modes"], c["pattern_kind"], c["q"], c["p_over_n"])]["acc"][25] for c in xs])
    dci = boot_ci(diffs, np.random.default_rng([seed, 57]), resamples, mp["ci_level"])
    changed = sum(c["trials_with_bits25_ne_bits100"] for c in xs)
    ntot = sum(c["trials"] for c in xs)
    frac_changed = changed / ntot
    mean_diff = float(np.mean(diffs))
    if abs(mean_diff) >= 0.02 and frac_changed >= 0.25:
        vt = "supported"
    elif abs(mean_diff) < 0.01 and frac_changed <= 0.10:
        vt = "rejected"
    else:
        vt = "partial_or_inconclusive"
    still_x = sum(1 for c in xs if c["gain_tests"][str(smax)]["verdict_bh"] == "degrades")
    hyp["d_truncation"] = {"verdict": vt, "mean_accuracy_step100_minus_step25": r6(mean_diff), "ci_bootstrap": [r6(dci[0]), r6(dci[1])],
                           "fraction_trials_bits25_ne_bits100": r6(frac_changed), "x_cells_still_degrading_at_step100_bh": still_x,
                           "x_cells": len(xs)}
    kd = sum(c["dynamics_counts_step100"]["period_2"] for c in xs)
    kf = sum(c["dynamics_counts_step100"]["fixed_point"] for c in xs)
    hyp["d_oscillation"] = {"verdict": wilson_verdict(kd, ntot, 0.10, 0.05), "period_2_k": kd, "n": ntot,
                            "wilson_95": [r6(wilson(kd, ntot)[0]), r6(wilson(kd, ntot)[1])], "fixed_point_k": kf,
                            "still_moving_k": sum(c["dynamics_counts_step100"]["still_moving"] for c in xs)}
    out["hypotheses"] = hyp
    return out


def xcells_full(cells, xcells):
    keys = {(x["n_modes"], x["q"], x["p_over_n"]) for x in xcells}
    return [c for c in cells if c["pattern_kind"] == "random" and (c["n_modes"], c["q"], c["p_over_n"]) in keys]


# --------------------------------------------------------------- correlated

def bench_correlated(prof, cp, seed, B, alpha, steps=25, round2_rows=None):
    cells, gains_list, hcache = [], [], {}
    resamples = cp["bootstrap_resamples"]
    for n in prof["n_modes"]:
        for ri, rho in enumerate(prof["rho"]):
            for qi, q in enumerate(prof["q"]):
                for ratio in prof["p_over_n"]:
                    p = int(round(n * ratio))
                    assert abs(p / n - ratio) < 1e-12
                    rng = np.random.default_rng([seed, 71, n, ri, qi, p])
                    acc, cue_acc, near_acc, exact, retrieved, pair = [], [], [], 0, 0, []
                    for _ in range(prof["trials"]):
                        patterns = gen_patterns("random", rho, p, n, rng, hcache)
                        mem = build_memory(patterns)
                        target = patterns[0]
                        flips = (rng.random(n) < q) if q > 0 else np.zeros(n, dtype=bool)
                        cue_bits = np.where(flips, 1 - target, target)
                        cue_phase = cue_bits * np.pi
                        got = b2.wave_to_bits(b2.WaveState(np.ones(n), mem.recall(b2.WaveState(np.ones(n), cue_phase), steps=steps)[0]))
                        acc.append(float(np.mean(got == target)))
                        exact += int(np.array_equal(got, target))
                        cue_acc.append(float(np.mean(cue_bits == target)))
                        idx = int(np.argmax(mem.overlaps(cue_phase)))
                        near_acc.append(float(np.mean(patterns[idx] == target)))
                        retrieved += int(idx == 0)
                        if p > 1:
                            xi = 1 - 2 * patterns
                            g = xi @ xi.T / n
                            pair.append(float((g.sum() - np.trace(g)) / (p * (p - 1))))
                    acc, cue_acc, near_acc = map(np.asarray, (acc, cue_acc, near_acc))
                    gain, gain_n = acc - cue_acc, acc - near_acc
                    brng = np.random.default_rng([seed, 73, n, ri, qi, p])
                    gci, gnci, aci = (boot_ci(x, brng, resamples, cp["ci_level"]) for x in (gain, gain_n, acc))
                    ex_ci = wilson(exact, prof["trials"])
                    cell = {
                        "n_modes": n, "rho": rho, "q": q, "n_patterns": p, "p_over_n": ratio, "trials": prof["trials"],
                        "realised_mean_pairwise_overlap": r6(np.mean(pair)) if pair else None,
                        "recall_bit_accuracy": r6(acc.mean()), "recall_ci_bootstrap": [r6(aci[0]), r6(aci[1])],
                        "cue_only_bit_accuracy": r6(cue_acc.mean()), "nearest_stored_bit_accuracy": r6(near_acc.mean()),
                        "exact_recall_fraction": r6(exact / prof["trials"]), "exact_recall_ci_wilson": [r6(ex_ci[0]), r6(ex_ci[1])],
                        "nearest_stored_retrieval_fraction": r6(retrieved / prof["trials"]),
                        "gain_over_cue": r6(gain.mean()), "gain_over_cue_ci_paired_bootstrap": [r6(gci[0]), r6(gci[1])],
                        "recall_minus_nearest": r6(gain_n.mean()),
                        "recall_minus_nearest_ci_paired_bootstrap": [r6(gnci[0]), r6(gnci[1])],
                        "verdict_raw": None if q == 0 else b2.verdict(gci),
                    }
                    cells.append(cell)
                    if q > 0:
                        gains_list.append(gain)
    dcells = [c for c in cells if c["q"] > 0]
    ft = family_tests(gains_list, B, seed, 300, alpha)
    for c, p_raw, ph, pb, vh, vb in zip(dcells, ft["p"], ft["p_holm"], ft["p_bh"], ft["verdict_holm"], ft["verdict_bh"]):
        c.update({"p_signflip": r6(p_raw), "p_holm": r6(ph), "p_bh": r6(pb), "verdict_holm": vh, "verdict_bh": vb})
    per_rho = []
    for rho in prof["rho"]:
        cs = [c for c in dcells if c["rho"] == rho]
        row = {"rho": rho, "cells": len(cs)}
        for name, key in (("raw", "verdict_raw"), ("holm", "verdict_holm"), ("bh", "verdict_bh")):
            lab = {"raw": {"denoises": "D", "degrades": "X", "inconclusive": "~"}}.get(name)
            vals = [c[key] for c in cs]
            row[name] = {"denoises": vals.count("denoises"), "degrades": vals.count("degrades"), "inconclusive": vals.count("inconclusive")}
        per_rho.append(row)
    knees = []
    grid = sorted(prof["p_over_n"])
    for n in prof["n_modes"]:
        for rho in prof["rho"]:
            cs = sorted((c for c in cells if c["n_modes"] == n and c["rho"] == rho and c["q"] == 0), key=lambda c: c["p_over_n"])
            knees.append({"n_modes": n, "rho": rho, "knee_p_over_n": next((c["p_over_n"] for c in cs if c["recall_bit_accuracy"] < 0.95), None)})
    c0 = next((r for r in per_rho if r["rho"] == 0.0), None)
    c3 = next((r for r in per_rho if r["rho"] == 0.3), None)
    if c0 and c3:
        d0, d3 = c0["bh"]["denoises"], c3["bh"]["denoises"]
        cut = 6 if prof["trials"] >= 40 else 1
        he = "supported" if d3 <= d0 - cut else "rejected" if d3 >= d0 else "partial_or_inconclusive"
        hyp_e = {"verdict": he, "bh_denoises_rho0": d0, "bh_denoises_rho0.3": d3, "required_drop": cut}
    else:
        hyp_e = {"verdict": "not_testable"}
    repl = None
    if round2_rows is not None:
        r2 = {(r["n_modes"], r["corruption"], r["p_over_n"]): r["verdict_vs_cue"] for r in round2_rows if r["cue_model"] == "bit_flip" and r["corruption"] in (0.1, 0.3)}
        same, tot, diff = 0, 0, []
        for c in dcells:
            if c["rho"] == 0.0 and (c["n_modes"], c["q"], c["p_over_n"]) in r2:
                tot += 1
                ok = r2[(c["n_modes"], c["q"], c["p_over_n"])] == c["verdict_raw"]
                same += int(ok)
                if not ok:
                    diff.append({"n_modes": c["n_modes"], "q": c["q"], "p_over_n": c["p_over_n"], "round2": r2[(c["n_modes"], c["q"], c["p_over_n"])], "round3": c["verdict_raw"]})
        repl = {"cells_compared": tot, "same_raw_verdict": same, "differences": diff}
    return {"cells": cells, "per_rho_verdict_counts": per_rho, "knees_q0": knees, "hypothesis_e": hyp_e, "replication_vs_round2": repl,
            "multiplicity": "sign-flip permutation p; Holm and BH over all q > 0 cells (family correlated_vs_cue)"}


# --------------------------------------------------------- round-2 reanalysis

def reanalyse_round2(profile_name, seed2, alpha, B, seed3):
    p2, _, _ = b2.load_protocol2(allow_change=True)
    dn = p2["denoising"]
    sink = []
    out = b2.bench_denoising(dn["profiles"][profile_name], seed2, dn["recall_steps"], dn["bootstrap_resamples"], dn["ci_level"], trial_sink=sink)
    rows = out["rows"]
    stored = None
    reproduced = None
    if profile_name == "full" and ROUND2_STORED.exists():
        stored = json.loads(ROUND2_STORED.read_text())["denoising"]["rows"]
        s_map = {(r["n_modes"], r["cue_model"], r["corruption"], r["p_over_n"]): r for r in stored}
        mism = []
        for r in rows:
            s = s_map.get((r["n_modes"], r["cue_model"], r["corruption"], r["p_over_n"]))
            keys = ("recall_bit_accuracy", "cue_only_bit_accuracy", "nearest_stored_bit_accuracy", "gain_over_cue",
                    "gain_over_cue_ci_paired_bootstrap", "verdict_vs_cue", "recall_minus_nearest", "verdict_vs_nearest")
            if s is None or any(s[k] != r[k] for k in keys):
                mism.append({"n_modes": r["n_modes"], "cue_model": r["cue_model"], "corruption": r["corruption"], "p_over_n": r["p_over_n"]})
        reproduced = {"cells": len(rows), "mismatching_cells": len(mism), "mismatches": mism[:10], "all_reproduced": len(mism) == 0}
    cells = [(r, t) for r, t in zip(rows, sink) if not r["is_control"]]
    fam = {}
    for name, key in (("round2_vs_cue", "gain"), ("round2_vs_nearest", "gain_vs_nearest")):
        ft = family_tests([t[key] for _, t in cells], B, seed3, 200 if name.endswith("cue") else 201, alpha)
        raw = [r["verdict_vs_cue"] if name.endswith("cue") else {"recall_better": "denoises", "recall_worse": "degrades", "inconclusive": "inconclusive"}[r["verdict_vs_nearest"]] for r, _ in cells]
        def surv(adj_key):
            return {v: {"raw": raw.count(v), "survive": sum(1 for rv, av in zip(raw, ft[adj_key]) if rv == v and av == v)} for v in ("denoises", "degrades")}
        became = {"from_inconclusive_to_significant": {}}
        fam[name] = {
            "raw_counts": {v: raw.count(v) for v in ("denoises", "degrades", "inconclusive")},
            "holm": {"counts": {v: ft["verdict_holm"].count(v) for v in ("denoises", "degrades", "inconclusive")}, "survivors": surv("verdict_holm"),
                     "flipped_raw_to_inconclusive": sum(1 for rv, av in zip(raw, ft["verdict_holm"]) if rv != "inconclusive" and av == "inconclusive"),
                     "newly_significant": sum(1 for rv, av in zip(raw, ft["verdict_holm"]) if rv == "inconclusive" and av != "inconclusive"),
                     "sign_reversed": sum(1 for rv, av in zip(raw, ft["verdict_holm"]) if {rv, av} == {"denoises", "degrades"})},
            "bh": {"counts": {v: ft["verdict_bh"].count(v) for v in ("denoises", "degrades", "inconclusive")}, "survivors": surv("verdict_bh"),
                   "flipped_raw_to_inconclusive": sum(1 for rv, av in zip(raw, ft["verdict_bh"]) if rv != "inconclusive" and av == "inconclusive"),
                   "newly_significant": sum(1 for rv, av in zip(raw, ft["verdict_bh"]) if rv == "inconclusive" and av != "inconclusive"),
                   "sign_reversed": sum(1 for rv, av in zip(raw, ft["verdict_bh"]) if {rv, av} == {"denoises", "degrades"})},
            "changed_cells_holm": [{"n_modes": r["n_modes"], "cue_model": r["cue_model"], "corruption": r["corruption"], "p_over_n": r["p_over_n"],
                                    "raw": rv, "holm": av, "p": r6(p)} for (r, _), rv, av, p in zip(cells, raw, ft["verdict_holm"], ft["p"]) if rv != av],
            "min_possible_p": r6(1 / (1 + B)), "largest_p_among_significant_raw": r6(max([p for p, rv in zip(ft["p"], raw) if rv != "inconclusive"] or [0])),
        }
    return {"stored_cells_reproduced": reproduced, "round2_raw_rows_from_rerun": rows, "families": fam, "permutations": B, "alpha": alpha}


# ------------------------------------------------------------ cache + large

def cache_info():
    out = {"sysfs": [], "lscpu": []}
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    try:
        for d in sorted(base.glob("index*")):
            rd = lambda f: (d / f).read_text().strip() if (d / f).exists() else None
            size = rd("size") or ""
            mult = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}.get(size[-1:], 1)
            out["sysfs"].append({"level": int(rd("level")), "type": rd("type"), "size": size,
                                 "size_bytes": int(size.rstrip("KMG")) * mult if size else None, "shared_cpu_list": rd("shared_cpu_list")})
    except (OSError, ValueError):
        pass
    try:
        txt = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        out["lscpu"] = [ln for ln in txt.splitlines() if "cache" in ln.lower() or ln.startswith("Model name") or ln.startswith("CPU(s):")]
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def dense_recall(w, cue_phase, steps, dtype):
    s = np.exp(1j * np.asarray(cue_phase)).astype(dtype)
    for _ in range(steps):
        s = np.exp(1j * np.angle(w @ s)).astype(dtype, copy=False)
    return np.angle(s)


def build_dense(bits, dtype):
    bits = np.asarray(bits)
    p, n = bits.shape
    pm = np.exp(1j * np.where(bits == 1, np.pi, 0.0)).astype(dtype)
    pc = pm.conj()
    w = np.empty((n, n), dtype=dtype)
    rows = max(1, CHUNK_BYTES // (np.dtype(dtype).itemsize * n))
    for i0 in range(0, n, rows):
        i1 = min(n, i0 + rows)
        w[i0:i1] = pm[:, i0:i1].T @ pc
    idx = np.arange(n)
    w[idx, idx] = 0
    return w


def _mem_ok(peak, cap, wait_s=0):
    waited = 0
    while True:
        avail = r1._read_meminfo_available()
        if peak > cap:
            return False, f"peak {peak} B exceeds cap {cap} B", avail
        if avail is None or peak <= 0.5 * avail:
            return True, None, avail
        if waited >= wait_s:
            return False, f"peak {peak} B exceeds half of MemAvailable {avail} B at the time of the check (waited {waited} s)", avail
        time.sleep(30)
        waited += 30


def _per_op_batch(fn, samples, k, lc):
    res = b2.controlled_batch(lambda: [fn() for _ in range(k)], samples, lc)
    res["recall_seconds"] = [x / k for x in res["recall_seconds"]]
    for a in res["attempts"]:
        a["recall_seconds"] = [x / k for x in a["recall_seconds"]]
    res["k_ops_per_sample"] = k
    return res


def worker3(args_json):
    a = json.loads(args_json)
    rng = np.random.default_rng(a["seed"])
    out = {"environment": r1.environment_info(), "matvec": [], "streaming": [], "large": []}
    t_start = time.time()
    lc, cap = a["load_control"], a["max_peak_bytes"]
    mv = a["matvec"]
    for dname, dtype, ns in (("complex128", np.complex128, mv["n_list"]), ("complex64", np.complex64, mv["n_list"] + mv["n_list_complex64_extra"])):
        for n in ns:
            size = np.dtype(dtype).itemsize * n * n
            ok, why, avail = _mem_ok(size + CHUNK_BYTES, cap)
            if not ok:
                out["matvec"].append({"dtype": dname, "n_modes": n, "matrix_bytes": size, "measured": False, "reason": why})
                continue
            w = np.empty((n, n), dtype=dtype)
            w[:] = 0.5 + 0.5j
            s = np.exp(1j * rng.random(n) * 6.28).astype(dtype)
            for _ in range(mv["warmup"]):
                w @ s
            t0 = time.perf_counter()
            w @ s
            t1 = max(time.perf_counter() - t0, 1e-7)
            k = max(1, math.ceil(0.005 / t1))
            res = _per_op_batch(lambda: w @ s, mv["samples"], k, lc)
            out["matvec"].append({"dtype": dname, "n_modes": n, "matrix_bytes": size, "measured": True, **res})
            del w
    st = a["streaming"]
    for size in st["sizes_bytes"]:
        m = size // 8
        arr = np.ones(m, dtype=np.float64)
        for _ in range(st["warmup"]):
            arr.sum()
        t0 = time.perf_counter()
        arr.sum()
        t1 = max(time.perf_counter() - t0, 1e-7)
        k = max(1, math.ceil(0.005 / t1))
        res = _per_op_batch(lambda: arr.sum(), st["samples"], k, lc)
        out["streaming"].append({"bytes": int(m * 8), "measured": True, **res})
        del arr
    lg = a["large"]
    for n in lg["n_list"]:
        peak = 16 * n * n + CHUNK_BYTES
        ok, why, avail = _mem_ok(peak, cap, lg["wait_for_memory_seconds"])
        if not ok:
            out["large"].append({"n_modes": n, "measured": False, "reason": why})
            continue
        bits = rng.integers(0, 2, size=(8, n))
        w = build_dense(bits, np.complex128)
        cue = bits_to_wave(bits[0]).phase
        for _ in range(lg["warmup"]):
            dense_recall(w, cue, lg["steps"], np.complex128)
        res = b2.controlled_batch(lambda: dense_recall(w, cue, lg["steps"], np.complex128), lg["repeats"], lc)
        out["large"].append({"n_modes": n, "measured": True, "repeats": lg["repeats"], "weights_bytes": 16 * n * n,
                             "peak_bytes": peak, "mem_available_before_bytes": avail, "store_seconds": [0.0], **res})
        del w
    out["worker_wall_seconds"] = time.time() - t_start
    out["loadavg_1_5_15_at_end"] = list(os.getloadavg())
    return out


def run_worker3(mode, plan, seed):
    env = dict(os.environ)
    for k in r1.THREAD_VARS:
        if mode == "blas_threads_1":
            env[k] = "1"
        else:
            env.pop(k, None)
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps({**plan, "seed": seed})],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"round-3 worker failed ({mode}): {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def drop_events(series, key, threshold=0.75):
    """series: list of (size_bytes, bandwidth). Adjacent drops to <= threshold of the previous point."""
    ev = []
    for (s0, b0), (s1, b1) in zip(series, series[1:]):
        if b0 > 0 and b1 / b0 <= threshold:
            ev.append({key + "_lo": s0, key + "_hi": s1, "bandwidth_before": b0, "bandwidth_after": b1, "ratio": b1 / b0})
    return ev


def analyse_cache(raw, cinfo):
    caches = []
    for c in cinfo["sysfs"]:
        if c["type"] in ("Data", "Unified") and c["size_bytes"]:
            caches.append({"level": c["level"], "type": c["type"], "bytes": c["size_bytes"], "shared_cpu_list": c["shared_cpu_list"]})
    series = {}
    for dname in ("complex128", "complex64"):
        pts = []
        for m in raw["matvec"]:
            if m["dtype"] == dname and m["measured"] and m["load_controlled"]:
                t = float(np.median(m["recall_seconds"]))
                pts.append({"n_modes": m["n_modes"], "matrix_bytes": m["matrix_bytes"], "median_seconds": t, "bandwidth_gb_s": m["matrix_bytes"] / t / 1e9})
        pts.sort(key=lambda x: x["matrix_bytes"])
        slopes = [{"n_from": a["n_modes"], "n_to": b["n_modes"],
                   "slope": math.log(b["median_seconds"] / a["median_seconds"]) / math.log(b["n_modes"] / a["n_modes"])} for a, b in zip(pts, pts[1:])]
        series[dname] = {"points": pts, "local_slopes": slopes,
                         "drop_events": drop_events([(p["matrix_bytes"], p["bandwidth_gb_s"]) for p in pts], "bytes")}
    spts = []
    for m in raw["streaming"]:
        if m["measured"] and m["load_controlled"]:
            t = float(np.median(m["recall_seconds"]))
            spts.append({"bytes": m["bytes"], "median_seconds": t, "bandwidth_gb_s": m["bytes"] / t / 1e9})
    spts.sort(key=lambda x: x["bytes"])
    stream_events = drop_events([(p["bytes"], p["bandwidth_gb_s"]) for p in spts], "bytes")
    excluded = [{"what": f"matvec {m['dtype']} N={m['n_modes']}"} for m in raw["matvec"] if m["measured"] and not m["load_controlled"]] + \
               [{"what": f"streaming {m['bytes']} B"} for m in raw["streaming"] if m["measured"] and not m["load_controlled"]]
    ev = series["complex128"]["drop_events"]
    if not ev:
        m1 = {"verdict": "inconclusive", "reason": "no complex128 matvec drop event (bandwidth falling to <= 0.75 of the previous size)"}
        m2 = {"verdict": "inconclusive", "reason": "no matvec drop event"}
    else:
        hits = [{"event": e, "caches_bracketed": [c for c in caches if e["bytes_lo"] <= c["bytes"] <= e["bytes_hi"]]} for e in ev]
        m1 = {"verdict": "supported" if any(h["caches_bracketed"] for h in hits) else "rejected", "events": hits}
        overl = [{"matvec_event": e, "streaming_events_overlapping": [s for s in stream_events if not (s["bytes_hi"] < e["bytes_lo"] or s["bytes_lo"] > e["bytes_hi"])]} for e in ev]
        m2 = {"verdict": "supported" if any(o["streaming_events_overlapping"] for o in overl) else "rejected", "events": overl}
    return {"caches_reported_by_vm": caches, "matvec": series, "streaming_points": spts, "streaming_drop_events": stream_events,
            "excluded_uncontrolled_batches": excluded, "m1_reported_cache": m1, "m2_streaming_drop": m2}


def analyse_large(entries, base_sc, seed):
    ctrl = [{"n_modes": m["n_modes"], "measured": True, "recall_seconds": m["recall_seconds"], "store_seconds": [0.0], "weights_bytes": m["weights_bytes"]}
            for m in entries if m["measured"] and m["load_controlled"]]
    sc = {**base_sc, "fit_min_n": min([c["n_modes"] for c in ctrl] or [0]), "fit_min_points": 3,
          "holdout_check": {"fit_n_range": [0, 0], "predict_n": []}}
    an = r1.analyze_scaling(ctrl, sc, seed) if ctrl else {"status": "no load-controlled complex128 point"}
    return {"analysis": an,
            "unmeasured": [{"n_modes": m["n_modes"], "reason": m["reason"]} for m in entries if not m["measured"]],
            "excluded_uncontrolled": [m["n_modes"] for m in entries if m["measured"] and not m["load_controlled"]],
            "batches": [{"n_modes": m["n_modes"], "measured": m["measured"],
                         **({"load_controlled": m["load_controlled"], "attempts": m["attempts"]} if m["measured"] else {"reason": m["reason"]})} for m in entries]}


def timing_round3(profile_name, protocol, seed):
    cm, lg = protocol["cache_microbenchmark"], protocol["complex128_large"]
    full = profile_name == "full"
    cprof, lprof = cm["profiles"][profile_name], lg["profiles"][profile_name]
    base = r1.load_protocol(allow_change=True)[0]["scaling"]
    npts = cprof["streaming_points"]
    sizes = sorted({int(round(x / 4096)) * 4096 for x in np.geomspace(16 * 1024, 1024 ** 3, npts)})
    plan = {
        "load_control": {k: cm["load_control"][k] for k in ("threshold_external_busy_cores", "max_attempts", "retry_wait_seconds")},
        "max_peak_bytes": base["max_peak_bytes"],
        "matvec": {"n_list": cm["matvec"]["n_list"] if full else cprof["n_list"],
                   "n_list_complex64_extra": cm["matvec"]["n_list_complex64_extra"] if full else cprof["n_list_complex64_extra"],
                   "samples": cm["matvec"]["samples"] if full else cprof["samples"], "warmup": cm["matvec"]["warmup"] if full else cprof["warmup"]},
        "streaming": {"sizes_bytes": sizes, "samples": cm["streaming_read"]["samples"] if full else cprof["samples"],
                      "warmup": cm["streaming_read"]["warmup"] if full else cprof["warmup"]},
        "large": {"n_list": lg["n_list"] if full else lprof["n_list"], "repeats": lg["repeats"] if full else lprof["repeats"],
                  "warmup": lg["warmup"] if full else lprof["warmup"], "steps": lg["recall_steps"],
                  "wait_for_memory_seconds": 300 if full else 0},
    }
    cinfo = cache_info()
    sc = {**base, "ci_level": 0.95, "bootstrap_resamples": 2000, "recall_steps": lg["recall_steps"]}
    modes = {}
    for mode in cm["thread_modes"]:
        raw = run_worker3(mode, plan, seed)
        modes[mode] = {"environment": {**raw["environment"], "loadavg_1_5_15_at_end": raw["loadavg_1_5_15_at_end"], "worker_wall_seconds": raw["worker_wall_seconds"]},
                       "cache": analyse_cache(raw, cinfo),
                       "matvec_batches": [{k: m[k] for k in ("dtype", "n_modes", "measured") if k in m} | ({"load_controlled": m["load_controlled"], "attempts": len(m["attempts"])} if m["measured"] else {"reason": m["reason"]}) for m in raw["matvec"]],
                       "complex128_large": analyse_large(raw["large"], sc, seed)}
    return {"plan": plan, "cache_info": cinfo, "modes": modes}


# --------------------------------------------------------------------- main

def run3(profile_name, seed_override=None, allow_change=False):
    protocol, digest, locked = load_protocol3(allow_change)
    seed = protocol["seed"] if seed_override is None else seed_override
    mt = protocol["tests_for_multiplicity"]
    B, alpha = mt["permutations"][profile_name], mt["alpha"]
    mp, cp = protocol["mechanism"], protocol["correlated"]
    cells, trials, mism = bench_mechanism(mp["profiles"][profile_name], mp, seed)
    mech = analyse_mechanism(cells, trials, mp["profiles"][profile_name], mp, seed, B, alpha)
    mech["cells"] = cells
    mech["trajectory_vs_recall_mismatches"] = mism
    round2 = reanalyse_round2(profile_name, b2.load_protocol2(allow_change=True)[0]["seed"], alpha, B, seed)
    corr = bench_correlated(cp["profiles"][profile_name], cp, seed, B, alpha,
                            round2_rows=round2["round2_raw_rows_from_rerun"] if profile_name == "full" else None)
    round2_out = {k: v for k, v in round2.items() if k != "round2_raw_rows_from_rerun"}
    timing = timing_round3(profile_name, protocol, seed)
    deterministic = {"protocol_sha256": digest, "profile": profile_name, "seed": seed, "mechanism": mech,
                     "correlated": corr, "multiplicity_round2": round2_out, "timing_plan": timing["plan"]}
    return {
        "protocol": protocol["protocol"], "protocol_sha256": digest, "protocol_locked": locked,
        "profile": profile_name, "seed": seed, "deterministic_sha256": canonical_digest(deterministic),
        "hash_scope": protocol["determinism"]["hash_scope"],
        "mechanism": mech, "correlated": corr, "multiplicity_round2": round2_out, "timing": timing,
        "limitations": [
            "Synthetic classical software; no quantum-hardware claim.",
            "Hypothesis verdicts follow the pre-declared rules; 'partial_or_inconclusive' means the data fell between the support and reject thresholds.",
            "The mechanism analysis covers random and Hadamard patterns and bit-flip cues only.",
            "The cache microbenchmark can show consistency with a cache or bandwidth explanation, not prove it; TLB and hypervisor effects are not separated.",
        ],
    }


def summarize3(report):
    lines = [f"profile={report['profile']} protocol_locked={report['protocol_locked']} protocol_sha256={report['protocol_sha256']}"]
    m = report["mechanism"]
    lines.append(f"mechanism: x_cells={m['n_x_cells']} trajectory_vs_recall_mismatches={m['trajectory_vs_recall_mismatches']}")
    for k, v in m["hypotheses"].items():
        lines.append(f"hypothesis {k}: {v['verdict']}")
    c = report["correlated"]
    for r in c["per_rho_verdict_counts"]:
        lines.append(f"correlated rho={r['rho']}: raw={r['raw']} bh={r['bh']}")
    lines.append(f"hypothesis e_correlation: {c['hypothesis_e']['verdict']}")
    for name, f in report["multiplicity_round2"]["families"].items():
        lines.append(f"{name}: raw={f['raw_counts']} holm={f['holm']['counts']} bh={f['bh']['counts']}")
    rep = report["multiplicity_round2"]["stored_cells_reproduced"]
    if rep is not None:
        lines.append(f"round-2 stored cells reproduced: {rep['all_reproduced']} (mismatches {rep['mismatching_cells']})")
    for mode, t in report["timing"]["modes"].items():
        lines.append(f"cache[{mode}]: m1={t['cache']['m1_reported_cache']['verdict']} m2={t['cache']['m2_streaming_drop']['verdict']}")
        lg = t["complex128_large"]["analysis"]
        lines.append(f"complex128_large[{mode}]: measured N={[p['n_modes'] for p in lg.get('per_n', [])]}")
    lines.append(f"deterministic_sha256: {report['deterministic_sha256']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Round-3 cseq_wave benchmark")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-protocol-change", action="store_true")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        print(json.dumps(worker3(args.worker)))
        return 0
    if not (args.quick or args.full):
        ap.error("choose --quick or --full")
    report = run3("quick" if args.quick else "full", args.seed, args.allow_protocol_change)
    print(summarize3(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
        print("wrote:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
