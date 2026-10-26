"""Round-4 benchmark for cseq_wave (SYNTHETIC classical software).

Plan: benchmark_protocol_round4.json (hash in benchmark_protocol_round4.lock),
committed before this file existed and before any round-4 experiment was run.

  1. EARLY STEP: is accuracy at steps 2-5 reliably higher than at step 25 and
     at step 1, and does early stopping at a fixed step chosen on separate
     (selection) seeds beat the 25-step default and the cue on held-out seeds?
  2. SPURIOUS STATES: what kind of states does the dynamics end in at high load
     (stored, mixture_3, mixture_5, partial_overlap, unstructured)?
  3. THREADS: complex128 recall time versus BLAS thread count (1, 2, 3, 4,
     default) at N = 2048, 4096 (and 8192 as a declared extension).

Usage (repo root):

    python benchmark_round4.py --quick                              # CI smoke
    python benchmark_round4.py --full --json report_round4.json     # manual, all parts
    python benchmark_round4.py --full --part early --json early.json   # chunked
    python benchmark_round4.py --full --part threads --thread-run 1 --json thr1.json
    python benchmark_round4.py --assemble early.json spurious.json thr1.json thr2.json --json report_round4.json

No quantum-hardware claim. Timings, load, CPU/wall and memory decisions are
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
import benchmark_round3 as b3  # noqa: E402
from benchmark_rigorous import boot_ci, r6, wilson  # noqa: E402
from cseq_wave.evidence import canonical_digest  # noqa: E402

PROTOCOL_PATH = ROOT / "benchmark_protocol_round4.json"
LOCK_PATH = ROOT / "benchmark_protocol_round4.lock"
EARLY_WINDOW = (2, 3, 4, 5)


def load_protocol4(allow_change=False):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    digest = canonical_digest(protocol)
    locked = LOCK_PATH.read_text().strip() if LOCK_PATH.exists() else None
    ok = digest == locked
    if not ok and not allow_change:
        raise SystemExit(
            f"round-4 protocol hash {digest} does not match benchmark_protocol_round4.lock ({locked}). "
            "Use --allow-protocol-change to run anyway (recorded as protocol_locked=false)."
        )
    return protocol, digest, ok


def resolve_profile(block, profile):
    """Profile settings with the 'as listed' entries replaced by the block's top-level lists."""
    return {k: (block[k] if v == "as listed" else v) for k, v in block["profiles"][profile].items()}


def _n(x, key, profile):
    """Value that may be a {quick, full} dict."""
    return x[key][profile] if isinstance(x[key], dict) else x[key]


# ------------------------------------------------------------ early step

def early_cell_trials(n, q, qi, ratio, trials, seed, tag, check=False):
    """Per-trial bit accuracy to the target at steps 0..25 (step 0 = cue)."""
    p = int(round(n * ratio))
    assert abs(p / n - ratio) < 1e-12
    rng = np.random.default_rng([seed, tag, n, qi, p])
    acc = np.empty((trials, 26))
    mism = 0
    for i in range(trials):
        patterns = rng.integers(0, 2, size=(p, n))
        mem = b3.build_memory(patterns)
        target = patterns[0]
        flips = rng.random(n) < q
        cue_bits = np.where(flips, 1 - target, target).astype(np.int8)
        traj, _ = b3.trajectory(mem.weights, cue_bits, 25)
        acc[i] = (traj == target[None, :]).mean(axis=1)
        if check:
            ref = b2.wave_to_bits(b2.WaveState(np.ones(n), mem.recall(
                b2.WaveState(np.ones(n), cue_bits * np.pi), steps=25)[0]))
            mism += int(not np.array_equal(ref, traj[25]))
    return acc, mism


def pooled_bootstrap(gain_list, rng, B, level):
    """Equal-weight mean over cells of the cell mean; trials resampled within each cell."""
    tot = np.zeros(B)
    for g in gain_list:
        g = np.asarray(g, dtype=float)
        idx = rng.integers(0, len(g), size=(B, len(g)))
        tot += g[idx].mean(axis=1)
    tot /= len(gain_list)
    lo, hi = np.percentile(tot, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def early_verdict(n_higher, n_cells, pooled_mean, lo, hi):
    if n_cells == 0:
        return "not_testable"
    if n_higher >= math.ceil(0.75 * n_cells) and lo > 0 and pooled_mean >= 0.01:
        return "supported"
    if n_higher <= math.floor(0.25 * n_cells) or hi <= 0:
        return "rejected"
    return "partial_or_inconclusive"


def compare_family(gain_list, fam_id, seed, B, alpha):
    ps = [b3.signflip_p(g, B, np.random.default_rng([seed, 74, fam_id, i])) for i, g in enumerate(gain_list)]
    bh = b3.bh_adjust(ps)
    means = [float(np.mean(g)) for g in gain_list]
    verd = [b3.adjusted_verdict(a, m, alpha, "higher", "lower").replace("inconclusive", "not_significant") for a, m in zip(bh, means)]
    return ps, list(bh), means, verd


def bench_early(profile, ep, st, seed):
    prof = resolve_profile(ep, profile)
    sel_t = _n(ep["seed_sets"]["selection"], "trials_per_cell", profile)
    ho_t = _n(ep["seed_sets"]["held_out"], "trials_per_cell", profile)
    B = st["permutations"][profile]
    BB = st["bootstrap"]["resamples"][profile]
    alpha, level = st["alpha"], st["bootstrap"]["ci_level"]
    cands = ep["candidate_stop_steps"]

    def cells_for(ratios):
        return [(n, q, qi, ratio) for n in prof["n_modes"] for qi, q in enumerate(prof["q"]) for ratio in ratios]

    prim = cells_for(prof["p_over_n_primary"])
    contr = cells_for(prof["p_over_n_contrast_exploratory"])
    sel = {c: early_cell_trials(c[0], c[1], c[2], c[3], sel_t, seed, 71)[0] for c in prim + contr}
    ho, mism = {}, 0
    for c in prim + contr:
        ho[c], m = early_cell_trials(c[0], c[1], c[2], c[3], ho_t, seed, 72, check=True)
        mism += m

    def choose(cells_):
        scores = [float(np.mean([sel[c][:, s].mean() for c in cells_])) for s in cands]
        return cands[int(np.argmax(scores))], scores

    s_star, sel_scores = choose(prim)

    def gains(cells_, s):
        out = {"E1a_early_vs_step25": [], "E1b_early_vs_step1": [], "E2a_stop_vs_step25": [], "E2b_stop_vs_cue": []}
        for c in cells_:
            a = ho[c]
            early = a[:, list(EARLY_WINDOW)].mean(axis=1)
            out["E1a_early_vs_step25"].append(early - a[:, 25])
            out["E1b_early_vs_step1"].append(early - a[:, 1])
            out["E2a_stop_vs_step25"].append(a[:, s] - a[:, 25])
            out["E2b_stop_vs_cue"].append(a[:, s] - a[:, 0])
        return out

    def run_family(cells_, tag, with_verdict):
        g = gains(cells_, s_star)
        res = {}
        for fi, (name, lst) in enumerate(g.items()):
            ps, bh, means, verd = compare_family(lst, fi + (0 if tag == "primary" else 10), seed, B, alpha)
            brng = np.random.default_rng([seed, 75, fi, 0 if tag == "primary" else 1])
            lo, hi = pooled_bootstrap(lst, brng, BB, level)
            pooled = float(np.mean(means))
            counts = {k: verd.count(k) for k in ("higher", "not_significant", "lower")}
            entry = {"cells_higher": counts["higher"], "cells_not_significant": counts["not_significant"], "cells_lower": counts["lower"],
                     "pooled_mean_difference": r6(pooled), "pooled_ci": [r6(lo), r6(hi)],
                     "per_cell": [{"n_modes": c[0], "q": c[1], "p_over_n": c[3], "mean_difference": r6(m), "p_signflip": r6(p), "p_bh": r6(b), "verdict": v}
                                  for c, m, p, b, v in zip(cells_, means, ps, bh, verd)]}
            if with_verdict:
                entry["verdict"] = early_verdict(counts["higher"], len(cells_), pooled, lo, hi)
            res[name] = entry
        return res

    primary = run_family(prim, "primary", True)
    contrast = run_family(contr, "contrast", False) if contr else {}
    curves = [{"n_modes": c[0], "q": c[1], "p_over_n": c[3], "held_out_trials": ho_t,
               "accuracy_by_step_0_to_25": [r6(x) for x in ho[c].mean(axis=0)]} for c in prim + contr]
    percell = []
    for c in prim + contr:
        s_c = cands[int(np.argmax([sel[c][:, s].mean() for s in cands]))]
        d = ho[c][:, s_c] - ho[c][:, 25]
        percell.append({"n_modes": c[0], "q": c[1], "p_over_n": c[3], "s_star_cell": s_c,
                        "held_out_gain_over_step25": r6(d.mean()),
                        "held_out_gain_over_cue": r6((ho[c][:, s_c] - ho[c][:, 0]).mean())})
    return {
        "profile": profile, "selection_trials_per_cell": sel_t, "held_out_trials_per_cell": ho_t,
        "n_primary_cells": len(prim), "n_contrast_cells": len(contr),
        "s_star": s_star, "candidate_stop_steps": cands,
        "selection_only_not_for_inference": {"equal_weight_mean_accuracy_by_candidate": [r6(x) for x in sel_scores]},
        "primary": primary, "contrast_exploratory": contrast,
        "held_out_accuracy_curves": curves, "per_cell_selected_step_exploratory": percell,
        "trajectory_vs_recall_mismatches": mism,
    }


# ------------------------------------------------------------ spurious states

CATS4 = ["stored_target_pos", "stored_target_neg", "stored_other_pos", "stored_other_neg", "mixture_3", "mixture_5", "partial_overlap", "unstructured"]


def classify_state(bits, patterns, tau_stored=0.10, tau_mix=0.15):
    xi = 1 - 2 * np.asarray(patterns, dtype=float)
    x = 1 - 2 * np.asarray(bits, dtype=float)
    n = len(x)
    m = xi @ x / n
    a = np.abs(m)
    amax = float(a.max())
    if amax >= 1 - 2 * tau_stored - 1e-12:
        j = int(np.argmax(a))
        return ("stored_" + ("target" if j == 0 else "other") + ("_pos" if m[j] > 0 else "_neg")), amax
    order = np.argsort(-a, kind="stable")
    for k in (3, 5):
        if len(order) < k:
            continue
        top = order[:k]
        mk = np.sign((np.sign(m[top])[:, None] * xi[top]).sum(axis=0))
        if float(np.mean(mk != x)) <= tau_mix + 1e-12:
            return f"mixture_{k}", amax
    return ("partial_overlap" if amax >= 0.5 else "unstructured"), amax


def mean_direction_cosine(bits, patterns):
    xi = 1 - 2 * np.asarray(patterns, dtype=float)
    x = 1 - 2 * np.asarray(bits, dtype=float)
    s = xi.sum(axis=0)
    nrm = np.linalg.norm(s)
    return float(abs(x @ s) / (math.sqrt(len(x)) * nrm)) if nrm > 0 else 0.0


def energy_per_bit(w_real, bits):
    x = 1 - 2 * np.asarray(bits, dtype=float)
    return float(-(x @ (w_real @ x)) / (2 * len(x)))


def _is_stored(c):
    return c.startswith("stored")


def bench_spurious(profile, sp, st, seed):
    prof = resolve_profile(sp, profile)
    trials = _n(sp, "trials_per_cell", profile)
    df = sp["definitions"]
    tau, tmix = df["tau_stored"], df["tau_mixture"]
    sens = [(s["tau_stored"], s["tau_mixture"]) for s in sp["sensitivity"]]
    cells, nonstored = [], []
    sens_counts = {f"{a}/{b}": {"stored": 0, "mixture": 0, "partial_overlap": 0, "unstructured": 0} for a, b in sens}
    ctrl = {"stored_5pct_flips_is_stored": [0, 0], "mixture3_5pct_flips_is_mixture3": [0, 0],
            "mixture5_5pct_flips_is_mixture5": [0, 0], "mixture5_5pct_flips_is_mixture3_or_5": [0, 0],
            "null_random_is_stored": [0, 0], "null_random_is_mixture": [0, 0]}
    ctrl_tot = 0
    ctrl_n = {n: {"total": 0, **{k: 0 for k in ctrl}} for n in prof["n_modes"]}
    for n in prof["n_modes"]:
        for qi, q in enumerate(prof["q"]):
            for ratio in prof["p_over_n"]:
                p = int(round(n * ratio))
                assert abs(p / n - ratio) < 1e-12
                rng = np.random.default_rng([seed, 73, n, qi, p])
                counts = {c: 0 for c in CATS4}
                dyn_by = {c: {d: 0 for d in b3.DYN} for c in CATS4}
                maxa_all, maxa_ns, ef, es = [], [], [], []
                for _ in range(trials):
                    patterns = rng.integers(0, 2, size=(p, n))
                    mem = b3.build_memory(patterns)
                    target = patterns[0]
                    flips = (rng.random(n) < q) if q > 0 else np.zeros(n, dtype=bool)
                    cue_bits = np.where(flips, 1 - target, target).astype(np.int8)
                    traj, _ = b3.trajectory(mem.weights, cue_bits, 100)
                    final = traj[100]
                    cat, amax = classify_state(final, patterns, tau, tmix)
                    counts[cat] += 1
                    dc = b3.dynamics_class(traj)
                    dyn_by[cat][dc] += 1
                    maxa_all.append(amax)
                    for (ta, tm) in sens:
                        c2, _ = classify_state(final, patterns, ta, tm)
                        key = f"{ta}/{tm}"
                        if _is_stored(c2):
                            sens_counts[key]["stored"] += 1
                        elif c2.startswith("mixture"):
                            sens_counts[key]["mixture"] += 1
                        else:
                            sens_counts[key][c2] += 1
                    if not _is_stored(cat):
                        wr = mem.weights.real
                        null_bits = rng.integers(0, 2, size=n)
                        nonstored.append({"cat": cat, "dyn": dc, "cos": mean_direction_cosine(final, patterns),
                                          "cos_null": mean_direction_cosine(null_bits, patterns)})
                        maxa_ns.append(amax)
                        ef.append(energy_per_bit(wr, final))
                        es.append(float(np.mean([energy_per_bit(wr, patterns[j]) for j in range(min(p, 4))])))
                    # harness controls (independent draws, same trial)
                    ctrl_tot += 1
                    ctrl_n[n]["total"] += 1
                    idx = rng.permutation(p)
                    xi = 1 - 2 * patterns.astype(int)

                    def flip5(v):
                        f = rng.random(n) < 0.05
                        return np.where(f, -v, v)

                    s_bits = ((1 - flip5(xi[idx[0]])) // 2).astype(int)
                    ctrl["stored_5pct_flips_is_stored"][0] += int(_is_stored(classify_state(s_bits, patterns, tau, tmix)[0]))
                    m3 = np.sign(xi[idx[:3]].sum(axis=0))
                    c3 = classify_state(((1 - flip5(m3)) // 2).astype(int), patterns, tau, tmix)[0]
                    ctrl["mixture3_5pct_flips_is_mixture3"][0] += int(c3 == "mixture_3")
                    if p >= 5:
                        m5 = np.sign(xi[idx[:5]].sum(axis=0))
                        c5 = classify_state(((1 - flip5(m5)) // 2).astype(int), patterns, tau, tmix)[0]
                        ctrl["mixture5_5pct_flips_is_mixture5"][0] += int(c5 == "mixture_5")
                        ctrl["mixture5_5pct_flips_is_mixture3_or_5"][0] += int(c5 in ("mixture_3", "mixture_5"))
                    cn = classify_state(rng.integers(0, 2, size=n), patterns, tau, tmix)[0]
                    ctrl["null_random_is_stored"][0] += int(_is_stored(cn))
                    ctrl["null_random_is_mixture"][0] += int(cn.startswith("mixture"))
                    for k in ctrl:
                        ctrl_n[n][k] = ctrl[k][0] - sum(ctrl_n[m][k] for m in ctrl_n if m != n)
                cells.append({"n_modes": n, "q": q, "n_patterns": p, "p_over_n": ratio, "trials": trials,
                              "category_counts": counts, "dynamics_by_category": dyn_by,
                              "max_overlap_all_states_quantiles_5_25_50_75_95": [r6(x) for x in np.percentile(maxa_all, [5, 25, 50, 75, 95])],
                              "nearest_stored_symmetric_distance_median_all": r6((1 - np.median(maxa_all)) / 2),
                              "nearest_stored_symmetric_distance_median_non_stored": r6((1 - np.median(maxa_ns)) / 2) if maxa_ns else None,
                              "energy_per_bit_final_non_stored_mean": r6(np.mean(ef)) if ef else None,
                              "energy_per_bit_stored_mean_same_trials": r6(np.mean(es)) if es else None})
    for k in ctrl:
        ctrl[k][1] = ctrl_tot
    ctrl_rates = {k: r6(v[0] / v[1]) if v[1] else None for k, v in ctrl.items()}
    n_ns = len(nonstored)
    n_all = sum(c["trials"] for c in cells)

    def frac(pred):
        return sum(1 for s in nonstored if pred(s))

    k_mix, k_part, k_unst, k_fix = (frac(lambda s: s["cat"].startswith("mixture")), frac(lambda s: s["cat"] == "partial_overlap"),
                                    frac(lambda s: s["cat"] == "unstructured"), frac(lambda s: s["dyn"] == "fixed_point"))
    valid = (ctrl_rates["stored_5pct_flips_is_stored"] >= 0.95 and ctrl_rates["mixture3_5pct_flips_is_mixture3"] >= 0.95
             and ctrl_rates["null_random_is_stored"] + ctrl_rates["null_random_is_mixture"] <= 0.05)
    hy = sp["hypotheses"]

    def wv(k, key):
        if not valid and key != "Sp4_fixed_points":
            return "harness_invalid"
        h = hy[key]
        return b3.wilson_verdict(k, n_ns, float(h["supported"].split(">=")[1]), float(h["rejected"].split("<=")[1]))

    def wil(k):
        lo, hi = wilson(k, n_ns) if n_ns else (float("nan"), float("nan"))
        return [r6(lo), r6(hi)]

    diff = np.array([s["cos"] - s["cos_null"] for s in nonstored]) if nonstored else np.array([])
    if len(diff):
        lo, hi = boot_ci(diff, np.random.default_rng([seed, 76]), st["bootstrap"]["resamples"][profile], st["bootstrap"]["ci_level"])
        sp5 = "supported" if (lo > 0 and diff.mean() >= 0.10) else ("rejected" if hi < 0.05 else "partial_or_inconclusive")
        sp5d = {"mean_cosine_final": r6(np.mean([s["cos"] for s in nonstored])), "mean_cosine_null": r6(np.mean([s["cos_null"] for s in nonstored])),
                "mean_difference": r6(diff.mean()), "ci": [r6(lo), r6(hi)]}
    else:
        sp5, sp5d = "not_testable", {}
    cat_tot = {c: sum(x["category_counts"][c] for x in cells) for c in CATS4}
    dyn_tot = {c: {d: sum(x["dynamics_by_category"][c][d] for x in cells) for d in b3.DYN} for c in CATS4}
    return {
        "profile": profile, "tau_stored": tau, "tau_mixture": tmix, "n_final_states": n_all, "n_non_stored": n_ns,
        "pooled_category_counts": cat_tot, "pooled_dynamics_by_category": dyn_tot,
        "pooled_sensitivity_counts": sens_counts,
        "harness": {"control_rates": ctrl_rates, "controls_per_rate": ctrl_tot, "valid": bool(valid),
                    "control_rates_by_n_descriptive": {str(n): {k: r6(v / d["total"]) for k, v in d.items() if k != "total"} for n, d in ctrl_n.items() if d["total"]}},
        "hypotheses": {
            "Sp1_mixture_states": {"k": k_mix, "n": n_ns, "wilson_95": wil(k_mix), "verdict": wv(k_mix, "Sp1_mixture_states")},
            "Sp2_unstructured": {"k": k_unst, "n": n_ns, "wilson_95": wil(k_unst), "verdict": wv(k_unst, "Sp2_unstructured")},
            "Sp3_partial_overlap": {"k": k_part, "n": n_ns, "wilson_95": wil(k_part), "verdict": wv(k_part, "Sp3_partial_overlap")},
            "Sp4_fixed_points": {"k": k_fix, "n": n_ns, "wilson_95": wil(k_fix), "verdict": wv(k_fix, "Sp4_fixed_points")},
            "Sp5_mean_direction": {**sp5d, "verdict": sp5 if valid else "harness_invalid"},
        },
        "cells": cells,
    }


# ------------------------------------------------------------ threads

def worker4(args_json):
    a = json.loads(args_json)
    rng = np.random.default_rng(a["seed"])
    out = {"environment": r1.environment_info(), "large": []}
    t_start = time.time()
    lc, cap = a["load_control"], a["max_peak_bytes"]
    for n in a["n_list"]:
        peak = 16 * n * n + b3.CHUNK_BYTES
        ok, why, avail = b3._mem_ok(peak, cap, a["wait_for_memory_seconds"])
        if not ok:
            out["large"].append({"n_modes": n, "measured": False, "reason": why})
            continue
        bits = rng.integers(0, 2, size=(8, n))
        w = b3.build_dense(bits, np.complex128)
        cue = b3.bits_to_wave(bits[0]).phase
        for _ in range(a["warmup"]):
            b3.dense_recall(w, cue, a["steps"], np.complex128)
        res = b2.controlled_batch(lambda: b3.dense_recall(w, cue, a["steps"], np.complex128), a["repeats"], lc)
        c0, t0 = time.process_time(), time.perf_counter()
        for _ in range(3):
            b3.dense_recall(w, cue, a["steps"], np.complex128)
        cpu_over_wall = (time.process_time() - c0) / (time.perf_counter() - t0)
        out["large"].append({"n_modes": n, "measured": True, "repeats": a["repeats"], "peak_bytes": peak, "mem_available_before_bytes": avail,
                             "cpu_over_wall": round(cpu_over_wall, 3), **res})
        del w
    out["worker_wall_seconds"] = time.time() - t_start
    out["loadavg_1_5_15_at_end"] = list(os.getloadavg())
    return out


def run_worker4(mode, plan, seed):
    env = dict(os.environ)
    for k in r1.THREAD_VARS:
        if mode == "default":
            env.pop(k, None)
        else:
            env[k] = str(int(mode))
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps({**plan, "seed": seed})],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"round-4 worker failed ({mode}): {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def threads_plan(profile, th):
    prof = th["profiles"][profile]
    full = profile == "full"
    modes = [str(k) for k in prof["thread_counts"]] + (["default"] if prof["include_default"] else [])
    n_list = prof["n_modes"]
    return {
        "n_list": n_list, "thread_modes": modes, "runs": th["runs"][profile],
        "run_order": {"run1": modes, "run2": list(reversed(modes))},
        "repeats": th["repeats"][profile], "warmup": th["warmup"][profile], "steps": th["recall_steps"],
        "load_control": {k: th["load_control"][k] for k in ("threshold_external_busy_cores", "max_attempts", "retry_wait_seconds")},
        "max_peak_bytes": 4 * 1024 ** 3, "wait_for_memory_seconds": 300 if full else 0,
    }


def med_ci(vals, rng, B, level):
    lo, hi = boot_ci(vals, rng, B, level, stat=np.median)
    return float(np.median(vals)), lo, hi


def speedup_ci(v1, vk, rng, B, level):
    v1, vk = np.asarray(v1, float), np.asarray(vk, float)
    i1 = rng.integers(0, len(v1), size=(B, len(v1)))
    ik = rng.integers(0, len(vk), size=(B, len(vk)))
    r = np.median(v1[i1], axis=1) / np.median(vk[ik], axis=1)
    lo, hi = np.percentile(r, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(np.median(v1) / np.median(vk)), float(lo), float(hi)


def analyse_threads(raw_runs, plan, st, seed, profile):
    """raw_runs: {run_index: {mode: worker_output}}"""
    B, level = st["bootstrap"]["resamples"][profile], st["bootstrap"]["ci_level"]
    tab, excluded, unmeasured = {}, [], []
    for run, modes in raw_runs.items():
        for mode, raw in modes.items():
            for e in raw["large"]:
                if not e["measured"]:
                    unmeasured.append({"run": run, "mode": mode, "n_modes": e["n_modes"], "reason": e["reason"]})
                    continue
                if not e["load_controlled"]:
                    excluded.append({"run": run, "mode": mode, "n_modes": e["n_modes"]})
                    continue
                tab[(int(run), mode, e["n_modes"])] = e
    rows = []
    for (run, mode, n), e in sorted(tab.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        rng = np.random.default_rng([seed, 77, run, n, hash_mode(mode)])
        med, lo, hi = med_ci(e["recall_seconds"], rng, B, level)
        row = {"run": run, "mode": mode, "n_modes": n, "median_seconds": med, "ci_median": [lo, hi], "repeats": len(e["recall_seconds"]),
               "cpu_over_wall": e["cpu_over_wall"], "attempts": len(e["attempts"]),
               "external_busy_cores_accepted": e["attempts"][e["accepted_attempt"]]["external_busy_cores"]}
        if mode != "1" and (run, "1", n) in tab:
            sp, slo, shi = speedup_ci(tab[(run, "1", n)]["recall_seconds"], e["recall_seconds"], rng, B, level)
            row.update({"speedup_vs_1": sp, "speedup_ci": [slo, shi]})
            if mode != "default":
                row["efficiency"] = sp / int(mode)
        rows.append(row)
    med = {(r["run"], r["mode"], r["n_modes"]): r["median_seconds"] for r in rows}
    runs = sorted({r["run"] for r in rows})
    ns = sorted({r["n_modes"] for r in rows})
    ks = sorted({int(r["mode"]) for r in rows if r["mode"] != "default"})
    # T1
    units = [r for r in rows if r["mode"] not in ("1", "default") and "speedup_ci" in r]
    k1 = sum(1 for r in units if r["speedup_ci"][0] > 1.10)
    t1 = _frac_verdict(k1, len(units), 0.80, 0.20)
    # T2
    t2u = [(run, n) for run in runs for n in ns if all((run, str(k), n) in med for k in (1, 2, 3, 4))]
    k2 = sum(1 for run, n in t2u if med[(run, "1", n)] > med[(run, "2", n)] > med[(run, "3", n)] > med[(run, "4", n)])
    t2 = "not_testable" if not t2u else ("supported" if k2 >= math.ceil(0.8 * len(t2u)) else ("rejected" if k2 <= math.floor(0.2 * len(t2u)) else "partial_or_inconclusive"))
    # T3
    modes_all = sorted({r["mode"] for r in rows})
    t3u = [(n, m) for n in ns for m in modes_all if (1, m, n) in med and (2, m, n) in med]
    k3 = sum(1 for n, m in t3u if 0.8 <= med[(2, m, n)] / med[(1, m, n)] <= 1.25)
    t3 = _frac_verdict(k3, len(t3u), 0.80, 0.50)
    # T4
    est, t4 = [], "not_testable"
    if 4096 in ns and 8192 in ns:
        for run in runs:
            for m in modes_all:
                if (run, m, 4096) in med and (run, m, 8192) in med:
                    est.append({"run": run, "mode": m, "estimate_t16384_seconds_EXTRAPOLATED_NOT_MEASURED": med[(run, m, 8192)] ** 2 / med[(run, m, 4096)]})
        if est:
            key = "estimate_t16384_seconds_EXTRAPOLATED_NOT_MEASURED"
            reach = all(any(e[key] <= 3.5 for e in est if e["run"] == run and e["mode"] in ("2", "3", "4")) for run in runs)
            none = all(e[key] > 3.5 for e in est if e["mode"] in ("1", "2", "3", "4"))
            t4 = "supported" if reach else ("rejected" if none else "partial_or_inconclusive")
    flagged = [r for r in rows if r["mode"] not in ("1", "default") and int(r["mode"]) >= 2 and r["n_modes"] >= 4096 and r["cpu_over_wall"] < 1.3]
    return {
        "rows": rows, "excluded_uncontrolled_batches": excluded, "unmeasured": unmeasured,
        "thread_effect_flags": [{"run": r["run"], "mode": r["mode"], "n_modes": r["n_modes"], "cpu_over_wall": r["cpu_over_wall"]} for r in flagged],
        "hypotheses": {
            "T1_speedup": {"units": len(units), "satisfying": k1, "verdict": t1},
            "T2_monotone": {"units": len(t2u), "satisfying": k2, "verdict": t2},
            "T3_repeatability": {"units": len(t3u), "satisfying": k3, "verdict": t3},
            "T4_original_claim": {"estimates": est, "verdict": t4, "note": "EXTRAPOLATED, NOT MEASURED; complex128 at N = 16384 is not measured"},
        },
    }


def hash_mode(mode):
    return 99 if mode == "default" else int(mode)


def _frac_verdict(k, n, sup, rej):
    if n == 0:
        return "not_testable"
    if k >= math.ceil(sup * n):
        return "supported"
    if k <= math.floor(rej * n):
        return "rejected"
    return "partial_or_inconclusive"


def timing_round4(profile, protocol, seed, only_run=None):
    th = protocol["threads"]
    plan = threads_plan(profile, th)
    raw_runs = {}
    for ri in range(1, plan["runs"] + 1):
        if only_run is not None and ri != only_run:
            continue
        raw_runs[ri] = {}
        for mode in plan["run_order"][f"run{ri}"] if f"run{ri}" in plan["run_order"] else plan["run_order"]["run1"]:
            raw_runs[ri][mode] = run_worker4(mode, plan, seed)
            print(f"threads: run {ri} mode {mode} done", file=sys.stderr, flush=True)
    an = analyse_threads(raw_runs, plan, protocol["statistics"], seed, profile)
    envs = {f"run{ri}_{m}": {**raw["environment"], "loadavg_1_5_15_at_end": raw["loadavg_1_5_15_at_end"], "worker_wall_seconds": raw["worker_wall_seconds"]}
            for ri, modes in raw_runs.items() for m, raw in modes.items()}
    batches = [{"run": ri, "mode": m, "n_modes": e["n_modes"], "measured": e["measured"],
                **({"load_controlled": e["load_controlled"], "attempts": e["attempts"]} if e["measured"] else {"reason": e["reason"]})}
               for ri, modes in raw_runs.items() for m, raw in modes.items() for e in raw["large"]]
    return {"plan": plan, "analysis": an, "environments": envs, "batches": batches, "runs_included": sorted(raw_runs)}


# ------------------------------------------------------------ main

def build_report(profile, protocol, digest, locked, seed, early, spurious, timing):
    deterministic = {"protocol_sha256": digest, "profile": profile, "seed": seed, "early_step": early,
                     "spurious_states": spurious, "timing_plan": timing["plan"]}
    return {
        "protocol": protocol["protocol"], "protocol_sha256": digest, "protocol_locked": locked, "profile": profile, "seed": seed,
        "deterministic_sha256": canonical_digest(deterministic), "hash_scope": protocol["determinism"]["hash_scope"],
        "early_step": early, "spurious_states": spurious, "timing": timing,
        "limitations": [
            "Synthetic classical software; no quantum-hardware, QEC or quantum-advantage claim.",
            "Early-step results hold for random patterns, bit-flip cues and the listed N and P/N only.",
            "Spurious-state categories are heuristic (greedy mixture detection); a state can be near a mixture of other patterns than the top-k overlaps.",
            "Thread timings are for this shared VM; a thread setting that did not take effect is flagged through cpu_over_wall.",
        ],
    }


def run4(profile, seed_override=None, allow_change=False, part="all", thread_run=None):
    protocol, digest, locked = load_protocol4(allow_change)
    seed = protocol["seed"] if seed_override is None else seed_override
    st = protocol["statistics"]
    early = bench_early(profile, protocol["early_step"], st, seed) if part in ("all", "early") else None
    spurious = bench_spurious(profile, protocol["spurious_states"], st, seed) if part in ("all", "spurious") else None
    timing = timing_round4(profile, protocol, seed, thread_run) if part in ("all", "threads") else {"plan": threads_plan(profile, protocol["threads"])}
    rep = build_report(profile, protocol, digest, locked, seed, early, spurious, timing)
    rep["part"] = part
    return rep


def assemble(paths, json_out=None):
    reps = [json.loads(Path(p).read_text()) for p in paths]
    base = reps[0]
    early = next((r["early_step"] for r in reps if r.get("early_step")), None)
    spur = next((r["spurious_states"] for r in reps if r.get("spurious_states")), None)
    plans = [r["timing"]["plan"] for r in reps]
    assert all(p == plans[0] for p in plans), "timing plans differ"
    thr = [r["timing"] for r in reps if "analysis" in r["timing"]]
    raw_note = "analysis of merged runs is not recomputed from raw batches; per-run analyses are kept"
    protocol, digest, locked = load_protocol4(True)
    deterministic = {"protocol_sha256": base["protocol_sha256"], "profile": base["profile"], "seed": base["seed"],
                     "early_step": early, "spurious_states": spur, "timing_plan": plans[0]}
    return {"protocol": base["protocol"], "protocol_sha256": base["protocol_sha256"], "protocol_locked": base["protocol_locked"],
            "profile": base["profile"], "seed": base["seed"], "deterministic_sha256": canonical_digest(deterministic),
            "hash_scope": base["hash_scope"], "early_step": early, "spurious_states": spur,
            "timing": {"plan": plans[0], "per_run_analyses": [{"runs_included": t["runs_included"], "analysis": t["analysis"]} for t in thr], "note": raw_note},
            "assembled_from": [str(p) for p in paths]}


def summarize4(report):
    lines = [f"profile={report['profile']} protocol_locked={report['protocol_locked']} protocol_sha256={report['protocol_sha256']}"]
    e = report.get("early_step")
    if e:
        lines.append(f"early: s_star={e['s_star']} mismatches={e['trajectory_vs_recall_mismatches']}")
        for k, v in e["primary"].items():
            lines.append(f"early {k}: {v['verdict']} higher={v['cells_higher']}/{e['n_primary_cells']} pooled={v['pooled_mean_difference']} ci={v['pooled_ci']}")
    s = report.get("spurious_states")
    if s:
        lines.append(f"spurious: harness_valid={s['harness']['valid']} non_stored={s['n_non_stored']}/{s['n_final_states']}")
        for k, v in s["hypotheses"].items():
            lines.append(f"spurious {k}: {v['verdict']}")
    t = report["timing"]
    if "analysis" in t:
        for k, v in t["analysis"]["hypotheses"].items():
            lines.append(f"threads {k}: {v['verdict']}")
    lines.append(f"deterministic_sha256: {report['deterministic_sha256']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Round-4 cseq_wave benchmark")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--part", choices=["all", "early", "spurious", "threads"], default="all")
    ap.add_argument("--thread-run", type=int, default=None, help="only this run index of the threads part")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--assemble", nargs="+", default=None, help="merge chunk JSON reports and recompute the overall hash")
    ap.add_argument("--allow-protocol-change", action="store_true")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        print(json.dumps(worker4(args.worker)))
        return 0
    dump = lambda rep: Path(args.json).write_text(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    if args.assemble:
        rep = assemble(args.assemble)
        print(f"deterministic_sha256: {rep['deterministic_sha256']}")
        if args.json:
            dump(rep)
            print("wrote:", args.json)
        return 0
    if not (args.quick or args.full):
        ap.error("choose --quick or --full")
    report = run4("quick" if args.quick else "full", args.seed, args.allow_protocol_change, args.part, args.thread_run)
    print(summarize4(report))
    if args.json:
        dump(report)
        print("wrote:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
