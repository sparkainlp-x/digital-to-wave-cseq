"""Rigorous benchmark for cseq_wave (SYNTHETIC classical software).

Companion to benchmark.py, which is kept unchanged. This file adds:

  1. SCALING: direct timings for N = 64..8192 (never allocating more than
     the protocol's peak-memory cap), warm-up, >= 7 repeats, median / IQR /
     bootstrap CI per N, BLAS threads pinned to 1 and default (separate
     subprocesses), a log-log fit on a range chosen a priori, bootstrap CI
     for alpha, local slopes between adjacent sizes, comparison with the
     theoretical N^2 matvec cost, a hold-out check of the extrapolation,
     and an interval-labelled extrapolation to unmeasured N.
  2. CAPACITY: P/N de-confounded from N (same P/N at N = 32..256), >= 30
     trials, bootstrap and Wilson CIs, noise sigma as a second axis, and
     null / reference baselines.
  3. REVERSE: adversarial cases and a many-seed property loop.
  4. PROTOCOL LOCK: the analysis plan lives in benchmark_protocol.json; its
     SHA-256 (cseq_wave.evidence.canonical_digest) is checked against
     benchmark_protocol.lock and recorded in the report. The report's
     deterministic hash excludes everything derived from timings.

Usage (repo root):

    python benchmark_rigorous.py --quick                       # CI smoke, seconds
    python benchmark_rigorous.py --full --json report_rigorous.json   # manual

No quantum-hardware claim. Noise is Gaussian phase noise added to cue
phases. Timings are machine-dependent and are not part of the hash.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cseq_wave.encode import (  # noqa: E402
    bits_to_text, bits_to_wave, text_to_bits, wave_to_bits,
)
from cseq_wave.evidence import canonical_digest  # noqa: E402
from cseq_wave.memory import HolographicMemory  # noqa: E402
from cseq_wave.wave import WaveState  # noqa: E402

PROTOCOL_PATH = ROOT / "benchmark_protocol.json"
LOCK_PATH = ROOT / "benchmark_protocol.lock"
THREAD_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]


# ----------------------------------------------------------------- protocol

def load_protocol(allow_change=False):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    digest = canonical_digest(protocol)
    locked = LOCK_PATH.read_text().strip() if LOCK_PATH.exists() else None
    ok = digest == locked
    if not ok and not allow_change:
        raise SystemExit(
            f"protocol hash {digest} does not match benchmark_protocol.lock ({locked}). "
            "Use --allow-protocol-change to run anyway (recorded as protocol_locked=false)."
        )
    return protocol, digest, ok


# --------------------------------------------------------------- statistics

def wilson(k, n, z=1.959964):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def boot_ci(values, rng, resamples, level, stat=np.mean):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    stats = stat(values[idx], axis=1)
    lo, hi = np.percentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def r6(x):
    return None if x is None else round(float(x), 6)


# ------------------------------------------------------------------ reverse

def _random_text(rng):
    n = int(rng.integers(1, 12))  # empty text is an adversarial case, tested separately
    chars = []
    for _ in range(n):
        cp = int(rng.integers(0x20, 0x2FFF))
        if 0xD800 <= cp <= 0xDFFF:
            cp = 0x41
        chars.append(chr(cp))
    return "".join(chars)


def bench_reverse(profile, seed, max_length):
    failures = []
    cases = {"codec_exact": 0, "wave_exact": 0, "text_exact": 0, "noisy_sigma_0.1_exact": 0}
    kinds = ["random", "zeros", "ones", "alternating", "single_one", "single_zero"]
    for s in range(profile["seeds"]):
        rng = np.random.default_rng([seed, 7, s])
        length = int(rng.integers(1, max_length + 1))
        kind = kinds[s % len(kinds)]
        if kind == "random":
            bits = rng.integers(0, 2, size=length)
        elif kind == "zeros":
            bits = np.zeros(length, dtype=int)
        elif kind == "ones":
            bits = np.ones(length, dtype=int)
        elif kind == "alternating":
            bits = np.arange(length) % 2
        elif kind == "single_one":
            bits = np.zeros(length, dtype=int)
            bits[int(rng.integers(0, length))] = 1
        else:
            bits = np.ones(length, dtype=int)
            bits[int(rng.integers(0, length))] = 0
        ws = bits_to_wave(bits)
        if np.array_equal(wave_to_bits(ws), bits):
            cases["codec_exact"] += 1
        else:
            failures.append({"check": "codec_sigma0", "seed": s, "length": length, "kind": kind})
        back = bits_to_wave(wave_to_bits(ws))
        err = float(np.max(np.abs(np.angle(np.exp(1j * (back.phase - ws.phase))))))
        if err == 0.0:
            cases["wave_exact"] += 1
        else:
            failures.append({"check": "wave_roundtrip", "seed": s, "length": length, "kind": kind, "err": err})
        noisy = WaveState(ws.amplitude, ws.phase + rng.normal(0.0, 0.1, size=length))
        if np.array_equal(wave_to_bits(noisy), bits):
            cases["noisy_sigma_0.1_exact"] += 1
        else:
            failures.append({"check": "codec_sigma0.1", "seed": s, "length": length, "kind": kind})
        text = _random_text(rng)
        if bits_to_text(wave_to_bits(bits_to_wave(text_to_bits(text)))) == text:
            cases["text_exact"] += 1
        else:
            failures.append({"check": "text_roundtrip", "seed": s, "text": ascii(text)})

    adversarial = []

    def case(name, fn):
        try:
            adversarial.append({"case": name, "outcome": fn()})
        except Exception as exc:  # recorded, not hidden
            adversarial.append({"case": name, "outcome": f"{type(exc).__name__}: {exc}"})

    case("empty_bits_encode", lambda: bits_to_wave(np.array([], dtype=int)).phase.tolist())
    case("empty_bits_round_trip", lambda: wave_to_bits(bits_to_wave(np.array([], dtype=int))).tolist())
    case("length_1_zero", lambda: wave_to_bits(bits_to_wave(np.array([0]))).tolist())
    case("length_1_one", lambda: wave_to_bits(bits_to_wave(np.array([1]))).tolist())
    case("bits_2d", lambda: bits_to_wave(np.array([[0, 1]])).phase.tolist())
    case("bits_value_2", lambda: bits_to_wave(np.array([0, 2])).phase.tolist())
    case("empty_text_to_bits_and_back_without_wave", lambda: repr(bits_to_text(text_to_bits(""))))
    case("empty_text_through_wave", lambda: repr(bits_to_text(wave_to_bits(bits_to_wave(text_to_bits(""))))))
    case("non_multiple_of_8_bits_to_text", lambda: repr(bits_to_text(np.array([1, 0, 1]))))

    def single_mode_memory():
        mem = HolographicMemory(1)
        mem.store(bits_to_wave(np.array([1])))
        mem.finalize()
        phases, _ = mem.recall(bits_to_wave(np.array([1])), steps=3)
        return {"recalled_bits": wave_to_bits(WaveState(np.ones(1), phases)).tolist(),
                "note": "diagonal removal leaves a 1x1 zero weight matrix, so N=1 cannot hold a pattern"}

    case("single_mode_memory_stores_1", single_mode_memory)
    case("memory_n_modes_0", lambda: HolographicMemory(0).n_modes)
    return {
        "seeds": profile["seeds"],
        "max_length": max_length,
        "counts": cases,
        "failures": failures,
        "adversarial": adversarial,
    }


# ----------------------------------------------------------------- capacity

def bench_capacity(profile, p_over_n, seed, steps, knee_threshold, resamples, level):
    rows = []
    for n in profile["n_modes"]:
        for si, sigma in enumerate(profile["sigmas"]):
            for ratio in p_over_n:
                p = int(round(n * ratio))
                assert abs(p / n - ratio) < 1e-12, (n, ratio)
                rng = np.random.default_rng([seed, 11, n, si, p])
                acc, cue_acc, nearest_acc, rand_acc, exact = [], [], [], [], 0
                for _ in range(profile["trials"]):
                    patterns = rng.integers(0, 2, size=(p, n))
                    mem = HolographicMemory(n)
                    for bits in patterns:
                        mem.store(bits_to_wave(bits))
                    mem.finalize()
                    target = patterns[0]
                    cue_phase = bits_to_wave(target).phase + (rng.normal(0.0, sigma, size=n) if sigma > 0 else 0.0)
                    phases, _ = mem.recall(WaveState(np.ones(n), cue_phase), steps=steps)
                    got = wave_to_bits(WaveState(np.ones(n), phases))
                    acc.append(float(np.mean(got == target)))
                    exact += int(np.array_equal(got, target))
                    cue_bits = wave_to_bits(WaveState(np.ones(n), cue_phase))
                    cue_acc.append(float(np.mean(cue_bits == target)))
                    # reference: nearest stored pattern by matched-filter overlap
                    ov = mem.overlaps(cue_phase)
                    nearest_acc.append(float(np.mean(patterns[int(np.argmax(ov))] == target)))
                    rand_acc.append(float(np.mean(rng.integers(0, 2, size=n) == target)))
                ci = boot_ci(acc, np.random.default_rng([seed, 13, n, si, p]), resamples, level)
                wlo, whi = wilson(exact, profile["trials"])
                rows.append({
                    "n_modes": n, "sigma": sigma, "n_patterns": p, "p_over_n": ratio,
                    "trials": profile["trials"],
                    "recall_bit_accuracy": r6(np.mean(acc)),
                    "recall_ci_bootstrap": [r6(ci[0]), r6(ci[1])],
                    "exact_recall_fraction": r6(exact / profile["trials"]),
                    "exact_recall_ci_wilson": [r6(wlo), r6(whi)],
                    "cue_only_bit_accuracy": r6(np.mean(cue_acc)),
                    "nearest_stored_pattern_bit_accuracy": r6(np.mean(nearest_acc)),
                    "random_pattern_bit_accuracy": r6(np.mean(rand_acc)),
                })
    knees = []
    for n in profile["n_modes"]:
        for sigma in profile["sigmas"]:
            cell = sorted((r for r in rows if r["n_modes"] == n and r["sigma"] == sigma),
                          key=lambda r: r["p_over_n"])
            knee = next((r["p_over_n"] for r in cell if r["recall_bit_accuracy"] < knee_threshold), None)
            knees.append({"n_modes": n, "sigma": sigma, "knee_p_over_n": knee})
    stability = []
    grid = sorted(p_over_n)
    for sigma in profile["sigmas"]:
        ks = [k["knee_p_over_n"] for k in knees if k["sigma"] == sigma]
        if any(k is None for k in ks):
            stable = None
            note = "knee not reached at some N within the tested P/N range"
        else:
            span = grid.index(max(ks)) - grid.index(min(ks))
            stable = span <= 1
            note = f"grid-step span {span}"
        stability.append({"sigma": sigma, "knees": ks, "stable_within_one_grid_step": stable, "note": note})
    return {"rows": rows, "knees": knees, "knee_stability": stability,
            "knee_accuracy_threshold": knee_threshold}


# ------------------------------------------------------------------ scaling

def _read_meminfo_available():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def environment_info():
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    except Exception:
        blas = None
    return {
        "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
        "cpu_model": cpu, "cpu_count": os.cpu_count(), "blas_build": blas,
        "thread_env": {k: os.environ.get(k) for k in THREAD_VARS},
        "mem_available_bytes": _read_meminfo_available(),
        "loadavg_1_5_15_at_start": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }


def scaling_worker(args_json):
    """Runs in a subprocess so BLAS thread env vars take effect at import."""
    a = json.loads(args_json)
    rng = np.random.default_rng(a["seed"])
    out = {"environment": environment_info(), "measurements": []}
    t_start = time.time()
    for n in a["n_list"]:
        peak = a["peak_bytes_factor"] * 16 * n * n
        avail = _read_meminfo_available()
        if peak > a["max_peak_bytes"]:
            out["measurements"].append({"n_modes": n, "measured": False,
                                        "reason": f"peak {peak} B exceeds cap {a['max_peak_bytes']} B"})
            continue
        if avail is not None and peak > 0.5 * avail:
            out["measurements"].append({"n_modes": n, "measured": False,
                                        "reason": f"peak {peak} B exceeds half of available RAM {avail} B"})
            continue
        reps = a["repeats_large"] if n >= a["large_from_n"] else a["repeats"]
        pat = bits_to_wave(rng.integers(0, 2, size=n))
        mem = HolographicMemory(n)
        mem.store(pat)
        mem.finalize()
        for _ in range(a["warmup"]):
            mem.recall(pat, steps=a["steps"])
        recall = []
        for _ in range(reps):
            t0 = time.perf_counter()
            mem.recall(pat, steps=a["steps"])
            recall.append(time.perf_counter() - t0)
        store = []
        for _ in range(min(reps, 5)):
            m2 = HolographicMemory(n)
            t0 = time.perf_counter()
            m2.store(pat)
            store.append(time.perf_counter() - t0)
            del m2
        out["measurements"].append({"n_modes": n, "measured": True, "repeats": reps, "warmup": a["warmup"],
                                    "recall_seconds": recall, "store_seconds": store,
                                    "weights_bytes": 16 * n * n})
        del mem
    out["loadavg_1_5_15_at_end"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    out["worker_wall_seconds"] = time.time() - t_start
    return out


def run_scaling_subprocess(mode, plan, seed):
    env = dict(os.environ)
    if mode == "blas_threads_1":
        for k in THREAD_VARS:
            env[k] = "1"
    else:
        for k in THREAD_VARS:
            env.pop(k, None)
    payload = json.dumps({**plan, "seed": seed})
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--scaling-worker", payload],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"scaling worker failed ({mode}): {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def _fit(ns, ts):
    alpha, c = np.polyfit(np.log(ns), np.log(ts), 1)
    return float(alpha), float(c)


def analyze_scaling(measurements, sc, seed):
    meas = [m for m in measurements if m["measured"]]
    rng = np.random.default_rng([seed, 17])
    level, B = sc["ci_level"], sc["bootstrap_resamples"]
    lo_p, hi_p = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    ns = np.array([m["n_modes"] for m in meas], dtype=float)
    samples = [np.asarray(m["recall_seconds"]) for m in meas]
    medians = np.array([np.median(s) for s in samples])
    boots = np.array([[np.median(rng.choice(s, size=len(s), replace=True)) for s in samples]
                      for _ in range(B)])  # B x len(meas)
    per_n = []
    for i, m in enumerate(meas):
        s = samples[i]
        q25, q75 = np.percentile(s, [25, 75])
        n = m["n_modes"]
        per_n.append({
            "n_modes": n, "repeats": len(s), "median_seconds": float(medians[i]),
            "q25_seconds": float(q25), "q75_seconds": float(q75), "iqr_seconds": float(q75 - q25),
            "min_seconds": float(s.min()), "max_seconds": float(s.max()),
            "median_bootstrap_ci_seconds": [float(np.percentile(boots[:, i], lo_p)), float(np.percentile(boots[:, i], hi_p))],
            "store_median_seconds": float(np.median(m["store_seconds"])),
            "effective_gflops": float(sc["recall_steps"] * 8 * n * n / medians[i] / 1e9),
            "weights_bytes": m["weights_bytes"],
        })
    local = []
    for i in range(len(meas) - 1):
        sl = np.log(boots[:, i + 1] / boots[:, i]) / math.log(ns[i + 1] / ns[i])
        local.append({"n_from": int(ns[i]), "n_to": int(ns[i + 1]),
                      "slope": float(math.log(medians[i + 1] / medians[i]) / math.log(ns[i + 1] / ns[i])),
                      "slope_ci": [float(np.percentile(sl, lo_p)), float(np.percentile(sl, hi_p))]})
    result = {"per_n": per_n, "local_slopes": local,
              "unmeasured": [m for m in measurements if not m["measured"]]}
    mask = ns >= sc["fit_min_n"]
    if mask.sum() >= sc["fit_min_points"]:
        alpha, c = _fit(ns[mask], medians[mask])
        alphas, cs = [], []
        for b in boots:
            a_b, c_b = _fit(ns[mask], b[mask])
            alphas.append(a_b)
            cs.append(c_b)
        alphas, cs = np.array(alphas), np.array(cs)
        ci = [float(np.percentile(alphas, lo_p)), float(np.percentile(alphas, hi_p))]
        claims = sc["original_claims_checked"]
        fit = {
            "fit_range": f"N >= {sc['fit_min_n']} (fixed a priori)", "fit_n": [int(x) for x in ns[mask]],
            "alpha": alpha, "alpha_ci": ci, "theoretical_alpha": sc["theoretical_alpha"],
            "ci_contains_theoretical": bool(ci[0] <= sc["theoretical_alpha"] <= ci[1]),
            "ci_overlaps_original_claim_1.4_1.5": bool(ci[1] >= claims["alpha_range"][0] and ci[0] <= claims["alpha_range"][1]),
            "extrapolations": [],
        }
        for n_f in sc["extrapolate_to"]:
            pred = np.exp(cs) * n_f ** alphas
            fit["extrapolations"].append({
                "n_modes": n_f, "status": "EXTRAPOLATED, NOT MEASURED",
                "point_seconds": float(math.exp(c) * n_f ** alpha),
                "interval_seconds": [float(np.percentile(pred, lo_p)), float(np.percentile(pred, hi_p))],
                "interval_meaning": "bootstrap sampling noise of the fit only; excludes model error (cache and memory-bandwidth effects), see holdout_check",
                "weights_bytes": 16 * n_f * n_f,
            })
        result["fit"] = fit
    else:
        result["fit"] = {"status": f"not run: fewer than {sc['fit_min_points']} measured points with N >= {sc['fit_min_n']}"}
    # hold-out check of extrapolation quality (pre-declared ranges)
    ho = sc["holdout_check"]
    lo_n, hi_n = ho["fit_n_range"]
    hmask = (ns >= lo_n) & (ns <= hi_n)
    if hmask.sum() >= sc["fit_min_points"]:
        a_h, c_h = _fit(ns[hmask], medians[hmask])
        rows = []
        for n_p in ho["predict_n"]:
            if n_p in ns:
                j = int(np.where(ns == n_p)[0][0])
                pred = math.exp(c_h) * n_p ** a_h
                rows.append({"n_modes": n_p, "predicted_seconds": pred, "measured_median_seconds": float(medians[j]),
                             "predicted_over_measured": pred / float(medians[j])})
        result["holdout_check"] = {"fit_n_range": ho["fit_n_range"], "alpha": a_h, "rows": rows}
    else:
        result["holdout_check"] = {"status": "not run: too few points in range"}
    claims = sc["original_claims_checked"]
    claim_rows = []
    for key, n_c in (("recall25_seconds_at_4096_range", 4096),):
        p = [x for x in per_n if x["n_modes"] == n_c]
        if p:
            lo_c, hi_c = claims[key]
            claim_rows.append({"claim": f"recall 25 steps at N={n_c} in [{lo_c}, {hi_c}] s", "measured_median_seconds": p[0]["median_seconds"],
                               "measured_ci": p[0]["median_bootstrap_ci_seconds"],
                               "median_within_claim": bool(lo_c <= p[0]["median_seconds"] <= hi_c)})
    result["original_claims"] = claim_rows
    return result


def bench_scaling(profile_name, protocol, seed):
    sc = protocol["scaling"]
    prof = sc["profiles"][profile_name]
    plan = {"n_list": prof["n_list"], "repeats": prof["repeats"], "repeats_large": prof["repeats_large"],
            "large_from_n": prof["large_from_n"], "warmup": sc["warmup_runs"], "steps": sc["recall_steps"],
            "max_peak_bytes": sc["max_peak_bytes"], "peak_bytes_factor": sc["peak_bytes_factor"]}
    modes = {}
    for mode in sc["thread_modes"]:
        raw = run_scaling_subprocess(mode, plan, seed)
        modes[mode] = {"environment": {**raw["environment"], "loadavg_1_5_15_at_end": raw.get("loadavg_1_5_15_at_end"), "worker_wall_seconds": raw.get("worker_wall_seconds")}, "analysis": analyze_scaling(raw["measurements"], sc, seed)}
    return {"plan": plan, "modes": modes}


# --------------------------------------------------------------------- main

def run(profile_name, seed_override=None, allow_change=False):
    protocol, digest, locked = load_protocol(allow_change)
    seed = protocol["seed"] if seed_override is None else seed_override
    cap = protocol["capacity"]
    reverse = bench_reverse(protocol["reverse"]["profiles"][profile_name], seed, protocol["reverse"]["max_length"])
    capacity = bench_capacity(cap["profiles"][profile_name], cap["p_over_n"], seed, cap["recall_steps"],
                              cap["knee_accuracy_threshold"], cap["bootstrap_resamples"], cap["ci_level"])
    scaling = bench_scaling(profile_name, protocol, seed)
    deterministic = {
        "protocol_sha256": digest, "profile": profile_name, "seed": seed,
        "reverse": reverse, "capacity": capacity, "scaling_plan": scaling["plan"],
    }
    return {
        "protocol": protocol["protocol"], "protocol_sha256": digest, "protocol_locked": locked,
        "profile": profile_name, "seed": seed,
        "deterministic_sha256": canonical_digest(deterministic),
        "hash_scope": "protocol hash, profile, seed, reverse results, capacity results and the scaling plan; excludes every timing and everything derived from timings (alpha, CIs, slopes, extrapolations) and the environment",
        "reverse": reverse, "capacity": capacity, "scaling": scaling,
        "limitations": [
            "Synthetic classical software; no quantum-hardware claim.",
            "Timings are machine-, thread- and load-dependent. This box is shared with other jobs: see loadavg_* in each mode's environment; high load inflates and destabilizes timings, most visibly with default BLAS threads.",
            "Extrapolation intervals reflect sampling noise of the fit, not model error.",
            "Capacity uses random BPSK patterns, one target per memory, 25 recall steps and Gaussian cue phase noise.",
        ],
    }


def summarize(report):
    lines = [f"profile={report['profile']} protocol_locked={report['protocol_locked']} protocol_sha256={report['protocol_sha256']}"]
    rv = report["reverse"]
    lines.append(f"reverse: seeds={rv['seeds']} counts={rv['counts']} failures={len(rv['failures'])}")
    for k in report["capacity"]["knees"]:
        lines.append(f"knee N={k['n_modes']} sigma={k['sigma']}: P/N={k['knee_p_over_n']}")
    for mode, m in report["scaling"]["modes"].items():
        fit = m["analysis"]["fit"]
        lines.append(f"scaling[{mode}]: alpha={fit.get('alpha')} ci={fit.get('alpha_ci')}" if "alpha" in fit else f"scaling[{mode}]: {fit}")
    lines.append(f"deterministic_sha256: {report['deterministic_sha256']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rigorous cseq_wave benchmark")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="override the protocol seed (recorded)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-protocol-change", action="store_true")
    ap.add_argument("--scaling-worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.scaling_worker:
        print(json.dumps(scaling_worker(args.scaling_worker)))
        return 0
    if not (args.quick or args.full):
        ap.error("choose --quick or --full")
    report = run("quick" if args.quick else "full", args.seed, args.allow_protocol_change)
    print(summarize(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("wrote:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
