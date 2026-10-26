"""Round-2 benchmark for cseq_wave (SYNTHETIC classical software).

Plan: benchmark_protocol_round2.json (hash in benchmark_protocol_round2.lock),
committed before this file existed and before any round-2 experiment was run.

  1. DENOISING: does recall reduce the error of a corrupted cue? Bit-flip cues
     (q = 0 control, 0.05 .. 0.30) and Gaussian phase noise large enough to
     flip bits, N = 64/128/256, P/N grid incl. the knee region, 50 trials per
     cell. Primary endpoint: per-trial gain of recall bit accuracy over the
     corrupted cue itself, paired bootstrap 95% CI, verdict per cell.
  2. LOAD-CONTROLLED SCALING: every timing batch records loadavg before and
     after plus the external CPU load (system busy cores minus this worker's
     own). Batches above the threshold are discarded and rerun; batches that
     never pass are flagged and excluded from fits.
  3. COMPLEX64 TIMING at N = 16384 (2 GiB matrix built in row chunks) when the
     memory rule allows. complex128 at 16384 is only ever an ESTIMATE.

Usage (repo root):

    python benchmark_round2.py --quick                           # CI smoke
    python benchmark_round2.py --full --json report_round2.json  # manual

No quantum-hardware claim. Timings, load and memory decisions are not part of
the deterministic hash.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import benchmark_rigorous as r1  # noqa: E402
from benchmark_rigorous import boot_ci, r6, wilson  # noqa: E402
from cseq_wave.encode import bits_to_wave, wave_to_bits  # noqa: E402
from cseq_wave.evidence import canonical_digest  # noqa: E402
from cseq_wave.memory import HolographicMemory  # noqa: E402
from cseq_wave.wave import WaveState  # noqa: E402

PROTOCOL_PATH = ROOT / "benchmark_protocol_round2.json"
LOCK_PATH = ROOT / "benchmark_protocol_round2.lock"
CHUNK_BYTES = 64 * 1024 * 1024


def load_protocol2(allow_change=False):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    digest = canonical_digest(protocol)
    locked = LOCK_PATH.read_text().strip() if LOCK_PATH.exists() else None
    ok = digest == locked
    if not ok and not allow_change:
        raise SystemExit(
            f"round-2 protocol hash {digest} does not match benchmark_protocol_round2.lock ({locked}). "
            "Use --allow-protocol-change to run anyway (recorded as protocol_locked=false)."
        )
    return protocol, digest, ok


# ---------------------------------------------------------------- denoising

def verdict(ci):
    if ci[0] > 0:
        return "denoises"
    if ci[1] < 0:
        return "degrades"
    return "inconclusive"


def _cue(kind, level, target, n, rng):
    """Return (cue_phase, cue_bits) for one trial."""
    if kind == "bit_flip":
        flips = (rng.random(n) < level) if level > 0 else np.zeros(n, dtype=bool)
        cue_bits = np.where(flips, 1 - target, target)
        return bits_to_wave(cue_bits).phase, cue_bits
    phase = bits_to_wave(target).phase + rng.normal(0.0, level, size=n)
    return phase, wave_to_bits(WaveState(np.ones(n), phase))


def bench_denoising(prof, seed, steps, resamples, level_ci):
    rows = []
    kinds = [("bit_flip", prof["q"]), ("phase_noise", prof["sigma"])]
    for n in prof["n_modes"]:
        for ki, (kind, levels) in enumerate(kinds):
            for li, lev in enumerate(levels):
                for ratio in prof["p_over_n"]:
                    p = int(round(n * ratio))
                    assert abs(p / n - ratio) < 1e-12, (n, ratio)
                    rng = np.random.default_rng([seed, 21, n, ki, li, p])
                    acc, cue_acc, near_acc, rand_acc, flipfrac = [], [], [], [], []
                    exact = retrieved = 0
                    for _ in range(prof["trials"]):
                        patterns = rng.integers(0, 2, size=(p, n))
                        mem = HolographicMemory(n)
                        for bits in patterns:
                            mem.store(bits_to_wave(bits))
                        mem.finalize()
                        target = patterns[0]
                        cue_phase, cue_bits = _cue(kind, lev, target, n, rng)
                        phases, _ = mem.recall(WaveState(np.ones(n), cue_phase), steps=steps)
                        got = wave_to_bits(WaveState(np.ones(n), phases))
                        acc.append(float(np.mean(got == target)))
                        exact += int(np.array_equal(got, target))
                        cue_acc.append(float(np.mean(cue_bits == target)))
                        flipfrac.append(float(np.mean(cue_bits != target)))
                        idx = int(np.argmax(mem.overlaps(cue_phase)))
                        near_acc.append(float(np.mean(patterns[idx] == target)))
                        retrieved += int(idx == 0)
                        rand_acc.append(float(np.mean(rng.integers(0, 2, size=n) == target)))
                    acc, cue_acc, near_acc = map(np.asarray, (acc, cue_acc, near_acc))
                    gain = acc - cue_acc
                    gain_n = acc - near_acc
                    brng = np.random.default_rng([seed, 23, n, ki, li, p])
                    gci = boot_ci(gain, brng, resamples, level_ci)
                    gnci = boot_ci(gain_n, brng, resamples, level_ci)
                    aci = boot_ci(acc, brng, resamples, level_ci)
                    ex_ci = wilson(exact, prof["trials"])
                    rt_ci = wilson(retrieved, prof["trials"])
                    control = (kind == "bit_flip" and lev == 0.0)
                    rows.append({
                        "n_modes": n, "cue_model": kind, "corruption": lev, "n_patterns": p, "p_over_n": ratio,
                        "trials": prof["trials"], "is_control": control,
                        "realised_cue_flip_fraction": r6(np.mean(flipfrac)),
                        "recall_bit_accuracy": r6(acc.mean()), "recall_ci_bootstrap": [r6(aci[0]), r6(aci[1])],
                        "cue_only_bit_accuracy": r6(cue_acc.mean()),
                        "nearest_stored_bit_accuracy": r6(near_acc.mean()),
                        "random_pattern_bit_accuracy": r6(np.mean(rand_acc)),
                        "gain_over_cue": r6(gain.mean()), "gain_over_cue_ci_paired_bootstrap": [r6(gci[0]), r6(gci[1])],
                        "verdict_vs_cue": "control" if control else verdict(gci),
                        "recall_minus_nearest": r6(gain_n.mean()),
                        "recall_minus_nearest_ci_paired_bootstrap": [r6(gnci[0]), r6(gnci[1])],
                        "verdict_vs_nearest": "control" if control else (
                            "recall_better" if gnci[0] > 0 else "recall_worse" if gnci[1] < 0 else "inconclusive"),
                        "exact_recall_fraction": r6(exact / prof["trials"]), "exact_recall_ci_wilson": [r6(ex_ci[0]), r6(ex_ci[1])],
                        "nearest_stored_retrieval_fraction": r6(retrieved / prof["trials"]),
                        "nearest_stored_retrieval_ci_wilson": [r6(rt_ci[0]), r6(rt_ci[1])],
                    })
    summary = []
    for kind, levels in kinds:
        for lev in levels:
            if kind == "bit_flip" and lev == 0.0:
                continue
            cells = [r for r in rows if r["cue_model"] == kind and r["corruption"] == lev]
            if cells:
                summary.append({"cue_model": kind, "corruption": lev, "cells": len(cells),
                                **{v: sum(c["verdict_vs_cue"] == v for c in cells) for v in ("denoises", "degrades", "inconclusive")}})
    control_cells = [r for r in rows if r["is_control"]]
    control_ok = all(r["cue_only_bit_accuracy"] == 1.0 for r in control_cells) if control_cells else None
    return {"rows": rows, "verdict_summary": summary, "control_cue_accuracy_is_exactly_1": control_ok,
            "ci": f"paired percentile bootstrap, {resamples} resamples, {level_ci:.0%}; no multiplicity correction"}


# ------------------------------------------------------------ complex64 impl

def build_c64(bits):
    """complex64 coupling matrix for the given (P, N) bit patterns, built in row chunks."""
    bits = np.asarray(bits)
    p, n = bits.shape
    pm = np.exp(1j * np.where(bits == 1, np.pi, 0.0)).astype(np.complex64)
    w = np.empty((n, n), dtype=np.complex64)
    rows = max(1, CHUNK_BYTES // (8 * n))
    pc = pm.conj()
    for i0 in range(0, n, rows):
        i1 = min(n, i0 + rows)
        w[i0:i1] = pm[:, i0:i1].T @ pc
    idx = np.arange(n)
    w[idx, idx] = 0
    return w


def recall_c64(w, cue_phase, steps):
    s = np.exp(1j * np.asarray(cue_phase)).astype(np.complex64)
    for _ in range(steps):
        s = np.exp(1j * np.angle(w @ s)).astype(np.complex64, copy=False)
    return np.angle(s)


def agreement_check(spec, seed, steps):
    out = []
    for n in spec["n_modes"]:
        for p in spec["n_patterns"]:
            same = 0
            agree = []
            for k in range(spec["seeds"]):
                rng = np.random.default_rng([seed, 31, n, p, k])
                patterns = rng.integers(0, 2, size=(p, n))
                mem = HolographicMemory(n)
                for b in patterns:
                    mem.store(bits_to_wave(b))
                mem.finalize()
                flips = rng.random(n) < spec["q"]
                cue_phase = bits_to_wave(np.where(flips, 1 - patterns[0], patterns[0])).phase
                b128 = wave_to_bits(WaveState(np.ones(n), mem.recall(WaveState(np.ones(n), cue_phase), steps=steps)[0]))
                b64 = wave_to_bits(WaveState(np.ones(n), recall_c64(build_c64(patterns), cue_phase, steps)))
                same += int(np.array_equal(b128, b64))
                agree.append(float(np.mean(b128 == b64)))
            out.append({"n_modes": n, "n_patterns": p, "seeds": spec["seeds"],
                        "identical_bit_vectors": same, "mean_bit_agreement": r6(np.mean(agree))})
    return out


# --------------------------------------------------------------- load control

def _busy_seconds():
    try:
        vals = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
    except (OSError, ValueError):
        return None
    idle = vals[3] + vals[4]
    return (sum(vals) - idle) / os.sysconf("SC_CLK_TCK")


def _own_cpu():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def controlled_batch(fn, reps, lc):
    """Run fn() reps times as one batch; gate on external busy cores; retry."""
    attempts = []
    accepted = None
    for k in range(lc["max_attempts"]):
        la0 = os.getloadavg()[0]
        b0, o0, t0 = _busy_seconds(), _own_cpu(), time.perf_counter()
        samples = []
        for _ in range(reps):
            s0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - s0)
        wall = time.perf_counter() - t0
        b1, o1 = _busy_seconds(), _own_cpu()
        ext = None if b0 is None or b1 is None else max(0.0, ((b1 - b0) - (o1 - o0)) / wall)
        ok = ext is not None and ext <= lc["threshold_external_busy_cores"]
        attempts.append({"recall_seconds": samples, "external_busy_cores": None if ext is None else round(ext, 3),
                         "loadavg1_before": round(la0, 2), "loadavg1_after": round(os.getloadavg()[0], 2),
                         "wall_seconds": round(wall, 3), "accepted": bool(ok)})
        if ok:
            accepted = k
            break
        if k + 1 < lc["max_attempts"]:
            time.sleep(lc["retry_wait_seconds"])
    return {"attempts": attempts, "accepted_attempt": accepted, "load_controlled": accepted is not None,
            "recall_seconds": attempts[accepted if accepted is not None else -1]["recall_seconds"]}


def worker(args_json):
    a = json.loads(args_json)
    rng = np.random.default_rng(a["seed"])
    out = {"environment": r1.environment_info(), "complex128": [], "complex64": []}
    t_start = time.time()
    steps = a["steps"]
    for n in a["c128"]["n_list"]:
        peak = a["peak_bytes_factor"] * 16 * n * n
        avail = r1._read_meminfo_available()
        if peak > a["max_peak_bytes"] or (avail is not None and peak > 0.5 * avail):
            out["complex128"].append({"n_modes": n, "measured": False, "reason": f"peak {peak} B over cap or over half of available RAM {avail} B"})
            continue
        reps = a["c128"]["repeats_large"] if n >= a["c128"]["large_from_n"] else a["c128"]["repeats"]
        pat = bits_to_wave(rng.integers(0, 2, size=n))
        mem = HolographicMemory(n)
        mem.store(pat)
        mem.finalize()
        for _ in range(a["c128"]["warmup_runs"]):
            mem.recall(pat, steps=steps)
        res = controlled_batch(lambda: mem.recall(pat, steps=steps), reps, a["load_control"])
        store = []
        for _ in range(3):
            m2 = HolographicMemory(n)
            t0 = time.perf_counter()
            m2.store(pat)
            store.append(time.perf_counter() - t0)
            del m2
        out["complex128"].append({"n_modes": n, "measured": True, "repeats": reps, "weights_bytes": 16 * n * n,
                                  "store_seconds": store, **res})
        del mem
    for n in a["c64"]["n_list"]:
        peak = 8 * n * n + CHUNK_BYTES
        avail = r1._read_meminfo_available()
        if peak > a["max_peak_bytes"]:
            out["complex64"].append({"n_modes": n, "measured": False, "reason": f"peak {peak} B exceeds cap {a['max_peak_bytes']} B"})
            continue
        if avail is not None and peak > 0.5 * avail:
            out["complex64"].append({"n_modes": n, "measured": False,
                                     "reason": f"peak {peak} B exceeds half of MemAvailable {avail} B at the time of the check"})
            continue
        bits = rng.integers(0, 2, size=(8, n))
        w = build_c64(bits)
        cue = bits_to_wave(bits[0]).phase
        for _ in range(a["c64"]["warmup_runs"]):
            recall_c64(w, cue, steps)
        res = controlled_batch(lambda: recall_c64(w, cue, steps), a["c64"]["repeats"], a["load_control"])
        out["complex64"].append({"n_modes": n, "measured": True, "repeats": a["c64"]["repeats"], "weights_bytes": 8 * n * n,
                                 "peak_bytes_formula": peak, "mem_available_before_bytes": avail, **res})
        del w
    out["worker_wall_seconds"] = time.time() - t_start
    out["loadavg_1_5_15_at_end"] = list(os.getloadavg())
    return out


def run_worker(mode, plan, seed):
    env = dict(os.environ)
    for k in r1.THREAD_VARS:
        if mode == "blas_threads_1":
            env[k] = "1"
        else:
            env.pop(k, None)
    payload = json.dumps({**plan, "seed": seed})
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", payload],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"round-2 worker failed ({mode}): {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def summarize_c64(entries, steps, resamples, level, seed):
    rng = np.random.default_rng([seed, 41])
    lo_p, hi_p = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    per_n, excluded, unmeasured = [], [], []
    for m in entries:
        if not m["measured"]:
            unmeasured.append({"n_modes": m["n_modes"], "reason": m["reason"]})
            continue
        if not m["load_controlled"]:
            excluded.append({"n_modes": m["n_modes"], "reason": "no attempt met the load threshold"})
            continue
        s = np.asarray(m["recall_seconds"])
        meds = np.median(rng.choice(s, size=(resamples, len(s))), axis=1)
        q25, q75 = np.percentile(s, [25, 75])
        n = m["n_modes"]
        per_n.append({"n_modes": n, "repeats": len(s), "median_seconds": float(np.median(s)), "iqr_seconds": float(q75 - q25),
                      "min_seconds": float(s.min()), "max_seconds": float(s.max()),
                      "median_bootstrap_ci_seconds": [float(np.percentile(meds, lo_p)), float(np.percentile(meds, hi_p))],
                      "effective_gflops": float(steps * 8 * n * n / np.median(s) / 1e9), "weights_bytes": m["weights_bytes"],
                      "attempts_used": len(m["attempts"])})
    slopes = [{"n_from": a["n_modes"], "n_to": b["n_modes"],
               "slope": float(math.log(b["median_seconds"] / a["median_seconds"]) / math.log(b["n_modes"] / a["n_modes"]))}
              for a, b in zip(per_n, per_n[1:])]
    return {"per_n": per_n, "local_slopes_descriptive": slopes, "unmeasured": unmeasured, "excluded_uncontrolled": excluded}


def _med_at(entries, n):
    for m in entries:
        if m["n_modes"] == n and m.get("measured") and m.get("load_controlled"):
            return float(np.median(m["recall_seconds"]))
    return None


def scaling_round2(profile_name, protocol, seed):
    s2 = protocol["scaling_round2"]
    r1proto, _, _ = r1.load_protocol(allow_change=True)
    base = r1proto["scaling"]
    prof = s2["profiles"][profile_name]
    full = profile_name == "full"
    c128, c64 = s2["complex128"], s2["complex64"]
    plan = {
        "steps": s2["recall_steps"], "max_peak_bytes": base["max_peak_bytes"], "peak_bytes_factor": base["peak_bytes_factor"],
        "load_control": {k: s2["load_control"][k] for k in ("threshold_external_busy_cores", "max_attempts", "retry_wait_seconds")},
        "c128": {"n_list": c128["n_list"] if full else prof["complex128_n_list"],
                 "repeats": c128["repeats"] if full else prof["repeats"],
                 "repeats_large": c128["repeats_large"] if full else prof["repeats_large"],
                 "large_from_n": c128["large_from_n"] if full else prof["large_from_n"],
                 "warmup_runs": c128["warmup_runs"]},
        "c64": {"n_list": c64["n_list"] if full else prof["complex64_n_list"],
                "repeats": c64["repeats"] if full else prof["complex64_repeats"], "warmup_runs": c64["warmup_runs"]},
    }
    sc = {**base, "fit_min_n": c128["fit_min_n"], "fit_min_points": c128["fit_min_points"],
          "bootstrap_resamples": c128["bootstrap_resamples"], "ci_level": c128["ci_level"], "recall_steps": s2["recall_steps"]}
    modes = {}
    for mode in s2["thread_modes"]:
        raw = run_worker(mode, plan, seed)
        controlled = [{"n_modes": m["n_modes"], "measured": True, "recall_seconds": m["recall_seconds"],
                       "store_seconds": m["store_seconds"], "weights_bytes": m["weights_bytes"]}
                      for m in raw["complex128"] if m["measured"] and m["load_controlled"]]
        c128_analysis = r1.analyze_scaling(controlled, sc, seed)
        c128_batches = [{"n_modes": m["n_modes"], "measured": m["measured"],
                         **({"load_controlled": m["load_controlled"], "attempts": m["attempts"], "accepted_attempt": m["accepted_attempt"]} if m["measured"] else {"reason": m["reason"]})}
                        for m in raw["complex128"]]
        c64_analysis = summarize_c64(raw["complex64"], s2["recall_steps"], c128["bootstrap_resamples"], c128["ci_level"], seed)
        c64_batches = [{"n_modes": m["n_modes"], "measured": m["measured"],
                        **({"load_controlled": m["load_controlled"], "attempts": m["attempts"], "accepted_attempt": m["accepted_attempt"],
                            "mem_available_before_bytes": m["mem_available_before_bytes"]} if m["measured"] else {"reason": m["reason"]})}
                       for m in raw["complex64"]]
        est = {"status_label": s2["complex128_estimate_at_16384"]["status_label"], "rows": []}
        t64_big = _med_at(raw["complex64"], 16384)
        for ref in (8192, 4096):
            a, b = _med_at(raw["complex128"], ref), _med_at(raw["complex64"], ref)
            if t64_big is not None and a is not None and b is not None:
                est["rows"].append({"ratio_reference_n": ref, "complex128_over_complex64_at_reference": a / b,
                                    "estimated_complex128_seconds_at_16384": t64_big * a / b,
                                    "primary": ref == 8192})
        if not est["rows"]:
            est["note"] = "not computed: needs load-controlled complex64 at 16384 and both dtypes at the reference N"
        modes[mode] = {"environment": {**raw["environment"], "loadavg_1_5_15_at_end": raw["loadavg_1_5_15_at_end"],
                                       "worker_wall_seconds": raw["worker_wall_seconds"]},
                       "complex128": {"analysis": c128_analysis, "batches": c128_batches},
                       "complex64": {"analysis": c64_analysis, "batches": c64_batches},
                       "complex128_estimate_at_16384": est}
    return {"plan": plan, "modes": modes}


# --------------------------------------------------------------------- main

def run2(profile_name, seed_override=None, allow_change=False):
    protocol, digest, locked = load_protocol2(allow_change)
    seed = protocol["seed"] if seed_override is None else seed_override
    dn = protocol["denoising"]
    denoising = bench_denoising(dn["profiles"][profile_name], seed, dn["recall_steps"], dn["bootstrap_resamples"], dn["ci_level"])
    agreement = agreement_check(protocol["scaling_round2"]["complex64"]["agreement_check"], seed, protocol["scaling_round2"]["recall_steps"])
    scaling = scaling_round2(profile_name, protocol, seed)
    deterministic = {"protocol_sha256": digest, "profile": profile_name, "seed": seed,
                     "denoising": denoising, "complex64_agreement": agreement, "scaling_plan": scaling["plan"]}
    return {
        "protocol": protocol["protocol"], "protocol_sha256": digest, "protocol_locked": locked,
        "profile": profile_name, "seed": seed,
        "deterministic_sha256": canonical_digest(deterministic),
        "hash_scope": protocol["determinism"]["hash_scope"],
        "denoising": denoising, "complex64_agreement": agreement, "scaling": scaling,
        "limitations": [
            "Synthetic classical software; no quantum-hardware claim.",
            "Denoising verdicts are per cell with 95% paired-bootstrap intervals and no multiplicity correction.",
            "complex64 timing is a different dtype from the original complex128 claim; complex128 at N=16384 is an estimate only.",
            "Timings are machine-, thread- and load-dependent; see external_busy_cores per attempt.",
        ],
    }


def summarize2(report):
    lines = [f"profile={report['profile']} protocol_locked={report['protocol_locked']} protocol_sha256={report['protocol_sha256']}"]
    d = report["denoising"]
    lines.append(f"control cue accuracy exactly 1.0: {d['control_cue_accuracy_is_exactly_1']}")
    for s in d["verdict_summary"]:
        lines.append(f"denoise {s['cue_model']} {s['corruption']}: cells={s['cells']} denoises={s['denoises']} degrades={s['degrades']} inconclusive={s['inconclusive']}")
    for a in report["complex64_agreement"]:
        lines.append(f"complex64 agreement N={a['n_modes']} P={a['n_patterns']}: identical {a['identical_bit_vectors']}/{a['seeds']}")
    for mode, m in report["scaling"]["modes"].items():
        fit = m["complex128"]["analysis"]["fit"]
        lines.append(f"scaling[{mode}] complex128: alpha={fit.get('alpha')} ci={fit.get('alpha_ci')}" if "alpha" in fit else f"scaling[{mode}] complex128: {fit}")
        lines.append(f"scaling[{mode}] complex64 measured N={[p['n_modes'] for p in m['complex64']['analysis']['per_n']]}")
    lines.append(f"deterministic_sha256: {report['deterministic_sha256']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Round-2 cseq_wave benchmark")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-protocol-change", action="store_true")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        print(json.dumps(worker(args.worker)))
        return 0
    if not (args.quick or args.full):
        ap.error("choose --quick or --full")
    report = run2("quick" if args.quick else "full", args.seed, args.allow_protocol_change)
    print(summarize2(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("wrote:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
