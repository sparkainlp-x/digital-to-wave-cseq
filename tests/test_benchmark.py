"""benchmark.py --quick: runs, key results hold, and the report hash is deterministic."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, name):
    out = tmp_path / name
    proc = subprocess.run(
        [sys.executable, "-W", "error", str(ROOT / "benchmark.py"), "--quick", "--json", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def test_quick_benchmark_is_deterministic_and_correct(tmp_path):
    a = _run(tmp_path, "a.json")
    b = _run(tmp_path, "b.json")
    assert a["sha256"] == b["sha256"]
    rows = a["results"]
    assert all(r["bit_accuracy"] == 1.0 for r in rows if r["test"] == "codec_reverse" and r["sigma"] == 0.0)
    assert all(r["exact"] and r["max_phase_error_rad"] == 0.0 for r in rows if r["test"] == "wave_reverse")
    text = [r for r in rows if r["test"] == "full_loop_text"][0]
    assert text["exact"] and text["text_out"] == "Hello wave"
    scaling = [r for r in rows if r["test"] == "scaling"]
    assert [r["weights_bytes"] for r in scaling] == [n * n * 16 for n in (16, 32, 64, 128)]
