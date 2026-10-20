"""Evidence passport: seeded, hashed, reproducible run records.

Every experiment emits a JSON passport with the protocol version, RNG
seed, parameters, metrics, and a SHA-256 digest of the canonical metric
payload -- the same evidence pattern used by the OES telemetry bench.
A passport lets anyone re-run the experiment and check the digest.
"""

import hashlib
import json
from datetime import datetime, timezone

import numpy as np

PROTOCOL = "cseq-wave/0.1.0"


def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays and tuples to JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def canonical_digest(payload):
    """SHA-256 of the canonical (sorted, compact) JSON form of payload."""
    canonical = json.dumps(
        to_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_passport(seed, parameters, metrics):
    """Build an evidence passport dict for one experiment run.

    The digest covers protocol, seed, parameters and metrics only;
    created_utc and limitations are outside the hash, so re-running with the
    same seed reproduces the digest.
    """
    payload = to_jsonable({
        "protocol": PROTOCOL,
        "seed": int(seed),
        "parameters": parameters,
        "metrics": metrics,
    })
    digest = canonical_digest(payload)
    return {
        **payload,
        "sha256": digest,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_label": "SYNTHETIC",
        "limitations": [
            "classical simulation of interference-based computing",
            "not quantum hardware; not a cosmological validation",
            "not a validation of the Coherent Sequence theory",
            "all signals synthetic; results are seed-dependent",
            "OES-32 on the recall trajectory flags convergence, not correctness",
        ],
    }


def save_passport(passport, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(passport, f, indent=2, sort_keys=True)
