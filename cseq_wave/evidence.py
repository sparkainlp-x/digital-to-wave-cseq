"""Evidence passport: seeded, hashed, reproducible run records.

Every experiment emits a JSON passport with the protocol version, RNG
seed, parameters, metrics, and a SHA-256 digest of the canonical metric
payload -- the same evidence pattern used by the OES telemetry bench.
A passport lets anyone re-run the experiment and check the digest.
"""

import hashlib
import json
from datetime import datetime, timezone

PROTOCOL = "cseq-wave/0.1.0"


def evidence_passport(seed, parameters, metrics):
    """Build an evidence passport dict for one experiment run."""
    payload = {
        "protocol": PROTOCOL,
        "seed": int(seed),
        "parameters": parameters,
        "metrics": metrics,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        **payload,
        "sha256": digest,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "limitations": [
            "classical simulation of interference-based computing",
            "not quantum hardware; not a cosmological validation",
            "all signals synthetic; results are seed-dependent",
        ],
    }


def save_passport(passport, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(passport, f, indent=2, sort_keys=True)
