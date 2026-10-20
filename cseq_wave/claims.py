"""Claims screen and evidence tags (same idea as spark-iis-api).

A simple regular-expression screen for the wave computer's own report
text. It is a guard against accidental overclaiming in generated reports,
not a guarantee: a pattern list cannot catch every paraphrase.

- Every reported number carries an evidence tag:
  SYNTHETIC (computed from seeded synthetic signals), TARGET (a
  configured threshold), REPORTED (measured on real data -- none here),
  UNRUN (not computed).
- Outgoing report text is screened for forbidden claims. A report that
  calls this a quantum computer, a validation of the theory, or a
  breakthrough proof is refused, not edited.
- The boundary statement below is attached to every verified report.
"""

import re

TAGS = ("SYNTHETIC", "TARGET", "REPORTED", "UNRUN")

BOUNDARY_STATEMENT = (
    "Classical simulation of interference-based computing (synthetic "
    "signals only). Not quantum hardware; not a validation of the "
    "Coherent Sequence theory; no physical or cosmological claim."
)

# Claims this software must never make about itself. Matched case-
# insensitively against outgoing report text.
FORBIDDEN_PATTERNS = [
    r"quantum (computer|hardware|advantage|supremacy)",
    r"validat(es|ed|ion) (of )?the (coherent sequence )?theory",
    r"proves? (the )?theory",
    r"breakthrough (is )?(proven|confirmed|demonstrated)",
    r"faster than (a )?(classical|supercomputer)",
    r"consciousness",
    r"therapeutic|medical (device|benefit|claim)",
    r"dark (matter|energy) (detected|explained|proven)",
    r"\bproven\b",
    r"\bvalidated\b",
    r"production[- ]ready",
    r"real[- ]world performance",
]


class ClaimViolation(Exception):
    """Raised when outgoing text contains a forbidden claim."""


_NEGATION = re.compile(r"\b(not|no|never|non|isn't|aren't|n't)\b", re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r"[.;:!?,()\n]")


def _is_disclaimer(text, start):
    """True when the match is negated inside its own clause.

    The negation must appear among the (at most) three words right before
    the match, with no clause break in between. "Not quantum hardware"
    and "not a validation of the theory" are disclaimers; "not slow: a
    quantum computer" is not, because the colon ends the negated clause.
    """
    window = text[max(0, start - 40):start]
    breaks = list(_CLAUSE_BREAK.finditer(window))
    if breaks:
        window = window[breaks[-1].end():]
    words = window.split()[-3:]
    return any(_NEGATION.fullmatch(w.strip("\"'")) for w in words)


def screen_text(text):
    """Return the list of forbidden patterns CLAIMED in text ([] = clean).

    Negated occurrences (the boundary statement's "not ...") are
    disclaimers, not claims, and are not returned.
    """
    hits = []
    for pat in FORBIDDEN_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            if not _is_disclaimer(text, m.start()):
                hits.append(pat)
                break
    return hits


def assert_clean(text):
    """Raise ClaimViolation if text contains a forbidden claim."""
    hits = screen_text(text)
    if hits:
        raise ClaimViolation(f"forbidden claim(s) in report text: {hits}")
    return text


def tag(value, evidence_tag):
    """Attach an evidence tag to a reported value."""
    if evidence_tag not in TAGS:
        raise ValueError(f"unknown evidence tag: {evidence_tag}")
    return {"value": value, "tag": evidence_tag}
