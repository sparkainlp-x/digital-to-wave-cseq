"""Shared fail-closed input validation (SYNTHETIC simulator boundaries).

Phase vectors
-------------
A phase vector must be a NONEMPTY, ONE-DIMENSIONAL array of finite real
numbers (signed/unsigned integer or floating dtype). Strings, bytes,
object arrays, complex values, booleans, NaN and +/-inf are rejected.
When ``n_modes`` is given the shape must be exactly ``(n_modes,)``;
validation never relies on NumPy broadcasting (a length-1 or scalar
input is NOT stretched to fit).

Binary vectors
--------------
Values are checked on the ORIGINAL input, before any integer cast, so a
fractional value such as 0.6 is rejected instead of being truncated to 0.
Accepted: integer, floating or boolean dtypes whose values are exactly 0
or 1 (1.0 is accepted; 0.5, NaN, inf are not). Strings, bytes, objects
and complex values are rejected.

Empty policy: an EMPTY binary vector is allowed by default, because the
text codec maps "" -> zero bits -> "" (text_to_bits / bits_to_text) and
that round trip is existing behaviour. Pass ``allow_empty=False`` where a
nonempty vector is required. A wave built from zero bits has zero modes
and is then rejected by every phase-vector boundary (memory, verifier).

Multidimensional policy: binary input must be one-dimensional. The codec
used to ``ravel()`` silently; v0.3.0 rejects ndim > 1 instead, so a 2-D
batch can no longer be mistaken for one long message. Callers that want
the old behaviour must flatten explicitly.
"""

import numbers

import numpy as np

_REAL_KINDS = ("i", "u", "f")


def _as_array(x, name):
    if isinstance(x, (str, bytes, bytearray)):
        raise TypeError(f"{name}: strings/bytes are not accepted")
    try:
        arr = np.asarray(x)
    except Exception as exc:  # ragged lists etc.
        raise TypeError(f"{name}: not array-like ({exc})") from None
    return arr


def validate_phase(x, n_modes=None, name="phase"):
    """Return a validated float64 copy of a phase vector or raise."""
    arr = _as_array(x, name)
    if arr.dtype.kind not in _REAL_KINDS:
        raise TypeError(f"{name}: dtype {arr.dtype} is not real numeric")
    if arr.ndim != 1:
        raise ValueError(f"{name}: must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name}: must be nonempty")
    out = arr.astype(np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name}: contains NaN or infinity")
    if n_modes is not None:
        n = validate_index(n_modes, name="n_modes")
        if out.shape != (n,):
            raise ValueError(f"{name}: expected shape ({n},), got {out.shape}")
    return out


def validate_bits(x, name="bits", allow_empty=True):
    """Return a validated int64 bit vector or raise (see module docstring)."""
    arr = _as_array(x, name)
    if arr.dtype.kind not in _REAL_KINDS + ("b",):
        raise TypeError(f"{name}: dtype {arr.dtype} is not accepted for bits")
    if arr.ndim != 1:
        raise ValueError(f"{name}: must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        if not allow_empty:
            raise ValueError(f"{name}: must be nonempty")
        return np.zeros(0, dtype=np.int64)
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: contains NaN or infinity")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("bits must be 0/1 exactly (checked before integer cast)")
    return arr.astype(np.int64)


def validate_index(x, upper=None, name="index"):
    """Nonnegative int (not bool, not float); optional exclusive upper bound."""
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (numbers.Integral, np.integer)):
        raise TypeError(f"{name}: must be an integer, got {type(x).__name__}")
    v = int(x)
    if v < 0:
        raise ValueError(f"{name}: must be nonnegative, got {v}")
    if upper is not None and v >= upper:
        raise ValueError(f"{name}: {v} out of range [0, {upper})")
    return v


def is_real_number(x):
    """True for finite real numbers that are not bool/str/complex."""
    if isinstance(x, (bool, np.bool_)):
        return False
    if not isinstance(x, (numbers.Real, np.integer, np.floating)):
        return False
    return bool(np.isfinite(float(x)))
