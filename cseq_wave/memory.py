"""Holographic associative memory (complex-valued Hopfield network).

Storage is interference: P phase patterns are superposed into one
coupling matrix

    W = sum_mu x^(mu) x^(mu)+ ,   W_jj = 0

exactly as a hologram stores many images in one interference pattern.
Recall is interference dynamics: a (possibly decohered) cue evolves by

    s <- exp(i * arg(W s))

-- each step re-phases every mode toward the consensus of the field,
i.e. constructive interference with the stored pattern it most resembles
amplifies it while the rest cancel. The fixed point is the recalled
wave-state.

Readout: the recall fidelity is reported AS the coherence C_seq of the
relative phase between recalled and target patterns (see wave.alignment).
"""

import hashlib

import numpy as np

from .validation import validate_index, validate_phase
from .wave import alignment

DIGEST_SCHEME = "cseq-patterns-sha256/1"


class HolographicMemory:
    def __init__(self, n_modes):
        self.n_modes = validate_index(n_modes, name="n_modes")
        if self.n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        self.weights = np.zeros((n_modes, n_modes), dtype=complex)
        self.n_patterns = 0
        self._patterns = []

    def store(self, wavestate):
        """Interfere one wave-state into the hologram."""
        if np.asarray(wavestate.phase).shape != (self.n_modes,):
            raise ValueError("pattern mode count mismatch")
        validate_phase(wavestate.phase, self.n_modes, name="stored pattern phase")
        x = wavestate.field
        if not np.all(np.isfinite(x)):
            raise ValueError("stored pattern field contains NaN or infinity")
        # unit-magnitude phases carry the information
        p = np.exp(1j * np.angle(x))
        self.weights += np.outer(p, p.conj())
        self._patterns.append(np.angle(p))
        self.n_patterns += 1

    def finalize(self):
        """Remove self-coupling (standard Hopfield prescription)."""
        np.fill_diagonal(self.weights, 0.0)

    def recall(self, cue, steps=25, track_coherence=True):
        """Evolve a cue wave-state; return (recalled_phases, trajectory).

        When track_coherence is True, trajectory[t] is the step-to-step
        self-consistency alignment(s_t, s_{t-1}), which approaches 1 once the
        state stops changing. It measures convergence to *some* fixed point
        (a stored pattern, a spurious mixture, or the all-zero-phase state
        of an empty memory). It does not say whether the recalled pattern is
        the right one. When track_coherence is False the trajectory is empty.
        """
        s = np.exp(1j * validate_phase(cue.phase, self.n_modes, name="cue phase"))
        traj = []
        prev = s.copy()
        for _ in range(steps):
            s = np.exp(1j * np.angle(self.weights @ s))
            if track_coherence:
                traj.append(alignment(np.angle(s), np.angle(prev)))
            prev = s.copy()
        return np.angle(s), np.asarray(traj)

    def overlaps(self, cue_phase):
        """Matched-filter scores vs every stored pattern.

        score_mu = |<x^(mu), cue>|^2 / N^2 in [0, 1]. A cue equal to a
        stored pattern scores 1 for it and about 1/N for random others. A
        BPSK cue with a fraction p of flipped phases scores about (1 - 2p)^2
        (0.36 at p = 0.2) for its source pattern, so pass the *recalled*
        phases to see the post-recall separation.
        """
        q = np.exp(1j * validate_phase(cue_phase, self.n_modes, name="cue phase"))
        scores = []
        for p in self._patterns:
            x = np.exp(1j * p)
            scores.append(float(abs(np.vdot(x, q)) ** 2) / self.n_modes**2)
        return np.asarray(scores)

    def fidelity(self, recalled_phase, target_phase, slice_output=True):
        """Recall fidelity as C_seq of the relative phase (0..1).

        With slice_output=True (default) the recalled wave is first passed
        through the coherent-receiver slicer, exactly as a BPSK demodulator
        would quantize it before readout.

        Caveats: for BPSK targets the sliced readout equals
        |2 * bit_accuracy - 1|, so it is blind to a global phase flip (a fully
        inverted pattern also scores 1.0). Slicing may RAISE OR LOWER the
        score relative to slice_output=False (recalled [1.5, 1.7] vs target
        [0, 0]: unsliced ~0.995, sliced 0.0). Sliced fidelity, unsliced
        fidelity and bit accuracy are three distinct numbers; report bit
        accuracy alongside whichever fidelity you use.
        """
        from .encode import slice_bpsk

        recalled_phase = validate_phase(recalled_phase, self.n_modes, name="recalled phase")
        target_phase = validate_phase(target_phase, self.n_modes, name="target phase")

        if slice_output:
            recalled_phase = slice_bpsk(recalled_phase)
        return alignment(recalled_phase, target_phase)

    def digest(self):
        """Canonical SHA-256 of the stored patterns (shape AND order).

        Canonical form: ASCII header "cseq-patterns-sha256/1;n_modes=N;
        n_patterns=P;" followed by every stored phase vector, in storage
        order, as little-endian float64 bytes of shape (P, N). Storing the
        same patterns in another order, or adding one, changes the digest.
        """
        h = hashlib.sha256()
        h.update(f"{DIGEST_SCHEME};n_modes={self.n_modes};n_patterns={self.n_patterns};".encode("ascii"))
        for p in self._patterns:
            h.update(np.ascontiguousarray(p, dtype="<f8").tobytes())
        return h.hexdigest()
