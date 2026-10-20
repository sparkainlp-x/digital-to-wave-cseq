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

import numpy as np

from .wave import alignment


class HolographicMemory:
    def __init__(self, n_modes):
        self.n_modes = n_modes
        self.weights = np.zeros((n_modes, n_modes), dtype=complex)
        self.n_patterns = 0
        self._patterns = []

    def store(self, wavestate):
        """Interfere one wave-state into the hologram."""
        x = wavestate.field
        if x.shape[0] != self.n_modes:
            raise ValueError("pattern mode count mismatch")
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

        trajectory[t] = C_seq alignment of state t with the cue's own
        start -- replaced by the caller with alignment-vs-target for the
        fidelity readout. When track_coherence is True we record the
        self-consistency C_seq(t) = alignment(s_t, s_{t-1}) instead, which
        rises as the field locks in.
        """
        s = np.exp(1j * np.asarray(cue.phase, dtype=float))
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

        score_mu = |<x^(mu), cue>|^2 / N^2 in [0, 1]. Constructive
        interference with the right pattern drives its score toward 1;
        the rest interfere destructively toward 0. This is the
        "wave-filtering" step: the hologram physically separates the
        high-probability (stored) response from alternatives.
        """
        q = np.exp(1j * np.asarray(cue_phase, dtype=float))
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
        """
        from .encode import slice_bpsk

        if slice_output:
            recalled_phase = slice_bpsk(recalled_phase)
        return alignment(recalled_phase, target_phase)
