"""Wave-state primitives.

A wave-state is a discrete complex field

    psi_j = A_j * exp(i * theta_j),   j = 0..N-1

over N modes. The central observable is the amplitude-weighted phase
coherence

    C_seq = |sum_j A_j * exp(i*theta_j)| / sum_j A_j,   0 <= C_seq <= 1

the computational counterpart of C_seq = |<exp(i*theta)>| in Lee (2026).
C_seq = 1 means perfect phase alignment (fully coherent); C_seq ~ 0
means random phases (fully decoherent). Amplitude weighting keeps dark
(low-amplitude) modes from voting on the phase.
"""

import numpy as np


class WaveState:
    """A discrete wave-state psi = A * exp(i*theta)."""

    def __init__(self, amplitude, phase):
        amplitude = np.asarray(amplitude, dtype=float)
        phase = np.asarray(phase, dtype=float)
        if amplitude.shape != phase.shape:
            raise ValueError("amplitude and phase must have the same shape")
        if np.any(amplitude < 0):
            raise ValueError("amplitude must be non-negative")
        self.amplitude = amplitude
        self.phase = phase

    @property
    def field(self):
        """Complex field psi_j = A_j * exp(i*theta_j)."""
        return self.amplitude * np.exp(1j * self.phase)

    @property
    def n_modes(self):
        return self.amplitude.shape[0]

    def coherence(self):
        """Phase coherence C_seq of this wave-state. See module docstring."""
        return coherence(self.amplitude, self.phase)

    def interfere(self, other, weight=1.0):
        """Superpose another wave-state (linear interference).

        Returns a new WaveState with summed complex fields; amplitude and
        phase are re-extracted from the interfered field. This is the
        primitive by which wave-states 'compute': aligned phases add
        (constructive interference), opposed phases cancel (destructive).
        """
        if self.n_modes != other.n_modes:
            raise ValueError("mode counts must match to interfere")
        total = self.field + weight * other.field
        return WaveState(np.abs(total), np.angle(total))

    def phase_shift(self, delta):
        """Apply a uniform or per-mode phase shift."""
        return WaveState(self.amplitude.copy(), self.phase + np.asarray(delta))

    def decohere(self, rng, flip_prob=0.0, jitter=0.0):
        """Simulate decoherence: random pi-flips and Gaussian phase jitter."""
        phase = self.phase.copy()
        if flip_prob > 0:
            flips = rng.random(phase.shape) < flip_prob
            phase[flips] += np.pi
        if jitter > 0:
            phase += rng.normal(0.0, jitter, phase.shape)
        return WaveState(self.amplitude.copy(), phase)


def coherence(amplitude, phase):
    """Amplitude-weighted phase coherence C_seq in [0, 1]."""
    amplitude = np.asarray(amplitude, dtype=float)
    phase = np.asarray(phase, dtype=float)
    denom = np.sum(amplitude)
    if denom == 0:
        return 0.0
    return float(abs(np.sum(amplitude * np.exp(1j * phase))) / denom)


def alignment(a_phase, b_phase, amplitude=None):
    """Phase alignment between two phase patterns: |<exp(i(a-b))>|.

    This is C_seq of the *relative* phase -- 1.0 when the patterns are
    identical, ~0 when unrelated. Used as the recall-fidelity readout.
    """
    a_phase = np.asarray(a_phase, dtype=float)
    b_phase = np.asarray(b_phase, dtype=float)
    if amplitude is None:
        amplitude = np.ones_like(a_phase)
    return coherence(amplitude, a_phase - b_phase)
