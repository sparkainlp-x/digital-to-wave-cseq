# Cue gate and verdicts (SYNTHETIC)

N = 256 modes, 0.3 rad jitter, 40 recall steps. tau = q0.99 of 256 never-stored cues per hologram.

## Separation (4 patterns, 20% flips, 240 trials of each kind)

| signal | stored cue | never-stored cue |
|---|---:|---:|
| cue max overlap, mean | 0.3238 | 0.0087 |
| cue max overlap, worst case | min 0.1699 | q0.99 0.0294, max 0.0342 |
| post-recall winner margin, worst case | min 0.3953 | max 0.9727 |

The cue-side overlap separates the two groups; the post-recall margin does not.

## Verdicts on stored cues by decoherence (20 holograms x 25 cues per row, ground-truth mode)

| patterns | flips | verified | unverified | confabulation suspected | ghost false accept | tau |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.2 | 1.000 | 0.000 | 0.000 | 0.0135 | 0.0320 |
| 4 | 0.3 | 1.000 | 0.000 | 0.000 | - | 0.0320 |
| 4 | 0.35 | 0.948 | 0.004 | 0.048 | - | 0.0320 |
| 4 | 0.4 | 0.564 | 0.004 | 0.432 | - | 0.0320 |
| 4 | 0.45 | 0.094 | 0.002 | 0.904 | - | 0.0320 |
| 20 | 0.2 | 0.020 | 0.980 | 0.000 | 0.0125 | 0.0424 |
| 20 | 0.3 | 0.010 | 0.990 | 0.000 | - | 0.0424 |
| 20 | 0.35 | 0.004 | 0.902 | 0.094 | - | 0.0424 |
| 20 | 0.4 | 0.002 | 0.412 | 0.586 | - | 0.0424 |
| 20 | 0.45 | 0.000 | 0.054 | 0.946 | - | 0.0424 |

Ghost false accepts were measured at 20% flips only (4 ghosts per stored cue). At 40% flips and above, most genuinely stored cues are rejected as suspected confabulation: the gate trades misses for safety. With 20 patterns the cue gate passes, but recall itself fails under 0.3 rad jitter (continuous phases leave the BPSK attractors), so the verdict is UNVERIFIED_RECALL. That is the verdict doing its job, not a gate failure.

## Known blind spot: mixture cue

Cue = first half of pattern 0 + second half of pattern 1. Cue overlaps [0.2991, 0.274, 0.001, 0.0002] (tau 0.0307), so the cue gate passes. Recalled overlaps [0.2659, 0.3077, 0.2197, 0.0002]: the dynamics land on a spurious mixture. Verdict: UNVERIFIED_RECALL.

sha256: `da895d888c3b35398ffe91841e364d2ed85c25093f41e307cbde4a0e334693e1`
