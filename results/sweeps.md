# Sweeps (SYNTHETIC)

N = 256 modes, 20% phase flips, 25 recall steps, 100 seeded trials per row.

## Capacity

Each row uses its own seeds, so with 100 trials small reversals between rows (for example 10 vs 15 patterns) are sampling noise, not a real effect.

| patterns | load P/N | bit accuracy | exact recall | fidelity (sliced) | fidelity (unsliced) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.004 | 1.0000 | 1.00 | 1.0000 | 1.0000 |
| 5 | 0.019 | 1.0000 | 1.00 | 1.0000 | 1.0000 |
| 10 | 0.039 | 0.9998 | 0.99 | 0.9997 | 0.9933 |
| 15 | 0.059 | 1.0000 | 1.00 | 1.0000 | 1.0000 |
| 20 | 0.078 | 0.9816 | 0.77 | 0.9632 | 0.9140 |
| 25 | 0.098 | 0.9922 | 0.83 | 0.9844 | 0.9805 |
| 30 | 0.117 | 0.8793 | 0.29 | 0.7585 | 0.6974 |
| 35 | 0.137 | 0.9266 | 0.31 | 0.8533 | 0.8123 |
| 40 | 0.156 | 0.7867 | 0.06 | 0.5734 | 0.5041 |
| 50 | 0.195 | 0.7065 | 0.00 | 0.4130 | 0.3552 |
| 60 | 0.234 | 0.6570 | 0.00 | 0.3141 | 0.2645 |

## OES-32 on the recall trajectory (200 trials, 5 patterns)

| cue | mean score | min score | fire rate |
|---|---:|---:|---:|
| stored | 0.9924 | 0.9904 | 1.00 |
| unstored | 0.9814 | 0.9722 | 1.00 |
| noise | 0.9835 | 0.9800 | 1.00 |
| empty | 0.9854 | 0.9850 | 1.00 |

## Never-stored cues

- mean max overlap of the recalled state with a stored pattern: 0.795
- share converging onto a stored pattern (max overlap > 0.9): 0.63
- mean fidelity against the cue's own bits: 0.098

sha256: `28d07d86a609c1d0dea047c934d39cedf471ca662a164d014b488f0a6b2b9469`
