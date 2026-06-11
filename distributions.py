"""
Distribution helpers.

We sample every (low, base, high) parameter from a Beta-PERT distribution.
Reasoning: with N≈10–15 events in the sheet, we don't have enough data
to estimate a full distribution shape. PERT requires only three numbers
the user can defend — minimum plausible, most-likely, maximum plausible —
and gives a sensible bell-like shape between them. It is the standard
choice in risk analysis and project estimation when expert ranges are
all you have.

PERT is a special case of the Beta distribution on [low, high] with
shape parameters derived from the mode:
    alpha = 1 + 4 * (mode - low) / (high - low)
    beta  = 1 + 4 * (high - mode) / (high - low)

For rates and percentages the same call works because PERT respects the
[low, high] bounds and we constrain those bounds to [0, 1].

`sample_pert` returns an ndarray of shape (n,) for one (low, base, high).
`sample_pert_dict` maps a {key: (l,b,h)} into {key: ndarray(n,)}.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

LBH = Tuple[float, float, float]


def sample_pert(low: float, base: float, high: float, n: int,
                rng: np.random.Generator) -> np.ndarray:
    """Beta-PERT samples on [low, high] with mode at base."""
    if not (low <= base <= high):
        raise ValueError(f"PERT requires low <= base <= high, got {low}/{base}/{high}")
    if high == low:
        return np.full(n, base, dtype=float)
    span = high - low
    alpha = 1.0 + 4.0 * (base - low) / span
    beta_ = 1.0 + 4.0 * (high - base) / span
    raw = rng.beta(alpha, beta_, size=n)
    return low + raw * span


def sample_pert_dict(
    triplets: Dict[str, LBH], n: int, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    return {k: sample_pert(*v, n=n, rng=rng) for k, v in triplets.items()}


def sample_bernoulli(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bernoulli draws given a vector of probabilities (one per run)."""
    return (rng.random(size=p.shape) < p).astype(np.int8)
