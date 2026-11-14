"""Training-only class weights with explicit legacy reproduction."""

import numpy as np

POLICIES = ("median_sqrt_v2", "legacy", "uniform")


def class_weights(counts, policy="median_sqrt_v2"):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Class counts must be a finite nonnegative vector")
    if policy not in POLICIES:
        raise ValueError(f"Unknown class-weight policy: {policy}")
    present = counts > 0
    weights = np.zeros_like(counts)
    if not present.any():
        return weights
    frequencies = counts / counts.sum()
    if policy == "uniform":
        weights[present] = 1
    elif policy == "legacy":
        weights[present] = 1 / np.sqrt(frequencies[present])
        weights[present] /= weights[present].mean()
        weights[present] = np.clip(weights[present], 0.25, 4)
    else:
        # The median cannot be inflated by a single extremely rare class.
        weights[present] = np.clip(
            np.sqrt(np.median(frequencies[present]) / frequencies[present]), 0.25, 4)
    return weights
