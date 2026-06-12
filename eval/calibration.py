from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from sklearn.isotonic import IsotonicRegression

N_BINS: int = 10
MIN_BIN_SIZE: int = 5


def find_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Find temperature T > 0 that minimises NLL on calibration data.

    logits: shape (N, K) raw model outputs (pre-softmax)
    labels: shape (N,) integer class indices
    Returns scalar T.
    Optimises with scipy.optimize.minimize_scalar over T in (0.01, 10.0).
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)

    def nll(t: float) -> float:
        scaled = logits / t
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_softmax = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return float(-log_softmax[np.arange(len(labels)), labels].mean())

    result = scipy.optimize.minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
    return float(result.x)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probability array.

    probs: shape (N, K) already-normalised probabilities
    Converts back to logits (log), divides by T, re-softmaxes.
    Returns shape (N, K).
    """
    probs = np.asarray(probs, dtype=float)
    logits = np.log(np.clip(probs, 1e-10, None))
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


def fit_isotonic(
    probs: np.ndarray,
    outcomes: np.ndarray,
) -> IsotonicRegression:
    """Fit a 1D isotonic regression calibrator for a binary market.

    probs: shape (N,) model probability for the positive class
    outcomes: shape (N,) binary 0/1 outcomes
    Returns fitted IsotonicRegression (sklearn).
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(probs, outcomes)
    return ir


def reliability_diagram(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = N_BINS,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    """Plot a reliability diagram (calibration curve) for a binary market.

    Bins predictions into n_bins equal-width bins.
    For each bin: plot (mean predicted prob, fraction positive).
    Also plots the diagonal (perfect calibration) and a histogram of predictions.
    Saves to save_path if provided, otherwise shows.
    Bins with fewer than MIN_BIN_SIZE samples are plotted with hollow markers.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges[1:-1])

    mean_pred: list[float] = []
    frac_pos: list[float] = []
    bin_sizes: list[int] = []

    for b in range(n_bins):
        mask = bin_indices == b
        n = int(mask.sum())
        bin_sizes.append(n)
        if n == 0:
            mean_pred.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))
            frac_pos.append(float("nan"))
        else:
            mean_pred.append(float(probs[mask].mean()))
            frac_pos.append(float(outcomes[mask].mean()))

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax_main.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    for _i, (mp, fp, n) in enumerate(zip(mean_pred, frac_pos, bin_sizes, strict=False)):
        if np.isnan(fp):
            continue
        if n < MIN_BIN_SIZE:
            ax_main.plot(mp, fp, marker="o", fillstyle="none", color="steelblue", markersize=8)
        else:
            ax_main.plot(mp, fp, marker="o", color="steelblue", markersize=8)

    ax_main.set_ylabel("Fraction of positives")
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.legend(loc="upper left")
    if title:
        ax_main.set_title(title)

    ax_hist.hist(probs, bins=bin_edges, color="steelblue", alpha=0.7)
    ax_hist.set_xlabel("Mean predicted probability")
    ax_hist.set_ylabel("Count")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def expected_calibration_error(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = N_BINS,
) -> float:
    """Compute ECE: weighted mean absolute calibration error across bins."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges[1:-1])
    n_total = len(probs)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(probs[mask].mean())
        frac_pos = float(outcomes[mask].mean())
        ece += (n / n_total) * abs(mean_pred - frac_pos)

    return float(ece)
