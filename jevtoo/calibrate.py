"""Turning raw readouts into numbers you can actually branch on.

A raw language-model likelihood is a sufficient statistic, not a probability.
On MiniCPM5-2B the letter readout happened to be well calibrated out of the box
(ECE 0.040, 1/55 confidently wrong), while the text readout was not (ECE 0.065
only after fitting). Do not assume either way: measure, then fit if needed.

Everything here is plain Python so the package has no hard numeric dependency.
"""
from __future__ import annotations

import math


def sigmoid(z: float) -> float:
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def fit_platt(xs: list[float], ys: list[int], iters: int = 6000, lr: float = 0.5) -> tuple[float, float]:
    """Fit p = sigmoid(a*x + b). `xs` should be log-likelihoods, `ys` 0/1 labels."""
    a, b = 1.0, 0.0
    n = max(len(xs), 1)
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            e = sigmoid(a * x + b) - y
            ga += e * x
            gb += e
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def apply_platt(a: float, b: float, x: float) -> float:
    return sigmoid(a * x + b)


def ece(pairs: list[tuple[float, bool]], n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error over (confidence, correct) pairs.

    Always read an ECE next to its noise floor: the value a perfectly calibrated
    model would still show at that sample size. A raw ECE with no floor beside it
    is not evidence (see the jev-calibration-audit for the same point).
    """
    bins = [{"n": 0, "cs": 0.0, "cor": 0} for _ in range(n_bins)]
    for conf, ok in pairs:
        c = min(max(float(conf), 0.0), 1.0)
        b = bins[min(int(c * n_bins), n_bins - 1)]
        b["n"] += 1
        b["cs"] += c
        b["cor"] += 1 if ok else 0
    total = sum(b["n"] for b in bins)
    if not total:
        return 0.0, []
    out, detail = 0.0, []
    for i, b in enumerate(bins):
        if not b["n"]:
            continue
        acc = b["cor"] / b["n"]
        conf = b["cs"] / b["n"]
        out += (b["n"] / total) * abs(acc - conf)
        detail.append({"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": b["n"],
                       "confidence": conf, "accuracy": acc})
    return out, detail


def noise_floor(n: int, n_bins: int = 10) -> float:
    """Rough ECE noise floor at sample size n with n_bins bins.

    Sampling noise alone makes a perfectly calibrated model show a non-zero ECE.
    The jev-calibration-audit quotes floors of 0.022-0.055 at n in the low
    hundreds; scaling as ~1/sqrt(n) is enough to keep yourself honest.
    """
    if n <= 0:
        return float("nan")
    return 1.0 / math.sqrt(n) / math.sqrt(n_bins)


def abstain_curve(pairs: list[tuple[float, bool]], thresholds: list[float] | None = None) -> list[dict]:
    """Coverage vs accuracy when you only act above a confidence threshold.

    This is the axis that decides whether a decision model is deployable: an
    89% model that knows when it is unsure is more useful than a 92% model that
    does not.
    """
    thresholds = thresholds if thresholds is not None else [0.0, 0.5, 0.7, 0.85, 0.95, 0.99]
    rows = []
    for th in thresholds:
        sub = [(c, ok) for c, ok in pairs if c >= th]
        rows.append({
            "threshold": th,
            "coverage": len(sub) / len(pairs) if pairs else 0.0,
            "accuracy": (sum(1 for _, ok in sub if ok) / len(sub)) if sub else None,
            "errors": sum(1 for _, ok in sub if not ok),
        })
    return rows


def chance_corrected(accuracy: float, chance: float) -> float:
    """JevBench-style accuracy above an item-specific guessing baseline,
    clipped to [0, 100]."""
    if chance >= 1:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (accuracy - chance) / (1.0 - chance)))
