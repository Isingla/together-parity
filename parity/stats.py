"""Paired statistics for per-prompt comparisons."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


def _boot_idx(n: int, n_resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


@dataclass
class PairedAccuracy:
    n: int
    acc_ref: float
    acc_test: float
    delta: float          # test - ref, in [0,1]
    ci95: tuple[float, float]
    ci90: tuple[float, float]
    ref_only: int         # ref right, test wrong
    test_only: int        # test right, ref wrong
    mcnemar_p: float
    margin: float
    equivalent: bool      # TOST: 90% CI inside +/- margin
    different: bool       # 95% CI excludes 0 (and McNemar p<0.05)

    def verdict(self) -> str:
        pp = lambda x: f"{100 * x:+.1f} pp"
        ci = f"95% CI [{pp(self.ci95[0])}, {pp(self.ci95[1])}]"
        if self.different:
            return f"significant difference: {pp(self.delta)} ({ci}, McNemar p={self.mcnemar_p:.3g})"
        if self.ci95[0] > 0 or self.ci95[1] < 0:
            return (f"weak evidence of a difference: {pp(self.delta)} ({ci} excludes 0, but exact "
                    f"McNemar p={self.mcnemar_p:.3g} on only {self.ref_only + self.test_only} "
                    f"discordant prompts)")
        if self.equivalent:
            return (f"equivalent within ±{100 * self.margin:.1f} pp: {pp(self.delta)} "
                    f"({ci}, McNemar p={self.mcnemar_p:.3g})")
        return (f"inconclusive: no significant difference but CI too wide to claim "
                f"equivalence within ±{100 * self.margin:.1f} pp ({pp(self.delta)}, {ci})")


def paired_accuracy(ref: np.ndarray, test: np.ndarray, *, margin: float,
                    n_resamples: int = 10000, seed: int = 0) -> PairedAccuracy:
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    assert ref.shape == test.shape and ref.ndim == 1
    n = len(ref)
    diff = test - ref
    idx = _boot_idx(n, n_resamples, seed)
    boots = diff[idx].mean(axis=1)
    ci95 = tuple(np.percentile(boots, [2.5, 97.5]))
    ci90 = tuple(np.percentile(boots, [5, 95]))
    ref_only = int(((ref == 1) & (test == 0)).sum())
    test_only = int(((ref == 0) & (test == 1)).sum())
    disc = ref_only + test_only
    p = binomtest(test_only, disc, 0.5).pvalue if disc else 1.0
    if disc == 0:
        # Bootstrap collapses to [0, 0] with no discordant prompts, which would
        # "prove" equivalence at any n. Use the exact zero-event upper bound on the
        # discordance rate instead (rule of three: 95% -> 3/n, 90% -> 2.3/n).
        ci95, ci90 = (-3.0 / n, 3.0 / n), (-2.3 / n, 2.3 / n)
    return PairedAccuracy(
        n=n, acc_ref=ref.mean(), acc_test=test.mean(), delta=diff.mean(),
        ci95=ci95, ci90=ci90, ref_only=ref_only, test_only=test_only, mcnemar_p=p,
        margin=margin,
        equivalent=bool(ci90[0] > -margin and ci90[1] < margin),
        different=bool((ci95[0] > 0 or ci95[1] < 0) and p < 0.05),
    )


@dataclass
class FlipRate:
    """Does the optimized path change answers more than the reference path
    changes its *own* answers between two identical runs?"""
    n: int
    cross: float            # P(answer_ref != answer_test)
    noise: float            # P(answer_ref_run1 != answer_ref_run2)
    excess: float           # cross - noise
    ci95: tuple[float, float]

    @property
    def significant(self) -> bool:
        return self.ci95[0] > 0


def flip_rate(ref1: list, ref2: list, test: list, *, n_resamples: int = 10000,
              seed: int = 0) -> FlipRate:
    a = np.array([x != y for x, y in zip(ref1, test)], dtype=float)
    b = np.array([x != y for x, y in zip(ref1, ref2)], dtype=float)
    d = a - b
    idx = _boot_idx(len(d), n_resamples, seed)
    boots = d[idx].mean(axis=1)
    return FlipRate(len(d), a.mean(), b.mean(), d.mean(),
                    tuple(np.percentile(boots, [2.5, 97.5])))


def ratio_of_medians(num: np.ndarray, den: np.ndarray, *, n_resamples: int = 10000,
                     seed: int = 0) -> tuple[float, tuple[float, float]]:
    """median(num)/median(den) with independent bootstrap CI (speed samples aren't paired)."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    rng = np.random.default_rng(seed)
    bn = np.median(num[rng.integers(0, len(num), (n_resamples, len(num)))], axis=1)
    bd = np.median(den[rng.integers(0, len(den), (n_resamples, len(den)))], axis=1)
    r = bn / bd
    return float(np.median(num) / np.median(den)), tuple(np.percentile(r, [2.5, 97.5]))
