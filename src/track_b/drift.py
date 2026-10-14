"""Distribution-comparison helpers (PSI, KS) for the Track B drift report."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_EPS = 1e-4


def psi_from_proportions(reference: np.ndarray, current: np.ndarray) -> float:
    """Population Stability Index between two aligned probability vectors."""
    ref = np.clip(np.asarray(reference, dtype=float), _EPS, None)
    cur = np.clip(np.asarray(current, dtype=float), _EPS, None)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI with bin edges taken from the reference distribution's quantiles."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant reference
        edges = np.array([-np.inf, float(reference.iloc[0]), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    return psi_from_proportions(ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum())


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference.unique()) | set(current.unique()))
    ref = reference.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return psi_from_proportions(ref, cur)


def ks_test(reference: pd.Series, current: pd.Series) -> dict:
    res = stats.ks_2samp(reference, current)
    return {"statistic": round(float(res.statistic), 4), "p_value": float(res.pvalue)}


def psi_label(value: float) -> str:
    return "major shift" if value >= 0.25 else "moderate shift" if value >= 0.1 else "no meaningful shift"
