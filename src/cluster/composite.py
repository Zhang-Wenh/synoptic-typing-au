"""Composites and sequence diagnostics for the weather types.

Composite maps show what each type looks like. The sequence diagnostics say
what the set of types is: a collection of distinct states, or a discretisation
of something that moves continuously.

The transition matrix is the sharpest test available here. If the leading EOFs
are quadrature pairs of an eastward-propagating wave, days fill a ring in PC
space and k-means cuts that ring into sectors. Consecutive days then step
around the ring in a preferred direction, so the transition matrix carries a
strong cyclic component: type i is followed by type i+1 far more often than by
type i-1. Genuinely distinct regimes have no reason to be ordered that way.

Persistence is the companion measure. Sectors of a ring are crossed at the
speed the wave propagates, giving short and rather uniform residence times.
Regimes persist irregularly, and the spread of run lengths across types is
wider.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

ACCUM = "float64"


def composite(field: xr.DataArray, labels: xr.DataArray, k: int) -> xr.DataArray:
    """Mean field for each type.

    Accumulates in float64 for the same reason as everywhere else in this
    project: each type averages roughly two thousand days of a long float
    record.
    """
    aligned = labels.reindex(time=field["time"])
    if bool(aligned.isnull().any()):
        raise ValueError("labels do not cover every time step of the field")

    out = (
        field.astype(ACCUM)
        .groupby(aligned.rename("type_index"))
        .mean()
        .reindex(type_index=np.arange(k))
    )
    out.attrs = dict(field.attrs)
    out.attrs["composite"] = "mean over days assigned to each type"
    return out


def seasonal_distribution(labels: xr.DataArray, k: int) -> xr.DataArray:
    """Count of days per type per calendar month.

    A type confined to one season is a seasonal circulation state; one spread
    evenly across the year is not. Since the anomalies were deseasonalised
    before classification, strong seasonality here is a real result rather
    than a leftover annual cycle.
    """
    months = labels["time"].dt.month.values
    counts = np.zeros((k, 12), dtype=np.int64)
    np.add.at(counts, (labels.values, months - 1), 1)

    return xr.DataArray(
        counts,
        dims=("type_index", "month"),
        coords={"type_index": np.arange(k), "month": np.arange(1, 13)},
        name="seasonal_counts",
    )


def run_lengths(labels: np.ndarray, k: int) -> list[np.ndarray]:
    """Lengths of every consecutive run of each type, in days."""
    changes = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [labels.size]])

    runs: list[list[int]] = [[] for _ in range(k)]
    for start, end in zip(starts, ends):
        runs[labels[start]].append(end - start)
    return [np.array(r) for r in runs]


def persistence(labels: np.ndarray, k: int) -> xr.Dataset:
    """Mean and median run length per type."""
    runs = run_lengths(labels, k)
    return xr.Dataset(
        {
            "mean_run": ("type_index", np.array([r.mean() if r.size else np.nan for r in runs])),
            "median_run": ("type_index", np.array([np.median(r) if r.size else np.nan for r in runs])),
            "n_runs": ("type_index", np.array([r.size for r in runs])),
        },
        coords={"type_index": np.arange(k)},
    )


def transition_matrix(labels: np.ndarray, k: int, normalise: bool = True) -> np.ndarray:
    """Day-to-day transition counts, or probabilities if normalised.

    Element (i, j) is the frequency with which a day of type i is followed by
    a day of type j. The diagonal is persistence.
    """
    counts = np.zeros((k, k), dtype=ACCUM)
    np.add.at(counts, (labels[:-1], labels[1:]), 1.0)

    if normalise:
        totals = counts.sum(axis=1, keepdims=True)
        counts = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    return counts


def cyclic_asymmetry(transitions: np.ndarray, order: np.ndarray | None = None) -> float:
    """How much more often the sequence steps one way around a cycle.

    Returns (forward - backward) / (forward + backward), using off-diagonal
    transitions only. Zero means no preferred direction. A value approaching
    one means the types are traversed in a fixed cyclic order, which is what a
    propagating wave sliced into sectors produces.

    `order` is the arrangement of types around the cycle. When it is not
    given, the types are assumed to be in cyclic order already, which is only
    meaningful after `cyclic_order` has been used to find one.
    """
    k = transitions.shape[0]
    idx = np.arange(k) if order is None else np.asarray(order)
    position = np.empty(k, dtype=np.int64)
    position[idx] = np.arange(k)

    forward = backward = 0.0
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            step = (position[j] - position[i]) % k
            if step == 1:
                forward += transitions[i, j]
            elif step == k - 1:
                backward += transitions[i, j]

    total = forward + backward
    return float((forward - backward) / total) if total > 0 else 0.0


def cyclic_order(transitions: np.ndarray) -> np.ndarray:
    """Arrange types into the cycle the transitions most favour.

    Greedy: start from the type with the strongest single outgoing transition
    and repeatedly follow the largest unused successor. This is a heuristic,
    not an optimal Hamiltonian cycle, which is enough for a diagnostic -- if a
    strong cycle exists the greedy walk finds it, and if none exists the
    asymmetry computed from any ordering will be near zero anyway.
    """
    k = transitions.shape[0]
    off = transitions.copy()
    np.fill_diagonal(off, -np.inf)

    start = int(np.unravel_index(np.argmax(off), off.shape)[0])
    order = [start]
    remaining = set(range(k)) - {start}

    while remaining:
        current = order[-1]
        nxt = max(remaining, key=lambda j: off[current, j])
        order.append(nxt)
        remaining.remove(nxt)
    return np.array(order)


def sequence_report(labels: np.ndarray, k: int) -> dict:
    """Everything needed to judge whether the types are states or sectors."""
    transitions = transition_matrix(labels, k)
    order = cyclic_order(transitions)
    runs = persistence(labels, k)

    self_transition = float(np.diag(transitions).mean())
    return {
        "transitions": transitions,
        "cyclic_order": order,
        "cyclic_asymmetry": cyclic_asymmetry(transitions, order),
        "mean_persistence_days": float(runs["mean_run"].mean()),
        "persistence_spread": float(runs["mean_run"].std()),
        "mean_self_transition": self_transition,
    }
