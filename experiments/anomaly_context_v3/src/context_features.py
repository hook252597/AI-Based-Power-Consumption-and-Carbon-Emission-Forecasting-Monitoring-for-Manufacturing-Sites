"""Causal minute features for a retrospective pattern-detection experiment.

Every row is available only at the END of its timestamped minute. The input
activePower is in W, voltage is in V, current is in A, and power factors are
percent. The two electrical-relation features are observational proxies, not
fault criteria; wiring, sensor conventions, and harmonics may affect them.

Past windows exclude the current minute. A row needs its own complete minute
plus five immediately preceding complete minutes, within the same local day.
A missing/incomplete minute resets that history. Labels and rule thresholds are
never inspected. Any baseline/scaler must be fitted separately on training data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS = [
    "activePower", "voltageR", "voltageS", "voltageT",
    "currentR", "currentS", "currentT",
    "powerFactorR", "powerFactorS", "powerFactorT",
]
FAMILIES = ["voltage", "current", "powerFactor"]
PAIRS = [("R", "S"), ("R", "T"), ("S", "T")]

STATIC_FEATURES = (
    [f"{sensor}_{stat}" for sensor in SENSORS for stat in ("mean", "std")]
    + [f"{family}_{left}_minus_{right}" for family in FAMILIES for left, right in PAIRS]
    + [f"{family}_relative_range" for family in FAMILIES]
    + ["power_minus_vi_pf", "current_pf_weighted_sum"]
)
CONTEXT_FEATURES = (
    [f"{sensor}_delta_prev" for sensor in SENSORS]
    + [f"{sensor}_delta_past5_mean" for sensor in SENSORS]
    + [f"{sensor}_past5_std" for sensor in SENSORS]
)
ALL_FEATURES = STATIC_FEATURES + CONTEXT_FEATURES
REQUIRED_COLUMNS = ["module", "timestamp", "n"] + [
    f"{sensor}_{stat}" for sensor in SENSORS for stat in ("mean", "std")
]


def _relative_range(values: pd.DataFrame) -> pd.Series:
    """(max-min)/abs(three-phase mean); observed all-zero phases have zero spread.

    All three phases must be present. A zero mean with nonzero spread remains
    undefined (NaN). Missing observations are never filled with zero.
    """
    present = np.isfinite(values).all(axis=1)
    spread = values.max(axis=1, skipna=False) - values.min(axis=1, skipna=False)
    center = values.mean(axis=1, skipna=False).abs()
    result = spread.div(center.where(center.ne(0)))
    result.loc[present & center.eq(0) & spread.eq(0)] = 0.0
    return result.where(present)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return 64 numeric features and bool ``valid``, in input index/order.

    ``timestamp`` is the minute START, timezone-naive Korean local time. A row
    can be scored at ``timestamp + 1 minute``. Current mean/std and phase
    relations use only the 12 five-second observations in that minute. The
    previous-minute difference equals current mean minus shift(1). Five-minute
    baseline mean/std use shift(1).rolling(5), std ddof=0. Thus a prefix of input
    produces identical features to the same prefix of the full input.

    A valid row has n=12, complete_minute=True when that column is provided,
    finite current mean/std, nonnegative std, and five
    consecutive preceding complete minutes in the SAME local day. Missing or
    incomplete rows, timestamp gaps, and midnight reset history. All 64 feature
    values must be finite. Invalid rows remain in the output with their observed
    numeric values/NaNs and valid=False. No interpolation or zero filling occurs.

    Duplicate (module, timestamp), missing identifiers, off-minute timestamps,
    and timezone-aware timestamps are rejected rather than silently combined.
    The equipment identifier may be named module or module(equipment).
    Additional input columns (including labels) are ignored.
    """
    if "module(equipment)" in frame.columns:
        if "module" in frame.columns:
            if not frame["module"].equals(frame["module(equipment)"]):
                raise ValueError("module and module(equipment) identifiers disagree")
        else:
            frame = frame.rename(columns={"module(equipment)": "module"})
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame["module"].isna().any():
        raise ValueError("module must not contain missing identifiers")

    selected_columns = REQUIRED_COLUMNS + (["complete_minute"] if "complete_minute" in frame.columns else [])
    work = frame.loc[:, selected_columns].copy().reset_index(drop=True)
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="raise")
    if work["timestamp"].isna().any():
        raise ValueError("timestamp must not contain missing values")
    if getattr(work["timestamp"].dt, "tz", None) is not None:
        raise ValueError("timestamp must be timezone-naive Korean local time")
    if not work["timestamp"].eq(work["timestamp"].dt.floor("min")).all():
        raise ValueError("timestamp must identify the exact start of a minute")
    if work.duplicated(["module", "timestamp"]).any():
        raise ValueError("Duplicate module/timestamp rows must be resolved upstream")

    numeric_columns = [c for c in REQUIRED_COLUMNS if c not in ("module", "timestamp")]
    work[numeric_columns] = work[numeric_columns].apply(pd.to_numeric, errors="raise")
    feature_values = np.full((len(work), len(ALL_FEATURES)), np.nan, dtype=np.float64)
    validity = np.zeros(len(work), dtype=bool)
    mean_columns = [f"{s}_mean" for s in SENSORS]
    std_columns = [f"{s}_std" for s in SENSORS]

    for _, source in work.groupby("module", sort=False, dropna=False):
        group = source.sort_values("timestamp", kind="stable")
        positions = group.index.to_numpy()
        group = group.reset_index(drop=True)
        current_complete = (
            group["n"].eq(12)
            & np.isfinite(group[mean_columns + std_columns]).all(axis=1)
            & group[std_columns].ge(0).all(axis=1)
        )
        if "complete_minute" in group.columns:
            current_complete &= group["complete_minute"].eq(True).fillna(False)
        continuous = group["timestamp"].diff().eq(pd.Timedelta(minutes=1))
        same_day = group["timestamp"].dt.normalize().eq(
            group["timestamp"].shift(1).dt.normalize()
        )
        boundary = (
            ~continuous | ~same_day | ~current_complete
            | ~current_complete.shift(1, fill_value=False)
        )
        segments = boundary.cumsum()
        values = group[[f"{s}_{v}" for s in SENSORS for v in ("mean", "std")]].copy()
        for family in FAMILIES:
            for left, right in PAIRS:
                values[f"{family}_{left}_minus_{right}"] = (
                    group[f"{family}{left}_mean"] - group[f"{family}{right}_mean"]
                )
            phase = group[[f"{family}{p}_mean" for p in "RST"]]
            values[f"{family}_relative_range"] = _relative_range(phase)

        weighted_current = pd.DataFrame({
            phase: group[f"current{phase}_mean"] * group[f"powerFactor{phase}_mean"] / 100.0
            for phase in "RST"
        })
        vi_pf_proxy = pd.DataFrame({
            phase: group[f"voltage{phase}_mean"] * weighted_current[phase]
            for phase in "RST"
        }).sum(axis=1, min_count=3)
        values["power_minus_vi_pf"] = group["activePower_mean"] - vi_pf_proxy
        values["current_pf_weighted_sum"] = weighted_current.sum(axis=1, min_count=3)

        for sensor in SENSORS:
            means = group[f"{sensor}_mean"]
            by_segment = means.groupby(segments, sort=False)
            previous = by_segment.shift(1)
            past5_mean = by_segment.transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
            past5_std = by_segment.transform(lambda x: x.shift(1).rolling(5, min_periods=5).std(ddof=0))
            values[f"{sensor}_delta_prev"] = means - previous
            values[f"{sensor}_delta_past5_mean"] = means - past5_mean
            values[f"{sensor}_past5_std"] = past5_std

        ordered = values.loc[:, ALL_FEATURES]
        enough_history = group.groupby(segments, sort=False).cumcount().ge(5)
        valid = current_complete & enough_history & np.isfinite(ordered).all(axis=1)
        feature_values[positions] = ordered.to_numpy(dtype=np.float64)
        validity[positions] = valid.to_numpy(dtype=bool)

    result = pd.DataFrame(feature_values, index=frame.index, columns=ALL_FEATURES)
    result["valid"] = validity
    return result
