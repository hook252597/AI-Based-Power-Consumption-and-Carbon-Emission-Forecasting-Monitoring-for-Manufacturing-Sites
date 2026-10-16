"""Causal minute-level event evaluation and controlled synthetic feature probes.

These are tests on ONE-MINUTE AGGREGATES, not injected 5-second sensor traces.
They cannot establish real equipment-fault precision, recall, or safety limits.
No rule-derived label is used for model selection or evaluation here.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np
import pandas as pd

SENSORS = ("activePower", "voltageR", "voltageS", "voltageT", "currentR", "currentS", "currentT", "powerFactorR", "powerFactorS", "powerFactorT")
RULE_LIMITS = {**{f"voltage{p}": 210.0 for p in "RST"}, **{f"voltage{p}": 360.0 for p in ("RS", "ST", "TR")}, **{f"powerFactor{p}": 85.0 for p in "RST"}}
PROBE_KINDS = ("level_shift", "volatility_increase", "phase_imbalance", "relation_break", "persistent_drift")
EVENT_COLUMNS = ["module", "start", "end", "decision_time", "duration_minutes", "flagged_minutes", "observed_seconds", "alert_active_minutes"]
TEST_NOTICE = "One-minute aggregate synthetic functionality test; not a raw 5-second injection or real-fault accuracy measurement."


def _prepared(frame: pd.DataFrame, flags: Sequence[bool]) -> pd.DataFrame:
    if len(frame) != len(flags):
        raise ValueError("Flags must have exactly one value per input row, in input row order.")
    module_column = "module" if "module" in frame else "module(equipment)"
    required = {module_column, "timestamp", "n"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing minute columns: {sorted(missing)}")
    out = frame[[module_column, "timestamp", "n"]].rename(columns={module_column: "module"}).copy()
    out["complete"] = out["n"].eq(12)
    if "complete_minute" in frame:
        out["complete"] &= frame["complete_minute"].fillna(False).to_numpy(dtype=bool)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if out["timestamp"].isna().any():
        raise ValueError("Minute timestamps may not be missing.")
    if (out["timestamp"] != out["timestamp"].dt.floor("min")).any():
        raise ValueError("timestamp must be the beginning of its minute.")
    if out.duplicated(["module", "timestamp"]).any():
        raise ValueError("Duplicate equipment/minute rows are ambiguous.")
    values = pd.Series(np.asarray(flags), dtype="boolean").fillna(False).to_numpy(dtype=bool)
    out["flag"] = values & out["complete"].to_numpy()
    return out.sort_values(["module", "timestamp"], kind="stable").reset_index(drop=True)


def group_alert_events(frame: pd.DataFrame, flags: Sequence[bool], min_consecutive: int = 3) -> pd.DataFrame:
    """Group consecutive flagged complete minutes; alert at third minute END.

    Gaps, an unflagged minute, or incomplete n != 12 break an incident. start/end
    delimit a half-open evidence interval. A persistent run produces one alert.
    duration_minutes includes the evidence preceding the first alert decision.
    """
    return _events_prepared(_prepared(frame, flags), min_consecutive)


def _events_prepared(out: pd.DataFrame, min_consecutive: int) -> pd.DataFrame:
    if not isinstance(min_consecutive, int) or min_consecutive < 1:
        raise ValueError("min_consecutive must be a positive integer.")
    flag = out["flag"].to_numpy(dtype=bool)
    if not flag.any():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    adjacent = (out["module"].eq(out["module"].shift()) & out["timestamp"].diff().eq(pd.Timedelta(minutes=1))).to_numpy(dtype=bool)
    starts = np.flatnonzero(flag & ~np.r_[False, flag[:-1] & adjacent[1:]])
    ends = np.flatnonzero(flag & ~np.r_[flag[1:] & adjacent[1:], False])
    lengths = ends - starts + 1
    keep = lengths >= min_consecutive
    starts, ends, lengths = starts[keep], ends[keep], lengths[keep]
    start_times = out["timestamp"].iloc[starts].to_numpy()
    return pd.DataFrame({"module": out["module"].iloc[starts].to_numpy(),
                         "start": start_times,
                         "end": out["timestamp"].iloc[ends].to_numpy() + np.timedelta64(1, "m"),
                         "decision_time": start_times + np.timedelta64(min_consecutive, "m"),
                         "duration_minutes": lengths, "flagged_minutes": lengths,
                         "observed_seconds": lengths * 60,
                         "alert_active_minutes": lengths - min_consecutive + 1}, columns=EVENT_COLUMNS)



def summarize_alerts(frame: pd.DataFrame, flags: Sequence[bool], min_consecutive: int = 3) -> dict:
    """Count observable sample-time, raw runs, and actionable persistent runs.

    observed_hours sums equipment observation hours; it is not wall-clock hours.
    No flag is reported as a confirmed fault or false positive.
    """
    out = _prepared(frame, flags)
    events = _events_prepared(out, min_consecutive)
    raw_events = _events_prepared(out, 1)
    complete = out["complete"]
    observed_minutes = int(complete.sum())
    observed_samples = int(out["n"].clip(lower=0, upper=12).fillna(0).sum())
    hours = observed_minutes / 60.0
    count = len(events)
    return {"rows": len(out), "complete_minutes": observed_minutes,
            "incomplete_minutes": int((~complete).sum()), "observed_samples": observed_samples,
            "sample_observed_hours": observed_samples * 5 / 3600.0,
            "observed_equipment_hours": hours, "equipment_count": int(out["module"].nunique()),
            "flagged_minutes": int(out["flag"].sum()), "raw_run_count": len(raw_events),
            "alert_count": count, "alert_equipment_count": int(events["module"].nunique()),
            "persistent_flagged_minutes": int(events["flagged_minutes"].sum()),
            "longest_flagged_run_minutes": int(events["duration_minutes"].max()) if count else 0,
            "alerts_per_100_equipment_hours": count / hours * 100 if hours else None,
            "alerts_per_24_equipment_hours": count / hours * 24 if hours else None,
            "min_consecutive_minutes": min_consecutive,
            "definition": "One alert per equipment and contiguous flagged run; decision after minimum complete minutes. No actual fault labels."}


def fixed_rule_flags(frame: pd.DataFrame, limits: Mapping[str, float] | None = None) -> pd.Series:
    """Original physical lower-bound comparator, independent of AI features."""
    selected = RULE_LIMITS if limits is None else limits
    missing = [f"{key}_min" for key in selected if f"{key}_min" not in frame]
    if missing:
        raise ValueError(f"Missing rule inputs: {missing}")
    return pd.DataFrame({key: frame[f"{key}_min"].lt(value) for key, value in selected.items()}, index=frame.index).any(axis=1)


def assert_aggregate_consistency(frame: pd.DataFrame, *, require_rule_safe: bool = False) -> None:
    """Validate bounds and feasible first/second moments of minute aggregates."""
    for sensor in SENSORS:
        columns = [f"{sensor}_{suffix}" for suffix in ("mean", "std", "min", "max")]
        if not set(columns).issubset(frame.columns):
            raise ValueError(f"Missing aggregate columns for {sensor}")
        mean, std, low, high = [frame[col].to_numpy(dtype=float) for col in columns]
        if not np.isfinite(np.stack([mean, std, low, high])).all():
            raise ValueError(f"Non-finite aggregate for {sensor}")
        tol = np.maximum(np.abs(mean) * 1e-6, 1e-5)
        if np.any(low > mean + tol) or np.any(mean > high + tol) or np.any(std < 0):
            raise ValueError(f"Inconsistent min/mean/max/std for {sensor}")
        # Bhatia-Davis bound; rounding tolerance accommodates stored float32 moments.
        if np.any(std ** 2 > (high - mean) * (mean - low) + tol * (np.abs(mean) + 1)):
            raise ValueError(f"Standard deviation exceeds feasible range for {sensor}")
        if sensor.startswith("voltage") and np.any(low <= 0):
            raise ValueError("Voltage must remain positive.")
        if sensor.startswith("powerFactor") and (np.any(low < 0) or np.any(high > 100)):
            raise ValueError("Power factor must remain within 0..100 percent.")
    if require_rule_safe and fixed_rule_flags(frame).any():
        raise ValueError("Synthetic probe crosses an original rule limit.")


def _scale(scales: Mapping, module, sensor: str) -> float:
    equipment = scales.get(str(module), scales.get(module, {}))
    result = equipment.get(sensor, equipment.get(f"{sensor}_mean", np.nan))
    if isinstance(result, Mapping):
        result = result.get("scale", np.nan)
    result = float(result)
    return result if np.isfinite(result) and result > 0 else np.nan


def _shift(frame: pd.DataFrame, sensor: str, delta: np.ndarray | float) -> None:
    for stat in ("mean", "min", "max"):
        frame[f"{sensor}_{stat}"] = frame[f"{sensor}_{stat}"].to_numpy(dtype=float) + delta


def _inject_original(segment: pd.DataFrame, kind: str, scales: Mapping, module) -> tuple[pd.DataFrame, dict] | None:
    """Apply an explicitly controlled moment transform to the injection rows."""
    out = segment.copy(deep=True)
    details = {"severity": "moderate, training-scale based and clipped to stay inside original rules", "changes": {}}
    margin = 0.05
    if kind in {"level_shift", "persistent_drift"}:
        for sensor in ("voltageR", "voltageS", "voltageT"):
            scale = _scale(scales, module, sensor)
            if not np.isfinite(scale):
                return None
            allowed = float((out[f"{sensor}_min"] - RULE_LIMITS[sensor] - margin).min())
            delta = min(3.0 * scale, allowed * 0.65)
            if delta < scale * 0.5:
                return None
            changes = -delta if kind == "level_shift" else -np.linspace(delta / len(out), delta, len(out))
            _shift(out, sensor, changes)
            details["changes"][sensor] = {"final_mean_delta": -delta, "training_scale": scale, "final_scale_units": delta / scale}
    elif kind == "phase_imbalance":
        for sensor, direction in (("voltageR", 1), ("voltageS", -1)):
            scale = _scale(scales, module, sensor)
            if not np.isfinite(scale):
                return None
            allowed = float((out[f"{sensor}_min"] - RULE_LIMITS[sensor] - margin).min()) if direction < 0 else float((300 - out[f"{sensor}_max"]).min())
            delta = min(3.0 * scale, allowed * 0.65)
            if delta < scale * 0.5:
                return None
            _shift(out, sensor, direction * delta)
            details["changes"][sensor] = {"mean_delta": direction * delta, "training_scale": scale, "scale_units": delta / scale}
    elif kind == "relation_break":
        # Current changes while observed power/voltage/PF do not; this is a
        # sensor-relation consistency probe, not a claim about physical failure.
        for sensor in ("currentR", "currentS", "currentT"):
            scale = _scale(scales, module, sensor)
            if not np.isfinite(scale):
                return None
            delta = 4.0 * scale
            _shift(out, sensor, delta)
            details["changes"][sensor] = {"mean_delta": delta, "training_scale": scale, "scale_units": 4.0}
        details["interpretation"] = "Current-versus-power relationship perturbation; could represent measurement inconsistency."
    elif kind == "volatility_increase":
        for sensor in ("voltageR", "voltageS", "voltageT"):
            scale = _scale(scales, module, sensor)
            if not np.isfinite(scale):
                return None
            mean = out[f"{sensor}_mean"].to_numpy(dtype=float)
            prior = out[f"{sensor}_std"].to_numpy(dtype=float)
            requested = np.maximum(prior * 3.0, scale * 2.0)
            allowed = np.minimum(mean - RULE_LIMITS[sensor] - margin, 300.0 - mean)
            amp = np.minimum(requested, allowed * 0.65)
            if np.any(amp <= prior * 1.25) or np.any(amp <= 0):
                return None
            # Six samples at mean-amp and six at mean+amp have these exact
            # ddof=0 moments. No raw sample trace is emitted or asserted.
            out[f"{sensor}_std"] = amp
            out[f"{sensor}_min"] = mean - amp
            out[f"{sensor}_max"] = mean + amp
            details["changes"][sensor] = {"median_std_before": float(np.median(prior)), "median_std_after": float(np.median(amp)), "training_scale": scale}
    else:
        raise ValueError(f"Unknown synthetic kind: {kind}")
    try:
        assert_aggregate_consistency(out, require_rule_safe=True)
    except ValueError:
        return None
    return out, details


def _inject(segment: pd.DataFrame, kind: str, scales: Mapping, module) -> tuple[pd.DataFrame, dict] | None:
    """Apply an explicitly controlled moment transform to the injection rows."""
    out = segment.copy(deep=True)
    details = {"severity": "moderate, training-scale based and clipped to stay inside original rules", "changes": {}}
    margin = 0.05
    details["feasibility_revision"] = "2026-10-16: source voltage minima clipped at 210V; fixed direction changed without inspecting model scores."
    if kind in {"level_shift", "persistent_drift"}:
        for sensor in ("voltageR", "voltageS", "voltageT"):
            scale = _scale(scales, module, sensor)
            if not np.isfinite(scale):
                return None
            allowed = float((300.0 - out[f"{sensor}_max"] - margin).min())
            delta = min(3.0 * scale, allowed * 0.65)
            if delta < scale * 0.5:
                return None
            changes = delta if kind == "level_shift" else np.linspace(delta / len(out), delta, len(out))
            _shift(out, sensor, changes)
            details["changes"][sensor] = {"final_mean_delta": delta, "training_scale": scale, "final_scale_units": delta / scale}
    elif kind == "phase_imbalance":
        for sensor, direction in (("voltageR", 1),):
            scale = _scale(scales, module, sensor)
            if not np.isfinite(scale):
                return None
            allowed = float((out[f"{sensor}_min"] - RULE_LIMITS[sensor] - margin).min()) if direction < 0 else float((300 - out[f"{sensor}_max"]).min())
            delta = min(3.0 * scale, allowed * 0.65)
            if delta < scale * 0.5:
                return None
            _shift(out, sensor, direction * delta)
            details["changes"][sensor] = {"mean_delta": direction * delta, "training_scale": scale, "scale_units": delta / scale}
    elif kind == "relation_break":
        # Current changes while observed power/voltage/PF do not; this is a
        # sensor-relation consistency probe, not a claim about physical failure.
        for sensor in ("currentR", "currentS", "currentT"):
            scale = _scale(scales, module, sensor)
            if not np.isfinite(scale):
                return None
            delta = 4.0 * scale
            _shift(out, sensor, delta)
            details["changes"][sensor] = {"mean_delta": delta, "training_scale": scale, "scale_units": 4.0}
        details["interpretation"] = "Current-versus-power relationship perturbation; could represent measurement inconsistency."
    elif kind == "volatility_increase":
        for sensor in ("activePower", "currentR", "currentS", "currentT"):
            scale = _scale(scales, module, sensor)
            if not np.isfinite(scale):
                return None
            mean = out[f"{sensor}_mean"].to_numpy(dtype=float)
            prior = out[f"{sensor}_std"].to_numpy(dtype=float)
            requested = np.maximum(prior * 3.0, scale * 2.0)
            allowed = mean - margin
            amp = np.minimum(requested, allowed * 0.65)
            if np.any(amp <= prior * 1.25) or np.any(amp <= 0):
                return None
            # Six samples at mean-amp and six at mean+amp have these exact
            # ddof=0 moments. No raw sample trace is emitted or asserted.
            out[f"{sensor}_std"] = amp
            out[f"{sensor}_min"] = mean - amp
            out[f"{sensor}_max"] = mean + amp
            details["changes"][sensor] = {"median_std_before": float(np.median(prior)), "median_std_after": float(np.median(amp)), "training_scale": scale, "lower_bound": 0.0}
    else:
        raise ValueError(f"Unknown synthetic kind: {kind}")
    try:
        assert_aggregate_consistency(out, require_rule_safe=True)
    except ValueError:
        return None
    return out, details


def generate_synthetic_probes(frame: pd.DataFrame, training_scales: Mapping, max_per_type: int = 13,
                              seed: int = 20261016, pre_minutes: int = 60,
                              injection_minutes: int = 30, post_minutes: int = 30,
                              generation_audit: dict | None = None) -> list[dict]:
    """Return paired same-window controls and injected copies, balanced by equipment.

    Input frame is not modified. Every base window is rule-safe and complete.
    All selection and severity choices are frozen before inspecting model scores.
    Availability may yield fewer than requested probes; report actual counts.
    """
    if max_per_type < 1 or min(pre_minutes, injection_minutes, post_minutes) < 1:
        raise ValueError("Probe counts and minute intervals must be positive.")
    rng = np.random.default_rng(seed)
    window = pre_minutes + injection_minutes + post_minutes
    inputs: dict = {}
    required = [f"{sensor}_{stat}" for sensor in SENSORS for stat in ("mean", "std", "min", "max")]
    required += [f"voltage{p}_min" for p in ("RS", "ST", "TR")]
    module_column = "module" if "module" in frame else "module(equipment)"
    for module, source in frame.groupby(module_column, sort=True):
        data = source.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        good = data["n"].eq(12) & data[required].notna().all(axis=1) & ~fixed_rule_flags(data)
        if "complete_minute" in data:
            good &= data["complete_minute"].fillna(False).astype(bool)
        good &= data["timestamp"].eq(data["timestamp"].dt.floor("min"))
        adjacent = data["timestamp"].diff().eq(pd.Timedelta(minutes=1))
        segments = (~(good & good.shift(fill_value=False) & adjacent)).cumsum()
        starts = []
        for _, rows in data[good].groupby(segments[good], sort=False):
            if len(rows) >= window:
                starts.extend(range(int(rows.index[0]), int(rows.index[-1]) - window + 2))
        if starts:
            starts = np.asarray(starts, dtype=np.int64)
            rng.shuffle(starts)
            inputs[module] = (data, starts)
    equipment = list(inputs)
    rng.shuffle(equipment)
    probes = []
    used_windows: list[tuple[object, int, int]] = []
    audit = {"seed": seed, "candidate_limit_per_equipment_kind": 500,
             "model_scores_inspected": False, "thresholds_modified": False,
             "feasibility_revision": "upward voltage / R-phase-only increase / power-current volatility",
             "by_kind": {}}
    for kind in PROBE_KINDS:
        count = 0
        stats = {"requested": max_per_type, "generated": 0, "candidate_attempts": 0,
                 "original_direction_rejected": 0, "original_direction_feasible": 0,
                 "revised_injection_rejected": 0, "invalid_baseline_rejected": 0,
                 "no_eligible_equipment": not bool(equipment), "skipped": 0}
        audit["by_kind"][kind] = stats
        # Exactly one attempt slot per requested probe; no repeated retries over
        # the same 5,000 candidates. A failed slot remains a documented skip.
        for slot in range(max_per_type):
            if not equipment:
                break
            module = equipment[slot % len(equipment)]
            data, starts = inputs[module]
            for pos in starts[:500]:
                pos = int(pos)
                if any(m == module and pos < end and pos + window > begin for m, begin, end in used_windows):
                    continue
                stats["candidate_attempts"] += 1
                baseline = data.iloc[pos:pos + window].copy(deep=True).reset_index(drop=True)
                # Promote paired copies equally; float32->float64 is exact and
                # avoids assignment truncation without modifying source values.
                float_columns = baseline.select_dtypes(include="floating").columns
                baseline = baseline.astype({column: "float64" for column in float_columns})
                try:
                    assert_aggregate_consistency(baseline, require_rule_safe=True)
                except ValueError:
                    stats["invalid_baseline_rejected"] += 1
                    continue
                injection = baseline.iloc[pre_minutes:pre_minutes + injection_minutes]
                original = _inject_original(injection, kind, training_scales, module)
                stats["original_direction_feasible" if original is not None else "original_direction_rejected"] += 1
                changed = _inject(injection, kind, training_scales, module)
                if changed is None:
                    stats["revised_injection_rejected"] += 1
                    continue
                segment, details = changed
                injected = baseline.copy(deep=True)
                for column in segment.columns:
                    injected.loc[pre_minutes:pre_minutes + injection_minutes - 1, column] = segment[column].to_numpy()
                start = baseline["timestamp"].iloc[pre_minutes]
                end = start + pd.Timedelta(minutes=injection_minutes)
                count += 1
                stats["generated"] = count
                probes.append({"probe_id": f"{kind}-{count:03d}", "kind": kind, "module": module,
                               "baseline": baseline, "injected": injected,
                               "truth": {"start": start, "end": end, "injection_minutes": injection_minutes},
                               "severity": details, "notice": TEST_NOTICE, "seed": seed,
                               "original_direction_feasible_on_same_window": original is not None})
                used_windows.append((module, pos, pos + window))
                break
        stats["skipped"] = max_per_type - count
    if generation_audit is not None:
        generation_audit.clear()
        generation_audit.update(audit)

    return probes


def evaluate_synthetic_probe(probe: Mapping, baseline_flags: Sequence[bool], injected_flags: Sequence[bool],
                             min_consecutive: int = 3) -> dict:
    """Only decisions during injected evidence count; delayed post-event alerts do not.

    Boundary convention: start < decision <= end, because a decision at the
    end of the last injected minute uses only injected-period evidence.
    A baseline-overlap control is separately marked, never called a new success.
    """
    start, end = pd.Timestamp(probe["truth"]["start"]), pd.Timestamp(probe["truth"]["end"])
    base_events = group_alert_events(probe["baseline"], baseline_flags, min_consecutive)
    injected_events = group_alert_events(probe["injected"], injected_flags, min_consecutive)
    def during(events):
        return events[(events["decision_time"] > start) & (events["decision_time"] <= end)]
    base_during, injected_during = during(base_events), during(injected_events)
    baseline_overlap = base_events[(base_events["decision_time"] <= end) & (base_events["end"] > start)]
    detected = bool(len(injected_during))
    delay = (injected_during["decision_time"].min() - start).total_seconds() / 60 if detected else None
    return {"probe_id": probe["probe_id"], "kind": probe["kind"], "module": probe["module"],
            "injection_start": start.isoformat(), "injection_end": end.isoformat(),
            "detected_during_injection": detected, "first_decision_delay_minutes": delay,
            "injection_alert_count": len(injected_during), "baseline_injection_alert_count": len(base_during),
            "baseline_already_alerting": bool(len(baseline_overlap)),
            "incremental_detection_on_clean_control": detected and not bool(len(baseline_overlap)),
            "notice": TEST_NOTICE}


def summarize_synthetic_results(results: Sequence[Mapping]) -> dict:
    """Aggregate functionality-test results, with contaminated controls separated."""
    rows = list(results)
    clean = [row for row in rows if not row["baseline_already_alerting"]]
    detected = [row for row in rows if row["detected_during_injection"]]
    incremental = [row for row in clean if row["detected_during_injection"]]
    delays = [row["first_decision_delay_minutes"] for row in detected]
    return {"probe_count": len(rows), "detected_probe_count": len(detected),
            "synthetic_event_recall": len(detected) / len(rows) if rows else None,
            "baseline_already_alerting_count": len(rows) - len(clean),
            "clean_control_count": len(clean), "incremental_detected_count": len(incremental),
            "clean_control_synthetic_recall": len(incremental) / len(clean) if clean else None,
            "median_first_decision_delay_minutes": float(np.median(delays)) if delays else None,
            "by_kind": {kind: summarize_synthetic_results([row for row in rows if row["kind"] == kind])
                        for kind in sorted({row["kind"] for row in rows})} if len({row["kind"] for row in rows}) > 1 else {},
            "notice": TEST_NOTICE}
