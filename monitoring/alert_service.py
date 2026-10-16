"""Accurate alert counts. A marked hour is never treated as an hour-long event.

The portable event cache contains contiguous, verified five-second observations.
When it is unavailable (or rules change), hourly minima can establish only which
hours contain an excursion. They cannot establish its count or duration.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "ml/outputs/agent_data"
HOURLY_PATH = CACHE / "dashboard_hourly.parquet"
EVENTS_PATH = CACHE / "alert_events.parquet"
META_PATH = CACHE / "alert_events.json"
DEFAULT_CUTOFF = "2025-04-30 00:00:00"
SAMPLE_SECONDS = 5
MODULE = "module(equipment)"
DEFAULT_THRESHOLDS = {"phase_voltage": 210.0, "line_voltage": 360.0, "power_factor": 85.0}
EQUIPMENT_LIST = [
    ("1", "1 (PM-3)"), ("2", "2 (L-1전등)"), ("3", "3 (분쇄기(2))"),
    ("4", "4 (분쇄기(1))"), ("5", "5 (좌측분전반)"), ("11", "11 (우측분전반1)"),
    ("12", "12 (4호기)"), ("13", "13 (3호기)"), ("14", "14 (2호기)"),
    ("15", "15 (예비건조기)"), ("16", "16 (호이스트)"), ("17", "17 (6호기)"),
    ("18", "18 (우측분전반2)"),
]
NAMES = dict(EQUIPMENT_LIST)
ITEMS = {
    "phase_voltage_R": ("voltageR", "상전압 R", "V", "phase_voltage"),
    "phase_voltage_S": ("voltageS", "상전압 S", "V", "phase_voltage"),
    "phase_voltage_T": ("voltageT", "상전압 T", "V", "phase_voltage"),
    "line_voltage_RS": ("voltageRS", "선간전압 RS", "V", "line_voltage"),
    "line_voltage_ST": ("voltageST", "선간전압 ST", "V", "line_voltage"),
    "line_voltage_TR": ("voltageTR", "선간전압 TR", "V", "line_voltage"),
    "power_factor_R": ("powerFactorR", "역률 R", "%", "power_factor"),
    "power_factor_S": ("powerFactorS", "역률 S", "%", "power_factor"),
    "power_factor_T": ("powerFactorT", "역률 T", "%", "power_factor"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_equipment(value: Any) -> str | None:
    if value is None or str(value).strip().lower() in ("", "전체", "공장 전체", "all"):
        return None
    token = re.sub(r"[\s()（）-]", "", str(value)).lower()
    for channel, name in EQUIPMENT_LIST:
        label = name.split("(", 1)[1].rsplit(")", 1)[0]
        if token in (channel, re.sub(r"[\s()（）-]", "", name).lower(),
                     re.sub(r"[\s()（）-]", "", label).lower()):
            return name
    raise ValueError(f"등록되지 않은 설비입니다: {value}")


def normalize_thresholds(values: dict | None = None) -> dict:
    out = dict(DEFAULT_THRESHOLDS)
    for key, value in (values or {}).items():
        if key not in out:
            raise ValueError(f"지원하지 않는 기준 항목입니다: {key}")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("기준값은 0보다 큰 유한한 수여야 합니다.")
        out[key] = number
    return out


def _stamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("날짜가 올바르지 않습니다.")
    return stamp.tz_convert("Asia/Seoul").tz_localize(None) if stamp.tzinfo else stamp


def _iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def build_events(frame: pd.DataFrame, thresholds: dict | None = None,
                 ai_thresholds: dict | None = None) -> list[dict]:
    """Group flags only when adjacent observations are exactly five seconds apart.

    This function does not infer new AI flags or alter the trained model. Normal
    observations, missing samples and day boundaries separate intervals.
    End timestamps are exclusive: a single 12:00:00 sample ends at 12:00:05.
    """
    thresholds = normalize_thresholds(thresholds)
    ai_thresholds = ai_thresholds or {}
    required = {MODULE, "datetime", "ai_flag", "ai_score", *(v[0] for v in ITEMS.values())}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"5초 캐시에 필요한 컬럼이 없습니다: {sorted(missing)}")
    rows = []
    for module, source in frame.groupby(MODULE, sort=False):
        name = normalize_equipment(module)
        g = source.sort_values("datetime").reset_index(drop=True)
        times = pd.to_datetime(g["datetime"])
        if times.isna().any() or times.duplicated().any():
            raise ValueError("설비별 시간이 중복되거나 누락된 캐시는 사용할 수 없습니다.")
        if g["ai_flag"].isna().any() or not g["ai_flag"].isin([True, False, 0, 1]).all():
            raise ValueError("AI 판정값이 올바르지 않습니다.")
        specs = [(key, col, label, unit, thresholds[fam], "rule")
                 for key, (col, label, unit, fam) in ITEMS.items()]
        specs.append(("ai", "ai_score", "AI 패턴 탐지", "anomaly_score",
                      ai_thresholds.get(str(module), ai_thresholds.get(name)), "ai"))
        for key, col, label, unit, threshold, kind in specs:
            values = pd.to_numeric(g[col], errors="coerce")
            flags = g["ai_flag"].astype(bool) if kind == "ai" else values.lt(threshold)
            if bool((flags & ~np.isfinite(values)).any()):
                raise ValueError("탐지된 기록의 측정값이 유효하지 않습니다.")
            positions = np.flatnonzero(flags.to_numpy())
            if not len(positions):
                continue
            # Position adjacency prevents joining across a normal sample; time
            # adjacency prevents joining over missing observations.
            selected_times = times.iloc[positions]
            breaks = np.ones(len(positions), dtype=bool)
            if len(positions) > 1:
                breaks[1:] = ((np.diff(positions) != 1)
                              | (selected_times.diff().iloc[1:].dt.total_seconds().to_numpy() != SAMPLE_SECONDS)
                              | (selected_times.dt.normalize().diff().iloc[1:].dt.total_seconds().to_numpy() != 0))
            for group in np.split(positions, np.flatnonzero(breaks)[1:]):
                vals = values.iloc[group]
                at = vals.idxmax() if kind == "ai" else vals.idxmin()
                value = float(values.iloc[at])
                count = int(len(group))
                start, last = times.iloc[group[0]], times.iloc[group[-1]]
                rows.append({
                    "date": start.strftime("%Y-%m-%d"), "kind": kind,
                    "equipment": name, "item": label, "item_key": key,
                    "timestamp": _iso(start), "end": _iso(last + pd.Timedelta(seconds=SAMPLE_SECONDS)),
                    "extremum_timestamp": _iso(times.iloc[at]), "value": value, "unit": unit,
                    "threshold": float(threshold) if threshold is not None else None,
                    "deviation_pct": (value - threshold) / threshold * 100 if kind == "rule" else None,
                    "flagged_records": count, "duration_seconds": count * SAMPLE_SECONDS,
                    "duration_min": count * SAMPLE_SECONDS / 60,
                    "sample_duration_seconds": count * SAMPLE_SECONDS,
                    "affected_hours": int((last.floor("h") - start.floor("h")).total_seconds() // 3600) + 1,
                    "basis": "5s_exact",
                })
    return rows


def _hourly_events(frame: pd.DataFrame, thresholds: dict, kind: str,
                   ai_thresholds: dict | None = None) -> list[dict]:
    rows = []
    for name, g in frame.groupby("equipment", sort=False):
        for _, row in g.iterrows():
            specs = [(key, col, label, unit, thresholds[fam])
                     for key, (col, label, unit, fam) in ITEMS.items()]
            if kind == "ai":
                if not int(row.get("ai_flags", 0)):
                    continue
                specs = [("ai", "ai_score_max", "AI 패턴 탐지", "anomaly_score",
                          (ai_thresholds or {}).get(name))]
            for key, col, label, unit, threshold in specs:
                value = row.get(col)
                if pd.isna(value) or (kind == "rule" and value >= threshold):
                    continue
                count = int(row["ai_flags"]) if kind == "ai" else None
                at = pd.Timestamp(row["datetime"])
                rows.append({
                    "date": at.strftime("%Y-%m-%d"), "kind": kind, "equipment": name,
                    "item": label, "item_key": key, "timestamp": _iso(at),
                    "end": _iso(at + pd.Timedelta(hours=1)), "value": float(value),
                    "unit": unit, "threshold": threshold, "extremum_timestamp": None,
                    "min_at_hour": at.strftime("%H시"),
                    "deviation_pct": (float(value) - threshold) / threshold * 100 if kind == "rule" else None,
                    "flagged_records": count, "duration_seconds": None, "duration_min": None,
                    "sample_duration_seconds": count * SAMPLE_SECONDS if count is not None else None,
                    "affected_hours": 1, "basis": "hourly",
                    "note": "이상이 관측된 시간대이며 실제 시작·종료·연속 지속 시간은 알 수 없습니다.",
                })
    return rows


def _metrics(events: list[dict], basis: str, available: bool) -> dict:
    keys = ["equipment_count", "event_count", "affected_hours", "flagged_records",
            "sample_duration_seconds", "longest_duration_seconds"]
    if not available:
        return {**dict.fromkeys(keys), "basis": basis, "data_available": False}
    hours = set()
    for e in events:
        # Integer hour bins avoid creating thousands of tiny DatetimeIndexes.
        hour_ns = 3_600_000_000_000
        first = pd.Timestamp(e["timestamp"]).value // hour_ns
        last = (pd.Timestamp(e["end"]).value - 1) // hour_ns
        hours.update((e["equipment"], hour) for hour in range(first, last + 1))
    counts_known = all(e.get("flagged_records") is not None for e in events)
    count = sum(e["flagged_records"] for e in events) if counts_known else None
    return {"equipment_count": len({e["equipment"] for e in events}),
            "event_count": len(events) if basis == "5s_exact" else None,
            "affected_hours": len(hours), "flagged_records": count,
            "sample_duration_seconds": count * SAMPLE_SECONDS if count is not None else None,
            "longest_duration_seconds": max((e["duration_seconds"] for e in events), default=0)
            if basis == "5s_exact" else None,
            "basis": basis, "data_available": True}


def summarize_alerts(day: Any, hourly: pd.DataFrame, *, events: pd.DataFrame | None = None,
                     event_metadata: dict | None = None, thresholds: dict | None = None,
                     equipment: str | None = None, cutoff: Any = None) -> dict:
    """Pure aggregation entrypoint, also useful for independent tests."""
    d = _stamp(day).normalize()
    end = _stamp(cutoff or DEFAULT_CUTOFF).floor("h")
    th = normalize_thresholds(thresholds)
    name = normalize_equipment(equipment)
    h = hourly.copy()
    if len(h):
        h["datetime"] = pd.to_datetime(h["datetime"])
        if "equipment" not in h:
            h["equipment"] = h[MODULE].astype(str).str.split("(", n=1).str[0].str.strip().map(NAMES)
        h = h[h["equipment"].notna() & h["datetime"].dt.normalize().eq(d)
              & ((h["datetime"] + pd.Timedelta(hours=1)) <= end)]
        if name:
            h = h[h["equipment"].eq(name)]
    available = not h.empty
    meta = event_metadata or {}
    exact = (available and events is not None and bool(meta.get("complete"))
             and d.strftime("%Y-%m-%d") in meta.get("dates", {}))
    rule_basis = "5s_exact" if exact and th == meta.get("rule_thresholds") else "hourly"
    ai_basis = "5s_exact" if exact else "hourly"
    exact_rows = []
    if exact and len(events):
        subset = events[events["date"].astype(str).eq(d.strftime("%Y-%m-%d"))].copy()
        if name:
            subset = subset[subset["equipment"].eq(name)]
        # Only completed hours are exposed for historical cutoff consistency.
        for row in subset.to_dict("records"):
            start, finish = pd.Timestamp(row["timestamp"]), pd.Timestamp(row["end"])
            if start >= end:
                continue
            if finish > end:
                seconds = int((end - start).total_seconds())
                row.update(end=_iso(end), flagged_records=seconds // SAMPLE_SECONDS,
                           duration_seconds=seconds, duration_min=seconds / 60,
                           sample_duration_seconds=seconds, value=None, extremum_timestamp=None,
                           deviation_pct=None)
            for key, value in list(row.items()):
                if isinstance(value, float) and not math.isfinite(value):
                    row[key] = None
            exact_rows.append(row)
    ai_thresholds = {normalize_equipment(k): v for k, v in (meta.get("ai_thresholds") or {}).items()}
    rule = ([e for e in exact_rows if e["kind"] == "rule"] if rule_basis == "5s_exact"
            else _hourly_events(h, th, "rule", ai_thresholds) if available else [])
    ai = ([e for e in exact_rows if e["kind"] == "ai"] if ai_basis == "5s_exact"
          else _hourly_events(h, th, "ai", ai_thresholds) if available else [])
    rule.sort(key=lambda e: (-float(e.get("duration_seconds") or 0), e["timestamp"], e["equipment"]))
    ai.sort(key=lambda e: (-int(e.get("flagged_records") or 0), e["timestamp"], e["equipment"]))
    equipment_rows = []
    for _, label in EQUIPMENT_LIST:
        if name and label != name:
            continue
        group = h[h["equipment"].eq(label)] if available else h
        equipment_rows.append({"equipment": label, "data_available": not group.empty,
                               "observed_records": int(group["n"].sum()) if len(group) and "n" in group else None,
                               "observed_hours": len(group),
                               "rule": _metrics([e for e in rule if e["equipment"] == label], rule_basis, not group.empty),
                               "ai": _metrics([e for e in ai if e["equipment"] == label], ai_basis, not group.empty)})
    note = ("발생 구간은 설비·항목별 연속된 5초 기록을 묶은 수이며 고장 사건 수가 아닙니다. "
            "누적 시간은 탐지 표본 수×5초로 계산하며 연속 지속 시간과 다릅니다. "
            "설정 기준의 기록 수·누적 시간은 센서 항목별 합계라 같은 시각이 중복될 수 있습니다. "
            "관측 시간대 수는 설비별 시간대의 합계이며 지속 시간이 아닙니다. "
            "AI 판정과 설정 기준 이탈은 독립적이며 실제 고장을 확정하지 않습니다.")
    if rule_basis == "hourly" or ai_basis == "hourly":
        note += " 시간별 요약만 있는 결과는 실제 발생 구간 수와 연속 지속 시간을 알 수 없어 표시하지 않습니다."
    if not available:
        note = "선택한 날짜·설비에 기준 시점 이전의 실측 기록이 없습니다. 자료 없음은 정상이나 0건을 뜻하지 않습니다."
    return {"date": d.strftime("%Y-%m-%d"), "cutoff": _iso(end), "equipment": name,
            "data_available": available, "basis": rule_basis if rule_basis == ai_basis else "mixed",
            "rule": _metrics(rule, rule_basis, available), "ai": _metrics(ai, ai_basis, available),
            "equipment_rows": equipment_rows, "rule_events": rule, "ai_events": ai,
            "observed_records": int(h["n"].sum()) if available and "n" in h else None,
            "observed_hours": len(h), "thresholds": th, "note": note}


def _file_key(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size
    except OSError:
        return str(path), 0, 0


@lru_cache(maxsize=2)
def _load_hourly(key: tuple) -> pd.DataFrame:
    return pd.read_parquet(key[0]) if key[1] else pd.DataFrame()


@lru_cache(maxsize=2)
def _load_exact(event_key: tuple, meta_key: tuple, hourly_key: tuple) -> tuple[pd.DataFrame | None, dict, str | None]:
    if not event_key[1] or not meta_key[1]:
        return None, {}, "정확한 5초 경보 캐시가 없어 시간별 요약을 사용합니다."
    try:
        meta = json.loads(Path(meta_key[0]).read_text(encoding="utf-8-sig"))
        if (not meta.get("complete") or meta.get("format_version") != 1
                or meta.get("sample_seconds") != SAMPLE_SECONDS
                or meta.get("events_sha256") != sha256(Path(event_key[0]))
                or meta.get("hourly_sha256") != sha256(Path(hourly_key[0]))):
            raise ValueError("경보 캐시 식별 정보가 현재 시간별 데이터와 일치하지 않습니다.")
        events = pd.read_parquet(event_key[0])
        if len(events) != meta.get("event_rows"):
            raise ValueError("경보 캐시 행 수가 일치하지 않습니다.")
        return events, meta, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, {}, f"5초 경보 캐시 검증 실패: {error}"


def get_alert_summary(day: Any, thresholds: dict | None = None, equipment: str | None = None,
                      cutoff: Any = None) -> dict:
    hourly_key = _file_key(HOURLY_PATH)
    hourly = _load_hourly(hourly_key)
    events, metadata, error = _load_exact(_file_key(EVENTS_PATH), _file_key(META_PATH), hourly_key)
    result = summarize_alerts(day, hourly, events=events, event_metadata=metadata,
                              thresholds=thresholds, equipment=equipment, cutoff=cutoff)
    if error and result["data_available"]:
        result["cache_note"] = error
    return result
