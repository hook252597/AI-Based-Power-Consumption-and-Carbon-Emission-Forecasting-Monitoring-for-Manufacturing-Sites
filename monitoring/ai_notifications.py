"""Separate immutable model detections from an experimental notification policy.

Scores are model scores, not failure probabilities. No inference, model threshold
change or rule warning filtering happens here. Every decision uses only samples
whose five-second observation interval has finished by that decision time.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from monitoring.alert_service import (CACHE, EQUIPMENT_LIST, MODULE, SAMPLE_SECONDS,
    normalize_equipment, sha256)

POLICY_VERSION = "ai-notification-preview-v1"
JSON_HASH_METHOD = "canonical-json-sort-keys-compact-utf8-v1"
DEFAULT_POLICY = {"score_margin": 0.02, "duration_seconds": 60,
    "repeat_count": 3, "repeat_window_minutes": 30, "merge_window_minutes": 10}
SAMPLE_COLUMNS = ["timestamp", "equipment", "score", "base_threshold",
                  "original_event_id", "coverage_id"]


def json_content_sha256(path: Path) -> str:
    """Hash JSON values independent of Git checkout newlines, BOM or layout."""
    content = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def normalize_policy(policy: dict | None = None) -> dict:
    out = dict(DEFAULT_POLICY)
    for key, value in (policy or {}).items():
        if key not in out:
            raise ValueError(f"지원하지 않는 알림 설정: {key}")
        if isinstance(value, bool):
            raise ValueError("알림 설정에는 숫자를 입력하세요.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("알림 설정은 유한한 숫자여야 합니다.")
        out[key] = number
    if not 0 <= out["score_margin"] <= 1:
        raise ValueError("점수 여유는 0~1 범위여야 합니다.")
    for key in ("duration_seconds", "repeat_count", "repeat_window_minutes", "merge_window_minutes"):
        number = out[key]
        if number != int(number):
            raise ValueError(f"{key}에는 정수를 입력하세요.")
        out[key] = int(number)
    if out["duration_seconds"] < SAMPLE_SECONDS or out["duration_seconds"] % SAMPLE_SECONDS:
        raise ValueError("연속 시간은 5초 이상의 5초 배수여야 합니다.")
    if out["repeat_count"] < 2 or out["repeat_window_minutes"] < 1 or out["merge_window_minutes"] < 0:
        raise ValueError("반복은 2회 이상, 반복 시간창은 1분 이상, 묶음 시간은 0분 이상이어야 합니다.")
    return out


def _stamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("날짜가 올바르지 않습니다.")
    return stamp.tz_convert("Asia/Seoul").tz_localize(None) if stamp.tzinfo else stamp


def _id(prefix: str, *parts: Any) -> str:
    text = "|".join(str(p) for p in parts)
    return prefix + "_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def build_detection_samples(frame: pd.DataFrame, ai_thresholds: dict) -> pd.DataFrame:
    """Retain every existing AI-positive sample and raw observation coverage IDs.

    Coverage IDs come from ALL observations, including normal ones. Therefore a
    gap in the original source can never masquerade as continuous repetition.
    AI event IDs are formed BEFORE the optional score margin is applied.
    """
    required = {MODULE, "datetime", "ai_flag", "ai_score"}
    if required - set(frame):
        raise ValueError("5초 AI 캐시에 필요한 컬럼이 없습니다.")
    tables = []
    for module, source in frame.groupby(MODULE, sort=False):
        name = normalize_equipment(module)
        threshold = ai_thresholds.get(str(module), ai_thresholds.get(name))
        if threshold is None or not math.isfinite(float(threshold)):
            raise ValueError(f"{module}: 기존 AI 기준점수가 없습니다.")
        g = source.sort_values("datetime").reset_index(drop=True)
        times = pd.to_datetime(g["datetime"])
        if times.isna().any() or times.duplicated().any():
            raise ValueError("설비별 시각이 중복되거나 올바르지 않습니다.")
        if not g["ai_flag"].isin([True, False, 0, 1]).all():
            raise ValueError("기존 AI 판정이 올바르지 않습니다.")
        flags = g["ai_flag"].astype(bool).to_numpy()
        scores = pd.to_numeric(g["ai_score"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(scores[flags]).all():
            raise ValueError("AI 탐지 점수가 유효하지 않습니다.")
        if len(g) == 0:
            continue
        coverage_break = times.diff().dt.total_seconds().ne(SAMPLE_SECONDS) | times.dt.normalize().ne(times.dt.normalize().shift())
        coverage_groups = coverage_break.cumsum().to_numpy()
        coverage_starts = times[coverage_break].to_dict()
        coverage_map = {int(coverage_groups[pos]): _id("coverage", name, stamp.isoformat())
                        for pos, stamp in coverage_starts.items()}
        positions = np.flatnonzero(flags)
        if not len(positions):
            continue
        breaks = np.ones(len(positions), dtype=bool)
        if len(positions) > 1:
            breaks[1:] = ((np.diff(positions) != 1) |
                           (coverage_groups[positions[1:]] != coverage_groups[positions[:-1]]))
        event_groups = np.cumsum(breaks)
        event_map = {int(event_groups[i]): _id("ai", name, times.iloc[pos].isoformat())
                     for i, pos in enumerate(positions) if breaks[i]}
        tables.append(pd.DataFrame({"timestamp": times.iloc[positions].to_numpy(),
            "equipment": name, "score": scores[positions], "base_threshold": float(threshold),
            "original_event_id": [event_map[int(i)] for i in event_groups],
            "coverage_id": [coverage_map[int(i)] for i in coverage_groups[positions]]}))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=SAMPLE_COLUMNS)


def _validate_samples(samples: pd.DataFrame) -> pd.DataFrame:
    if set(SAMPLE_COLUMNS) - set(samples):
        raise ValueError("AI 알림 캐시 컬럼이 올바르지 않습니다.")
    out = samples.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if out["timestamp"].isna().any() or out.duplicated(["equipment", "timestamp"]).any():
        raise ValueError("AI 알림 시각이 누락되거나 중복되었습니다.")
    if not np.isfinite(out[["score", "base_threshold"]].to_numpy(dtype=float)).all():
        raise ValueError("AI 알림 점수가 유효하지 않습니다.")
    if out[["equipment", "original_event_id", "coverage_id"]].isna().any().any():
        raise ValueError("AI 알림 식별자가 누락되었습니다.")
    check = out.groupby("original_event_id", sort=False).agg(
        equipment_n=("equipment", "nunique"), coverage_n=("coverage_id", "nunique"),
        threshold_n=("base_threshold", "nunique"))
    if len(check) and not check.eq(1).all().all():
        raise ValueError("원본 AI 구간의 설비·관측 구간·기준이 일치하지 않습니다.")
    out = out.sort_values(["equipment", "timestamp"]).reset_index(drop=True)
    gaps = out.groupby("original_event_id", sort=False)["timestamp"].diff().dropna()
    if len(gaps) and not gaps.eq(pd.Timedelta(seconds=SAMPLE_SECONDS)).all():
        raise ValueError("원본 AI 구간 안에 관측 누락이 있습니다.")
    days = out["timestamp"].dt.normalize().groupby(out["original_event_id"]).nunique()
    if len(days) and not days.eq(1).all():
        raise ValueError("원본 AI 구간이 날짜 경계를 넘습니다.")
    return out


def evaluate_notifications(samples: pd.DataFrame, policy: dict | None = None,
                           cutoff: Any = None) -> dict:
    """Pure retrospective replay. Filtering at cutoff happens before decisions.

    A repeat occurrence is the FIRST qualifying sample of a DISTINCT original AI
    interval. Repeated score crossings within one interval count only once.
    The rolling repetition window includes the point exactly on its boundary.
    A new alert group begins after the full quiet gap since the LAST candidate.
    """
    settings = normalize_policy(policy)
    data = _validate_samples(samples)
    if cutoff is not None:
        end = _stamp(cutoff)
        data = data[data["timestamp"] + pd.Timedelta(seconds=SAMPLE_SECONDS) <= end].copy()
    data["qualified"] = data["score"] >= data["base_threshold"] + settings["score_margin"]
    logs = {}
    if len(data):
        grouped = data.groupby("original_event_id", sort=False).agg(
            equipment=("equipment", "first"), start=("timestamp", "min"),
            end=("timestamp", "max"), score_max=("score", "max"), peak_at=("score", "idxmax"),
            base_threshold=("base_threshold", "first"), ai_records=("score", "size"),
            qualified_records=("qualified", "sum"))
        for row in grouped.itertuples():
            logs[row.Index] = {"event_id": row.Index, "equipment": row.equipment,
                "date": row.start.strftime("%Y-%m-%d"), "start": row.start.isoformat(),
                "end": (row.end + pd.Timedelta(seconds=SAMPLE_SECONDS)).isoformat(),
                "score_max": float(row.score_max), "score_max_at": data.loc[row.peak_at, "timestamp"].isoformat(),
                "base_threshold": float(row.base_threshold), "ai_records": int(row.ai_records),
                "duration_seconds": int(row.ai_records * SAMPLE_SECONDS),
                "qualified_records": int(row.qualified_records), "max_qualifying_duration_seconds": 0,
                "notification_candidate": False, "log_only_reason": None}
    candidates = []
    for equipment, machine in data.groupby("equipment", sort=False):
        for (_, _), g in machine.groupby(["coverage_id", machine["timestamp"].dt.normalize()], sort=False):
            recent = deque()
            seen_events = set()
            last_qualified_time = None
            last_event = None
            run_start = None
            run_records = 0
            for row in g.itertuples(index=False):
                if not row.qualified:
                    last_qualified_time = None
                    run_records = 0
                    continue
                t = row.timestamp
                decision_time = t + pd.Timedelta(seconds=SAMPLE_SECONDS)
                if (last_qualified_time is None or t - last_qualified_time != pd.Timedelta(seconds=SAMPLE_SECONDS)
                        or row.original_event_id != last_event):
                    run_start, run_records = t, 0
                run_records += 1
                seconds = run_records * SAMPLE_SECONDS
                logs[row.original_event_id]["max_qualifying_duration_seconds"] = max(
                    logs[row.original_event_id]["max_qualifying_duration_seconds"], seconds)
                last_qualified_time, last_event = t, row.original_event_id
                reasons, supporting = [], [row.original_event_id]
                if seconds == settings["duration_seconds"]:
                    reasons.append("sustained")
                is_new = row.original_event_id not in seen_events
                if is_new:
                    seen_events.add(row.original_event_id)
                    oldest = decision_time - pd.Timedelta(minutes=settings["repeat_window_minutes"])
                    while recent and recent[0][0] < oldest:
                        recent.popleft()
                    recent.append((decision_time, row.original_event_id))
                    if len(recent) >= settings["repeat_count"]:
                        reasons.append("repeated")
                        supporting = [event_id for _, event_id in recent]
                if reasons:
                    candidates.append({"candidate_id": _id("candidate", equipment, decision_time.isoformat(), *reasons),
                        "event_id": row.original_event_id, "equipment": equipment,
                        "coverage_id": row.coverage_id, "date": t.strftime("%Y-%m-%d"),
                        "trigger_at": decision_time.isoformat(), "reasons": reasons,
                        "run_start": run_start.isoformat(), "duration_seconds": seconds,
                        "repeat_event_count": len(recent) if is_new else None,
                        "score_at_trigger": float(row.score), "base_threshold": float(row.base_threshold),
                        "score_cutoff": float(row.base_threshold + settings["score_margin"]),
                        "supporting_event_ids": supporting})
    candidates.sort(key=lambda c: (c["trigger_at"], c["equipment"]))
    notifications, latest = [], {}
    for candidate in candidates:
        key = (candidate["equipment"], candidate["coverage_id"], candidate["date"])
        t = pd.Timestamp(candidate["trigger_at"])
        group = latest.get(key)
        if (group is None or settings["merge_window_minutes"] == 0 or
                t >= pd.Timestamp(group["last_trigger_at"]) + pd.Timedelta(minutes=settings["merge_window_minutes"])):
            group = {"notification_id": _id("notification", candidate["equipment"], candidate["trigger_at"]),
                "equipment": candidate["equipment"], "first_trigger_at": candidate["trigger_at"],
                "last_trigger_at": candidate["trigger_at"], "first_score_at_trigger": candidate["score_at_trigger"], "candidate_count": 0,
                "reasons": [], "source_event_ids": [], "candidate_ids": [], "score_max_at_triggers": 0.0}
            notifications.append(group)
            latest[key] = group
        group["last_trigger_at"] = candidate["trigger_at"]
        group["candidate_count"] += 1
        group["candidate_ids"].append(candidate["candidate_id"])
        group["reasons"] = sorted(set(group["reasons"] + candidate["reasons"]))
        group["source_event_ids"] = sorted(set(group["source_event_ids"] + candidate["supporting_event_ids"]))
        group["score_max_at_triggers"] = max(group["score_max_at_triggers"], candidate["score_at_trigger"])
        for event_id in candidate["supporting_event_ids"]:
            logs[event_id]["notification_candidate"] = True
    all_logs = sorted(logs.values(), key=lambda row: (row["start"], row["equipment"]))
    for row in all_logs:
        if not row["notification_candidate"]:
            row["log_only_reason"] = ("below_score_margin" if row["qualified_records"] == 0
                                      else "insufficient_duration_and_repetition")
    def summarize(rows, alerts, triggers):
        return {"original_event_count": len(rows),
            "notified_original_event_count": sum(bool(r["notification_candidate"]) for r in rows),
            "log_only_event_count": sum(not r["notification_candidate"] for r in rows),
            "notification_count": len(alerts), "candidate_count": len(triggers),
            "ai_record_count": sum(r["ai_records"] for r in rows),
            "qualified_record_count": sum(r["qualified_records"] for r in rows),
            "sustained_candidate_count": sum("sustained" in r["reasons"] for r in triggers),
            "repeated_candidate_count": sum("repeated" in r["reasons"] for r in triggers)}
    names = sorted(set(data["equipment"]))
    return {"policy": settings, "policy_version": POLICY_VERSION, "all_logs": all_logs,
        "candidate_events": [r for r in all_logs if r["notification_candidate"]],
        "trigger_candidates": candidates, "log_only": [r for r in all_logs if not r["notification_candidate"]],
        "notifications": notifications, "summary": summarize(all_logs, notifications, candidates),
        "per_equipment": [{"equipment": name, **summarize([r for r in all_logs if r["equipment"] == name],
            [r for r in notifications if r["equipment"] == name], [r for r in candidates if r["equipment"] == name])}
            for name in names]}


def _key(path: Path) -> tuple:
    try:
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns
    except OSError:
        return str(path), 0, 0


@lru_cache(maxsize=2)
def _load(sample_key: tuple, meta_key: tuple, hourly_key: tuple, events_key: tuple, hourly_meta_key: tuple, event_meta_key: tuple) -> tuple:
    if not all(key[2] for key in (sample_key, meta_key, hourly_key, events_key, hourly_meta_key, event_meta_key)):
        return None, {}, "검증된 AI 알림 미리보기 캐시가 없습니다."
    try:
        meta = json.loads(Path(meta_key[0]).read_text(encoding="utf-8-sig"))
        if (not meta.get("complete") or meta.get("format_version") != 1 or meta.get("sample_seconds") != SAMPLE_SECONDS
                or meta.get("samples_sha256") != sha256(Path(sample_key[0]))
                or meta.get("hourly_sha256") != sha256(Path(hourly_key[0]))
                or meta.get("events_sha256") != sha256(Path(events_key[0]))):
            raise ValueError("현재 원본 탐지·시간별 요약과 AI 알림 캐시가 일치하지 않습니다.")
        hourly_meta = json.loads(Path(hourly_meta_key[0]).read_text(encoding="utf-8-sig"))
        event_meta = json.loads(Path(event_meta_key[0]).read_text(encoding="utf-8-sig"))
        if (meta.get("json_metadata_hash_method") != JSON_HASH_METHOD
                or meta.get("hourly_metadata_sha256") != json_content_sha256(Path(hourly_meta_key[0]))
                or meta.get("event_metadata_sha256") != json_content_sha256(Path(event_meta_key[0]))
                or not hourly_meta.get("complete") or not event_meta.get("complete")
                or meta.get("ai_thresholds") != hourly_meta.get("ai_thresholds")
                or meta.get("ai_thresholds") != event_meta.get("ai_thresholds")
                or meta.get("source_manifest_sha256") != event_meta.get("source_manifest_sha256")
                or meta.get("source_model_sha256") != event_meta.get("source_model_sha256")):
            raise ValueError("상위 AI 캐시의 완료·모델·기준 정보가 달라졌습니다.")
        samples = _validate_samples(pd.read_parquet(sample_key[0]))
        if len(samples) != meta.get("sample_rows"):
            raise ValueError("AI 알림 표본 수가 검증 정보와 다릅니다.")
        return samples, meta, None
    except (OSError, ValueError, TypeError, KeyError) as error:
        return None, {}, f"AI 알림 캐시 검증 실패: {error}"


def get_ai_notifications(day: Any = None, equipment: str | None = None,
                         policy: dict | None = None, cutoff: Any = None,
                         cache_dir: Path | None = None) -> dict:
    settings = normalize_policy(policy)
    name = normalize_equipment(equipment)
    cache = Path(cache_dir or CACHE)
    samples, meta, error = _load(_key(cache / "ai_notification_samples.parquet"),
        _key(cache / "ai_notification_samples.json"), _key(cache / "dashboard_hourly.parquet"),
        _key(cache / "alert_events.parquet"), _key(cache / "dashboard_hourly.json"),
        _key(cache / "alert_events.json"))
    unavailable = {"status": "unavailable", "data_available": False, "all_logs": [],
        "candidate_events": [], "trigger_candidates": [], "log_only": [], "notifications": [], "summary": {},
        "per_equipment": [], "coverage": {}, "policy": settings, "policy_version": POLICY_VERSION}
    if error:
        return {**unavailable, "note": error}
    dates = sorted(meta.get("dates", {}))
    if day is not None:
        selected = _stamp(day).strftime("%Y-%m-%d")
        dates = [selected] if selected in dates else []
    if cutoff is not None:
        stop = _stamp(cutoff)
        dates = [d for d in dates if _stamp(d) < stop]
    if not dates:
        return {**unavailable, "note": "선택 기간에 검증된 5초 실측 자료가 없습니다. 자료 없음은 정상이나 0건을 뜻하지 않습니다."}
    if cutoff is not None:
        completed = []
        for d in dates:
            first = meta["dates"][d].get("first_observation_at", d)
            if _stamp(first) + pd.Timedelta(seconds=SAMPLE_SECONDS) <= stop:
                completed.append(d)
        dates = completed
        if not dates:
            return {**unavailable, "note": "기준 시점까지 관측이 끝난 5초 실측 자료가 없습니다."}
    source = samples[samples["timestamp"].dt.strftime("%Y-%m-%d").isin(dates)]
    if name:
        source = source[source["equipment"].eq(name)]
    available_names = set()
    for d in dates:
        available_names.update(normalize_equipment(k) for k in meta["dates"][d].get("per_equipment", {}))
    if name and name not in available_names:
        return {**unavailable, "note": "선택 설비의 검증된 실측 자료가 없습니다."}
    result = evaluate_notifications(source, settings, cutoff=cutoff)
    known = {row["equipment"]: row for row in result["per_equipment"]}
    empty_summary = dict.fromkeys(result["summary"], 0)
    result["per_equipment"] = [{"equipment": label,
        **known.get(label, empty_summary if label in available_names else dict.fromkeys(empty_summary)),
        "data_available": label in available_names} for _, label in EQUIPMENT_LIST if not name or label == name]
    result.update(status="ok", data_available=True,
        coverage={"dates": dates, "start": dates[0], "end": dates[-1], "day_count": len(dates),
                  "equipment_count": 1 if name else len(available_names), "sample_seconds": SAMPLE_SECONDS,
                  "source_ai_records": meta.get("sample_rows"), "source_sha256": meta.get("source_sha256")},
        note=("실험용 알림 미리보기이며 알림을 실제 발송하지 않습니다. 기존 AI 탐지와 모델 기준점수는 그대로 보존합니다. "
              "점수는 고장 확률이 아닙니다. 점수 조건을 만족한 연속 시간 또는 서로 다른 원본 AI 구간의 반복으로 후보를 만듭니다. "
              "반복은 각 원본 구간의 첫 점수 조건 충족 기록을 1회로 세며 누락·날짜 경계에서 다시 셉니다. "
              "표본 종료 후 판단하고, 마지막 후보 이후 묶음 시간만큼 조용해야 새 알림을 만듭니다. 묶음 시간 범위는 연속 이상 지속 시간이 아닙니다. 설정 기준 경고와 독립적입니다."))
    return result
