from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitoring import alert_service as alerts
from tools.prepare_alert_events import _check_day


def raw(times, flags=None, equipment="1(PM-3)"):
    frame = pd.DataFrame({alerts.MODULE: equipment, "datetime": pd.to_datetime(times)})
    for column, _, _, family in alerts.ITEMS.values():
        frame[column] = 220.0 if family == "phase_voltage" else 380.0 if family == "line_voltage" else 90.0
    frame["ai_flag"] = flags or [False] * len(frame)
    frame["ai_score"] = [0.8 if x else 0.2 for x in frame["ai_flag"]]
    return frame


def hourlies(frame):
    frame = frame.copy()
    frame["hour"] = frame["datetime"].dt.floor("h")
    agg = {"n": ("ai_flag", "size"), "ai_flags": ("ai_flag", "sum"), "ai_score_max": ("ai_score", "max")}
    agg.update({column: (column, "min") for column, *_ in alerts.ITEMS.values()})
    return frame.groupby([alerts.MODULE, "hour"]).agg(**agg).reset_index().rename(columns={"hour": "datetime"})


def exact(frame, **kwargs):
    events = pd.DataFrame(alerts.build_events(frame))
    dates = {x.strftime("%Y-%m-%d"): {} for x in frame["datetime"].dt.normalize().unique()}
    metadata = {"complete": True, "rule_thresholds": alerts.DEFAULT_THRESHOLDS, "dates": dates}
    return alerts.summarize_alerts("2025-04-28", hourlies(frame), events=events, event_metadata=metadata, **kwargs)


def test_one_flag_is_five_seconds_not_one_hour():
    frame = raw(["2025-04-28 12:34:00"], [True])
    result = exact(frame)
    assert result["ai"]["event_count"] == 1
    assert result["ai"]["sample_duration_seconds"] == 5
    assert result["ai"]["longest_duration_seconds"] == 5
    assert result["ai_events"][0]["duration_min"] == pytest.approx(5 / 60)
    assert result["ai_events"][0]["end"] == "2025-04-28T12:34:05"
    assert result["ai"]["affected_hours"] == 1


def test_spread_across_twenty_four_hours_is_not_twenty_four_hour_duration():
    times = [pd.Timestamp("2025-04-28") + pd.Timedelta(hours=hour, seconds=5 * sample)
             for hour in range(24) for sample in range(16 if hour == 23 else 8)]
    assert len(times) == 200
    result = exact(raw(times, [True] * len(times), "15(예비건조기)"))
    assert result["ai"]["affected_hours"] == 24
    assert result["ai"]["event_count"] == 24
    assert result["ai"]["sample_duration_seconds"] == 1000
    assert result["ai"]["longest_duration_seconds"] == 80


def test_normal_samples_and_missing_samples_split_intervals():
    times = [pd.Timestamp("2025-04-28") + pd.Timedelta(seconds=x) for x in (0, 5, 10, 15, 25, 30)]
    events = alerts.build_events(raw(times, [True, True, False, True, True, True]))
    assert [x["duration_seconds"] for x in events] == [10, 5, 10]
    assert sum(x["flagged_records"] for x in events) == 5


def test_midnight_splits_daily_intervals():
    events = alerts.build_events(raw(["2025-04-28 23:59:55", "2025-04-29 00:00:00"], [True, True]))
    assert [x["duration_seconds"] for x in events] == [5, 5]
    assert [x["date"] for x in events] == ["2025-04-28", "2025-04-29"]


def test_rule_metrics_separate_sensor_intervals_from_equipment_hours():
    frame = raw(["2025-04-28 01:00:00"], [False])
    frame["voltageR"] = 200.0
    frame["voltageS"] = 201.0
    result = exact(frame)
    assert result["rule"]["equipment_count"] == 1
    assert result["rule"]["event_count"] == 2
    assert result["rule"]["flagged_records"] == 2
    assert result["rule"]["sample_duration_seconds"] == 10
    assert result["rule"]["affected_hours"] == 1
    assert "중복" in result["note"]
    assert result["ai"]["event_count"] == 0


def test_hourly_fallback_does_not_invent_rule_records_or_duration():
    frame = raw(["2025-04-28 01:00:00"], [True])
    frame["voltageR"] = 200.0
    result = alerts.summarize_alerts("2025-04-28", hourlies(frame))
    assert result["basis"] == "hourly"
    assert result["rule"]["event_count"] is None
    assert result["rule"]["flagged_records"] is None
    assert result["rule_events"][0]["duration_seconds"] is None
    assert result["rule_events"][0]["duration_min"] is None
    assert result["ai"]["event_count"] is None
    assert result["ai_events"][0]["duration_seconds"] is None
    assert result["ai"]["sample_duration_seconds"] == 5
    assert result["ai"]["longest_duration_seconds"] is None


def test_changed_rule_keeps_exact_ai_but_falls_back_for_rule_duration():
    frame = raw(["2025-04-28 01:00:00"], [True])
    frame["voltageR"] = 200.0
    before = exact(frame)
    result = exact(frame, thresholds={"phase_voltage": 205.0})
    assert result["basis"] == "mixed"
    assert result["ai"] == before["ai"]
    assert result["rule"]["event_count"] is None
    assert result["rule_events"][0]["threshold"] == 205
    assert result["rule_events"][0]["duration_seconds"] is None


def test_empty_date_is_unknown_not_normal_or_zero():
    frame = raw(["2025-04-27 01:00:00"], [True])
    result = exact(frame)
    assert result["data_available"] is False
    assert result["ai"]["equipment_count"] is None
    assert result["rule"]["sample_duration_seconds"] is None
    assert all(x["ai"]["event_count"] is None for x in result["equipment_rows"])


def test_equipment_alias_preserves_rtu_channel_and_machine_name():
    frame = pd.concat([raw(["2025-04-28 01:00:00"], [True], "12(4호기)"),
                       raw(["2025-04-28 01:00:00"], [False], "4(분쇄기(1))")])
    result = exact(frame, equipment="4호기")
    assert result["equipment"] == "12 (4호기)"
    assert result["ai"]["flagged_records"] == 1
    assert len(result["equipment_rows"]) == 1
    assert exact(frame, equipment="4")["ai"]["flagged_records"] == 0


def test_cutoff_excludes_future_hours_and_clips_continuous_event():
    times = ["2025-04-28 00:59:55", "2025-04-28 01:00:00", "2025-04-28 01:00:05"]
    result = exact(raw(times, [True] * 3), cutoff="2025-04-28 01:00:00")
    assert result["ai"]["sample_duration_seconds"] == 5
    assert result["ai"]["event_count"] == 1
    assert result["ai_events"][0]["end"] == "2025-04-28T01:00:00"
    assert result["ai_events"][0]["value"] is None
    assert result["observed_records"] == 1


def test_duplicates_and_invalid_thresholds_rejected():
    with pytest.raises(ValueError, match="중복"):
        alerts.build_events(raw(["2025-04-28", "2025-04-28"], [True, True]))
    for value in (float("nan"), float("inf"), -1, 0):
        with pytest.raises(ValueError):
            alerts.normalize_thresholds({"phase_voltage": value})


def test_exportable_summary_has_no_nan():
    json.dumps(exact(raw(["2025-04-28"], [True])), ensure_ascii=False, allow_nan=False)


def test_source_validation_rejects_different_flags_or_sensor_values():
    frame = raw(["2025-04-28"], [True])
    hourly = hourlies(frame)
    assert _check_day(frame, hourly, "2025-04-28")["ai_flags"] == 1
    altered = hourly.copy()
    altered["ai_flags"] = 0
    with pytest.raises(ValueError, match="ai_flags"):
        _check_day(frame, altered, "2025-04-28")
    altered = hourly.copy()
    altered["voltageR"] -= 1
    with pytest.raises(ValueError, match="voltageR"):
        _check_day(frame, altered, "2025-04-28")


def test_portable_cache_hash_mismatch_falls_back(monkeypatch, tmp_path):
    hourly_path = tmp_path / "hourly.parquet"
    events_path = tmp_path / "events.parquet"
    meta_path = tmp_path / "events.json"
    frame = raw(["2025-04-28"], [True])
    hourlies(frame).to_parquet(hourly_path, index=False)
    pd.DataFrame(alerts.build_events(frame)).to_parquet(events_path, index=False)
    meta = {"format_version": 1, "complete": True, "sample_seconds": 5, "event_rows": 1,
            "events_sha256": alerts.sha256(events_path), "hourly_sha256": alerts.sha256(hourly_path),
            "dates": {"2025-04-28": {}}, "rule_thresholds": alerts.DEFAULT_THRESHOLDS}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(alerts, "HOURLY_PATH", hourly_path)
    monkeypatch.setattr(alerts, "EVENTS_PATH", events_path)
    monkeypatch.setattr(alerts, "META_PATH", meta_path)
    assert alerts.get_alert_summary("2025-04-28")["basis"] == "5s_exact"
    meta["events_sha256"] = "wrong"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    result = alerts.get_alert_summary("2025-04-28")
    assert result["basis"] == "hourly"
    assert result["ai_events"][0]["duration_seconds"] is None
    assert "검증 실패" in result["cache_note"]


def test_real_april_28_regression_if_cache_available():
    if not alerts.META_PATH.is_file():
        pytest.skip("Portable exact cache was not built in this checkout")
    result = alerts.get_alert_summary("2025-04-28")
    assert result["basis"] == "5s_exact"
    assert result["observed_records"] == 224640
    assert result["ai"]["flagged_records"] == 488
    pm3 = next(row for row in result["equipment_rows"] if row["equipment"] == "1 (PM-3)")
    dryer = next(row for row in result["equipment_rows"] if row["equipment"] == "15 (예비건조기)")
    assert pm3["ai"]["sample_duration_seconds"] == 5
    assert pm3["ai"]["longest_duration_seconds"] == 5
    assert dryer["ai"]["sample_duration_seconds"] == 1000
    assert dryer["ai"]["affected_hours"] == 24
    assert dryer["ai"]["longest_duration_seconds"] < 1000
