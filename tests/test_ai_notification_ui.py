from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui_ai_notifications import candidate_table, equipment_table, log_table, notification_table, _csv
from monitoring.ai_notifications import DEFAULT_POLICY


def fixture_result():
    logs = []
    for index in range(4):
        logs.append(dict(event_id=f"event-{index}", equipment="13 (3호기)", date="2025-04-29",
            start=f"2025-04-29T10:0{index}:00", end=f"2025-04-29T10:0{index}:05",
            score_max=0.71, score_max_at=f"2025-04-29T10:0{index}:00", base_threshold=0.65,
            ai_records=1, duration_seconds=5, qualified_records=1 if index < 3 else 0,
            max_qualifying_duration_seconds=5 if index < 3 else 0,
            notification_candidate=index < 3,
            log_only_reason=None if index < 3 else "below_score_margin"))
    trigger = dict(candidate_id="trigger-1", event_id="event-2", equipment="13 (3호기)",
        trigger_at="2025-04-29T10:02:05", reasons=["repeated"], duration_seconds=5,
        repeat_event_count=3, score_at_trigger=0.71, base_threshold=0.65, score_cutoff=0.67,
        supporting_event_ids=["event-0", "event-1", "event-2"])
    notification = dict(notification_id="notice-1", equipment="13 (3호기)",
        first_trigger_at="2025-04-29T10:02:05", last_trigger_at="2025-04-29T10:02:05",
        candidate_count=1, reasons=["repeated"], source_event_ids=["event-0", "event-1", "event-2"],
        score_max_at_triggers=0.71)
    summary = dict(original_event_count=4, notified_original_event_count=3, log_only_event_count=1,
        notification_count=1, candidate_count=1, ai_record_count=4, qualified_record_count=3,
        sustained_candidate_count=0, repeated_candidate_count=1)
    return dict(status="ok", data_available=True, policy=DEFAULT_POLICY, summary=summary,
        all_logs=logs, candidate_events=logs[:3], log_only=logs[3:], trigger_candidates=[trigger],
        notifications=[notification], per_equipment=[dict(equipment="13 (3호기)", data_available=True, **summary)],
        coverage={}, note="")


def app(result=None):
    payload = fixture_result() if result is None else result
    source = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import streamlit as st
from unittest.mock import patch
import ui_ai_notifications as ui
payload = {payload!r}
def backend(**kwargs):
    st.session_state['_requested_policy'] = kwargs
    return payload
with patch.object(ui, 'get_ai_notifications', backend):
    ui.render_ai_notifications('2025-04-29', '13 (3호기)')
"""
    source = source.encode("ascii", "backslashreplace").decode("ascii")
    return AppTest.from_string(source, default_timeout=20).run()


def test_notification_and_original_record_denominators_are_distinct():
    result = app()
    assert not result.exception
    metrics = {element.label: element.value for element in result.metric}
    assert metrics == {"원래 AI 탐지": "4구간", "알림 후보": "1묶음", "후보에 포함된 원기록": "3구간", "기록만 보관": "1구간"}
    assert any("실제 알림은 발송하지" in element.value for element in result.info)
    assert any("고장 확률이 아니" in element.value for element in result.info)
    assert result.session_state['_requested_policy']['equipment'] is None


def test_unavailable_is_not_rendered_as_zero_or_normal():
    result = app(dict(status="unavailable", data_available=False, summary={}))
    assert not result.exception
    assert len(result.metric) == 0
    assert any("자료 없음은 정상이나 알림 0건을 뜻하지" in element.value for element in result.info)


def test_empty_but_observed_scope_has_zero_and_honest_notice():
    payload = fixture_result()
    for key in ("all_logs", "candidate_events", "log_only", "trigger_candidates", "notifications", "per_equipment"):
        payload[key] = []
    payload['summary'] = {key: 0 for key in payload['summary']}
    result = app(payload)
    assert not result.exception
    assert result.metric[0].value == "0구간"
    assert any("후보가 없다고 설비가 정상" in element.value for element in result.info)


def test_policy_changes_are_explicit_and_keep_original_logs():
    result = app()
    result.selectbox(key="ai_policy_score_margin").select(0.05)
    result.selectbox(key="ai_policy_duration").select(180)
    next(button for button in result.button if button.label == "이 설정으로 후보 다시 보기").click().run()
    assert not result.exception
    assert result.session_state['ai_notification_policy']['score_margin'] == 0.05
    assert result.session_state['ai_notification_policy']['duration_seconds'] == 180
    assert result.session_state['_requested_policy']['policy']['score_margin'] == 0.05
    assert {element.label: element.value for element in result.metric}["원래 AI 탐지"] == "4구간"


def test_period_and_detail_filter_preserve_whole_factory_overview():
    result = app()
    result.radio(key="ai_policy_period").set_value("전체 실측기간").run()
    result.selectbox(key="ai_notification_detail_equipment").select("13 (3호기)").run()
    assert not result.exception
    assert result.session_state['_requested_policy']['day'] is None
    assert result.session_state['_requested_policy']['equipment'] is None
    assert {element.label: element.value for element in result.metric}["원래 AI 탐지"] == "4구간"
    assert not any(button.label == "이 설비로 보고서 시작 →" for button in result.button)


def test_exports_preserve_all_logs_and_actual_trigger_time():
    payload = fixture_result()
    original = deepcopy(payload)
    table = log_table(payload['all_logs'])
    assert len(table) == 4
    assert (table['분류'] == "기록만 보관").sum() == 1
    assert "추가 점수 기준 미달" in _csv(table).decode('utf-8-sig')
    notice = notification_table(payload['notifications'])
    assert notice.iloc[0]['첫 조건 충족 시각'] == "2025-04-29T10:02:05"
    assert notice.iloc[0]['첫 조건 충족 시각'] != payload['all_logs'][0]['start']
    evidence = candidate_table(payload['trigger_candidates'])
    assert evidence.iloc[0]['알림 근거'] == "반복 횟수 충족"
    assert payload == original


def test_unobserved_equipment_never_gets_zero_in_overview():
    table = equipment_table([dict(equipment="13 (3호기)", data_available=False, original_event_count=0,
        notification_count=0, notified_original_event_count=0, log_only_event_count=0)])
    assert table.iloc[0]['원래 AI 탐지'] == "확인 불가"
    assert table.iloc[0]['알림 후보'] == "확인 불가"



def test_detail_selection_does_not_remove_other_equipment_from_overview():
    payload = fixture_result()
    payload['per_equipment'].append(dict(equipment="12 (4호기)", data_available=True,
        original_event_count=0, notified_original_event_count=0, log_only_event_count=0, notification_count=0))
    result = app(payload)
    result.selectbox(key="ai_notification_detail_equipment").select("12 (4호기)").run()
    assert not result.exception
    overview = result.dataframe[0].value
    assert set(overview['설비']) == {"13 (3호기)", "12 (4호기)"}
    assert {element.label: element.value for element in result.metric}["원래 AI 탐지"] == "4구간"
    assert any(button.label == "이 설비로 보고서 시작 →" for button in result.button)


def test_theme_change_preserves_selected_equipment_and_sensor_policy(monkeypatch):
    # Theme/selection behavior is independent of a developer's raw-data cache.
    import ui_sensor_insights as sensor_ui
    import monitoring.sensor_insights as sensor_backend
    from test_sensor_insight_ui import fixture_result as sensor_fixture
    def editable_backend(day, policy=None, cutoff=None):
        payload = sensor_fixture()
        payload.update(day=str(day), policy=sensor_backend.normalize_policy(policy), policy_editable=True)
        return payload
    monkeypatch.setattr(sensor_backend, 'get_sensor_insights', editable_backend)
    monkeypatch.setattr(sensor_ui, 'get_sensor_insights', editable_backend)
    result = AppTest.from_file(str(ROOT / "dashboard_app.py"), default_timeout=45).run()
    result.radio(key="nav_page").set_value("이상탐지").run()
    result.selectbox(key="sensor_insight_detail_equipment").select("13 (3호기)").run()
    result.number_input(key="sensor_cfg_voltage_severe_low").set_value(195.0)
    next(button for button in result.button if button.label == "이 기준 적용").click().run()
    result.radio(key="dashboard_theme_mode").set_value("dark").run()
    assert not result.exception
    assert result.session_state['theme'] == 'dark'
    assert result.session_state['sensor_insight_detail_equipment'] == '13 (3호기)'
    assert result.session_state['sensor_insight_policy']['voltage_severe_low'] == 195.0
    assert any(button.label == '이 설비로 보고서 시작 →' for button in result.button)
