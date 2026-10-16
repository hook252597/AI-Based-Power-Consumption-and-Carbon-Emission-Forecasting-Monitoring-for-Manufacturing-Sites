from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitoring.alert_service import EQUIPMENT_LIST
from monitoring.sensor_insights import DEFAULT_POLICY
from ui_sensor_insights import groups_table, unlinked_ai_logs, ai_log_table, policy_table
import dashboard_data as D


def fixture_result():
    summaries = []
    for _, name in EQUIPMENT_LIST:
        count = 2 if name == "13 (3호기)" else 1 if name == "17 (6호기)" else 0
        summaries.append(dict(equipment=name, data_available=True, voltage_min=196.0, voltage_max=235.0,
            pf_min=68.0, pf_max=95.0, original_event_count=count, ai_record_count=count,
            event_count=int(name == "13 (3호기)"), group_count=int(name == "13 (3호기)"),
            voltage_severe_count=int(name == "13 (3호기)"), voltage_sustained_count=0,
            pf_sharp_count=0, pf_sustained_count=0))
    logs = [dict(id=f"ai-{i}", event_id=f"ai-{i}", equipment=name, start=f"2025-04-29T00:0{i}:00",
        end=f"2025-04-29T00:0{i}:05", score_max=0.7, ai_records=1, duration_seconds=5, preserved=True)
        for i, name in enumerate(["13 (3호기)", "13 (3호기)", "17 (6호기)"], start=1)]
    event = dict(id="sensor-1", equipment="13 (3호기)", item="voltageR", kind="voltage_severe", direction="low",
        start="2025-04-29T00:01:00", end="2025-04-29T00:01:05", trigger_at="2025-04-29T00:01:05",
        value_at_trigger=197.0, extreme_value=196.0, duration_seconds=5, baseline=None, drop_pp=None,
        ai_score=0.7, source_ai_event_ids=["ai-1"])
    group = dict(id="group-1", equipment="13 (3호기)", item="voltageR", kind="voltage_severe", direction="low",
        first_trigger_at="2025-04-29T00:01:05", last_trigger_at="2025-04-29T00:01:05",
        event_count=1, event_ids=["sensor-1"], extreme_value=196.0, first_value_at_trigger=197.0)
    return dict(status="ok", data_available=True, day="2025-04-29", policy=dict(DEFAULT_POLICY),
        summary=dict(original_event_count=3, ai_record_count=3, event_count=1, group_count=1,
            voltage_severe_count=1, voltage_sustained_count=0, pf_sharp_count=0, pf_sustained_count=0),
        per_equipment=summaries, events=[event], groups=[group], all_ai_logs=logs, note="",
        source=dict(partition_sha256="source-hash", model_sha256="model-hash"),
        coverage=dict(dates=["2025-04-29"], sample_seconds=5, observed_records=3))


def app(payload=None, active_tab="이상탐지"):
    payload = fixture_result() if payload is None else payload
    source = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from copy import deepcopy
from unittest.mock import patch
import streamlit as st
import ui_sensor_insights as ui
st.session_state.setdefault('active_tab', {active_tab!r})
payload = {payload!r}
def backend(*args, **kwargs):
    st.session_state['_requested_sensor_policy'] = kwargs
    result = deepcopy(payload)
    result['policy'] = kwargs.get('policy', payload.get('policy'))
    return result
with patch.object(ui, 'get_sensor_insights', backend):
    ui.render_sensor_insights('2025-04-29')
"""
    return AppTest.from_string(source.encode('ascii', 'backslashreplace').decode('ascii'), default_timeout=20).run()


def card_buttons(result):
    return [button for button in result.button if button.key and button.key.startswith('sensor_overview_equipment_')]


def test_all_thirteen_equipment_cards_and_original_record_count_are_visible():
    result = app()
    assert not result.exception
    assert len(card_buttons(result)) == 13
    assert {item.label: item.value for item in result.metric} == {
        '원래 AI 탐지': '3구간', '확인할 정보': '1묶음', '정보가 있는 설비': '1대 / 13대'}
    assert any('실제 고장 판정 아님' in item.value for item in result.caption)


def test_unavailable_keeps_thirteen_unknown_cards_instead_of_zero():
    payload = fixture_result()
    payload.update(status='unavailable', data_available=False)
    payload['per_equipment'] = [dict(equipment=name, data_available=False) for _, name in EQUIPMENT_LIST]
    result = app(payload)
    assert not result.exception
    assert len(card_buttons(result)) == 13
    assert not result.metric
    assert any('자료 없음은 정상이나 0건' in item.value for item in result.info)
    assert sum(item.value == '실측 자료 없음' for item in result.markdown) == 13


def test_detail_filter_keeps_whole_factory_cards_and_navigates_from_overview():
    result = app(active_tab='현황·예측')
    result.button(key='sensor_overview_equipment_13').click().run()
    assert not result.exception
    assert len(card_buttons(result)) == 13
    assert result.session_state['sensor_insight_detail_equipment'] == '13 (3호기)'
    assert result.session_state['_pending_tab'] == '이상탐지'
    assert result.session_state['inspection_mode'] == '확인할 정보'
    assert {item.label: item.value for item in result.metric}['원래 AI 탐지'] == '3구간'


def test_apply_and_restore_policy_keep_original_model_logs():
    result = app()
    result.number_input(key='sensor_cfg_voltage_severe_low').set_value(190.0)
    next(button for button in result.button if button.label == '이 기준 적용').click().run()
    assert not result.exception
    assert result.session_state['sensor_insight_policy']['voltage_severe_low'] == 190.0
    assert result.session_state['_requested_sensor_policy']['policy']['voltage_severe_low'] == 190.0
    assert result.session_state['_requested_sensor_policy']['cutoff'] == D.NOW
    assert {item.label: item.value for item in result.metric}['원래 AI 탐지'] == '3구간'
    next(button for button in result.button if button.label == '과제용 기본값 복원').click().run()
    assert not result.exception
    assert result.session_state['sensor_insight_policy'] == DEFAULT_POLICY
    assert result.number_input(key='sensor_cfg_voltage_severe_low').value == 198.0


def test_invalid_policy_does_not_replace_applied_policy():
    result = app()
    result.number_input(key='sensor_cfg_voltage_severe_low').set_value(220.0)
    next(button for button in result.button if button.label == '이 기준 적용').click().run()
    assert not result.exception
    assert result.session_state['sensor_insight_policy']['voltage_severe_low'] == 198.0
    assert any('기준을 적용하지 못했어요' in item.value for item in result.warning)


def test_unlinked_log_partition_and_first_evidence_time_are_preserved():
    payload = fixture_result()
    original = deepcopy(payload)
    assert [row['event_id'] for row in unlinked_ai_logs(payload)] == ['ai-2', 'ai-3']
    assert len(ai_log_table(payload['all_ai_logs'])) == 3
    table = groups_table(payload['groups'], payload['events'], DEFAULT_POLICY)
    assert table.iloc[0]['첫 확인 시각'] == '2025-04-29T00:01:05'
    assert table.iloc[0]['첫 확인 시각'] != payload['events'][0]['start']
    assert table.iloc[0]['연결 AI 원구간 수'] == 1
    assert '낮아짐' in table.iloc[0]['확인 내용']
    assert '급락 구간 시작 전 5분 중앙값(구간 동안 고정)' in policy_table(DEFAULT_POLICY).iloc[2]['측정값 조건']
    assert payload == original


def test_report_context_is_pinned_when_policy_later_changes():
    result = app()
    result.selectbox(key='sensor_insight_detail_equipment').select('13 (3호기)').run()
    result.button(key='sensor_insight_start_report').click().run()
    assert not result.exception
    context = result.session_state['sensor_insight_report_context']
    assert context['day'] == '2025-04-29'
    assert context['equipment'] == '13 (3호기)'
    assert len(context['events']) == 1
    assert context['policy']['voltage_severe_low'] == 198.0
    assert context['source']['partition_sha256'] == 'source-hash'
    assert context['coverage']['sample_seconds'] == 5
    assert context['status'] == 'ok' and context['data_available'] is True
    result.number_input(key='sensor_cfg_voltage_severe_low').set_value(190.0)
    next(button for button in result.button if button.label == '이 기준 적용').click().run()
    assert result.session_state['sensor_insight_report_context']['policy']['voltage_severe_low'] == 198.0
    assert result.session_state['sensor_insight_policy']['voltage_severe_low'] == 190.0


def test_validation_failure_explains_cause_without_local_path_or_false_missing_date():
    payload = fixture_result()
    payload.update(status='unavailable', data_available=False,
        note="실측 자료 검증 실패: FileNotFoundError: [Errno 2] 'C:/private/sensor.parquet'")
    result = app(payload)
    assert not result.exception
    assert any('검증된 5초 실측 자료를 사용할 수 없어요' in item.value for item in result.info)
    notes = '\n'.join(item.value for item in result.caption)
    assert 'FileNotFoundError' in notes and '[로컬 파일]' in notes
    assert 'C:/private' not in notes


def test_portable_results_lock_policy_edits_but_allow_reset_and_guide():
    payload = fixture_result()
    payload['policy_editable'] = False
    payload['source']['mode'] = 'portable_snapshot'
    result = app(payload)
    assert not result.exception
    assert all(item.disabled for item in result.number_input)
    assert next(button for button in result.button if button.label == '이 기준 적용').disabled
    assert not next(button for button in result.button if button.label == '과제용 기본값 복원').disabled
    assert len(card_buttons(result)) == 13
    result.button(key='sensor_insight_guide').click().run()
    assert result.session_state['_pending_tab'] == '시작 가이드'
