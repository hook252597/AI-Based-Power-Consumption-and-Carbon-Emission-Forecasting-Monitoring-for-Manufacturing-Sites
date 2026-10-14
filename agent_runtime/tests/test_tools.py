from copy import deepcopy
from pathlib import Path
import json
import pytest

from agent_runtime.agent_config import DEFAULT_THRESHOLDS, PROJECT, load_settings
from agent_runtime.agent_data import DataService
from agent_runtime.agent_tools import AgentTools, tool_schemas


@pytest.fixture(scope='module')
def data():
    if not (PROJECT / 'outputs/agent_data/manifest.json').exists():
        pytest.skip('local real-data bundle is not installed')
    return DataService(PROJECT)


def test_registered_equipment_are_known_but_missing_day_is_unknown(data):
    tools = AgentTools(data, None, {})
    rows = tools.execute('get_equipment_list', {})
    assert len(rows) == 13
    assert {r['channel'] for r in rows} == {'1','2','3','4','5','11','12','13','14','15','16','17','18'}
    assert all(r['status'] == 'unknown' and not r['data_available'] for r in rows)


def test_actual_measurement_matches_summary_and_hourly_contract(data):
    tools = AgentTools(data, None, {'view_date': '2025-04-29'})
    r = tools.execute('get_measurements', {'date': '2025-04-29', 'equipment': '3호기', 'item': 'phase_voltage_R'})
    assert r['data_available']
    assert r['min'] == pytest.approx(191.69000244140625)
    assert r['min_hour'] == 6 and r['min_timestamp'].endswith('06:05:55')
    assert 6 in r['under_threshold_hours']
    assert [v['hour'] for v in r['series']] == list(range(24))
    assert r['series'][6]['value'] != r['min']  # hourly mean != observed minimum


def test_missing_day_threshold_change_records_no_fake_counts(data):
    state = {}
    tools = AgentTools(data, None, state)
    result = tools.execute('set_thresholds', {'power_factor': 90, 'reason': '사용자 요청'})
    assert result['after']['power_factor'] == 90
    assert not result['data_available'] and result['analysis_date'] == '2025-05-31'
    assert result['anomaly_count_after'] is None and result['newly_abnormal'] is None
    assert len(state['threshold_history']) == 1
    before = deepcopy(state)
    assert 'error' in tools.execute('set_thresholds', {'power_factor': 101})
    assert state == before


def test_real_day_threshold_reanalysis_preserves_frozen_ai(data):
    tools = AgentTools(data, None, {'view_date': '2025-04-29'})
    result = tools.execute('set_thresholds', {'power_factor': 90, 'reason': '요청'})
    assert result['data_available'] and result['ai_model_unchanged']
    assert result['ai_count'] == 459
    assert result['anomaly_count_before'] == 1647
    assert result['anomaly_count_after'] > 1647


def test_energy_forecast_and_ranking_units_are_consistent(data):
    tools = AgentTools(data, None, {'view_date': '2025-04-29'})
    p = tools.execute('get_power', {'date': '2025-04-29'})
    assert p['total_kwh'] == pytest.approx(939.3611060416667)
    assert p['total_cost_krw'] == pytest.approx(p['total_kwh'] * 180)
    ranking = tools.execute('get_power_ranking', {'date': '2025-04-29'})
    assert sum(r['share_pct'] for r in ranking['items']) == pytest.approx(100)
    forecast = tools.execute('get_forecast', {'month': '2025-05'})
    assert forecast['hours'] == 505 and not forecast['full_month']
    assert forecast['expected_kwh'] == pytest.approx(19760.340119841596)
    assert forecast['prev_month_diff_pct'] is None
    assert tools.execute('get_forecast', {'month': '2025-05', 'equipment': '3호기'})['mae_kwh_per_hour'] is None


def test_key_is_not_in_settings_repr(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    path = tmp_path / '.streamlit'
    path.mkdir()
    (path / 'secrets.toml').write_text('OPENAI_API_KEY="private-test-key"', encoding='utf-8')
    settings = load_settings(tmp_path)
    assert settings.api_key == 'private-test-key'
    assert settings.api_key not in repr(settings)
    assert len(tool_schemas()) == 12
    json.dumps(tool_schemas(), allow_nan=False)


def test_report_partial_update_does_not_turn_blank_action_into_none(data, tmp_path):
    from agent_runtime.agent_reports import ReportService
    tools = AgentTools(data, ReportService(tmp_path), {'view_date': '2025-04-29'})
    first = tools.execute('create_report', {'date': '2025-04-29', 'equipment': '3호기'})
    assert not first.get('error')
    second = tools.execute('create_report', {'date': '2025-04-29', 'equipment': '3호기',
                                             'extra_fields': {'점검자': '테스트'}})
    values = {r['name']: r['value'] for r in second['rows']}
    assert values['조치 내용'] in (None, '')
    assert values['점검자'] == '테스트'
    before = deepcopy(tools.state['report_draft'])
    bad = tools.execute('create_report', {'date': '2025-04-29', 'equipment': '3호기',
                                          'extra_fields': {'조치 시각': '2025-04-29 15:00'}})
    assert 'error' in bad and '점검 일시' in bad['allowed_user_fields']
    assert tools.state['report_draft'] == before


class GroupData:
    """No filesystem, API or production data access in grouped tool tests."""
    def __init__(self):
        self.calls = []

    def get_measurements_group(self, day, equipment, item, thresholds):
        self.calls.append((day, equipment, item, deepcopy(thresholds)))
        return {'date': day, 'equipment': None if equipment == '전체' else equipment, 'item': item,
                'grouped': True, 'data_available': day == '2025-04-29', 'items': [],
                'latest_available_date': '2025-04-29', 'equipment_count': 13 if equipment == '전체' else 1}

    def get_measurements(self, day, equipment, item, thresholds):
        self.calls.append(('single', day, equipment, item))
        return {'date': day, 'equipment': equipment, 'item': item, 'min': [200], 'max': [230],
                'summary': {'minimum_observed': 200, 'maximum_observed': 230,
                            'minimum_observed_timestamp': day + 'T06:05:55'},
                'threshold': 210, 'series': [{'timestamp': day+'T06:00:00', 'mean': 220, 'min': 200}]}


def test_schema_keeps_twelve_tools_and_existing_nine_items_with_group_enums():
    schemas = tool_schemas()
    assert len(schemas) == 12 and len({s['name'] for s in schemas}) == 12
    measurement = next(s for s in schemas if s['name'] == 'get_measurements')
    expected = {f'phase_voltage_{p}' for p in 'RST'} | {f'power_factor_{p}' for p in 'RST'}
    expected |= {f'line_voltage_{p}' for p in ('RS', 'ST', 'TR')} | {'phase_voltage', 'line_voltage', 'power_factor'}
    assert set(measurement['parameters']['properties']['item']['enum']) == expected
    assert measurement['strict'] and measurement['parameters']['additionalProperties'] is False
    assert set(measurement['parameters']['required']) == {'date', 'equipment', 'item'}


@pytest.mark.parametrize('equipment,item', [('전체','phase_voltage'), ('공장 전체','line_voltage'),
                                          ('모든 설비','power_factor'), ('ALL','phase_voltage_R'),
                                          ('3호기','phase_voltage')])
def test_bulk_tool_uses_one_group_operation_without_sensor_loops(equipment, item):
    data = GroupData()
    state = {'view_date': '2025-05-31', 'selected_equipment': '12(4호기)'}
    tools = AgentTools(data, None, state)
    before = deepcopy(state)
    result = tools.execute('get_measurements', {'date': '2025-04-29', 'equipment': equipment, 'item': item})
    assert result.get('grouped') and not result.get('error')
    assert len(data.calls) == 1 and data.calls[0][0] == '2025-04-29'
    assert data.calls[0][1] == ('3호기' if equipment == '3호기' else '전체')
    assert state == before  # Reading the tool never silently chooses a new date.


def test_bulk_missing_day_returns_latest_date_without_requery_or_state_change():
    data = GroupData()
    state = {'view_date': '2025-05-31', 'selected_equipment': '12(4호기)'}
    tools = AgentTools(data, None, state)
    result = tools.execute('get_measurements', {'date': '2025-05-31', 'equipment': '전체', 'item': 'phase_voltage'})
    assert result['data_available'] is False and result['items'] == []
    assert result['latest_available_date'] == '2025-04-29'
    assert len(data.calls) == 1 and data.calls[0][0] == state['view_date'] == '2025-05-31'


def test_individual_item_still_uses_existing_single_measurement_contract():
    data = GroupData()
    result = AgentTools(data, None, {}).execute('get_measurements',
        {'date': '2025-04-29', 'equipment': '3호기', 'item': 'phase_voltage_R'})
    assert result['min'] == 200 and result['max'] == 230 and result['min_hour'] == 6
    assert result['hourly_min'] == [200] and result['series'][0]['value'] == 220
    assert len(data.calls) == 1 and data.calls[0][0] == 'single'


@pytest.mark.parametrize('extra', [{'item':'voltage_everything'}, {'equipment':None}, {'start':'2025-04-01'}])
def test_group_schema_rejects_unknown_items_or_extra_arguments_before_data_access(extra):
    data = GroupData()
    args = {'date': '2025-04-29', 'equipment': '전체', 'item': 'phase_voltage', **extra}
    assert 'error' in AgentTools(data, None, {}).execute('get_measurements', args)
    assert not data.calls
