"""Twelve allowlisted in-process tools shared by the dashboard and chat."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
import json
import math
from zoneinfo import ZoneInfo
import jsonschema

from .agent_config import DEFAULT_THRESHOLDS, TODAY
from .agent_reports import _canonical

MEASUREMENTS = ['phase_voltage_R', 'phase_voltage_S', 'phase_voltage_T',
                'line_voltage_RS', 'line_voltage_ST', 'line_voltage_TR',
                'power_factor_R', 'power_factor_S', 'power_factor_T']
MEASUREMENT_GROUPS = ['phase_voltage', 'line_voltage', 'power_factor']


def prop(kind='string', nullable=False, **kwargs):
    return {'type': [kind, 'null'] if nullable else kind, **kwargs}


def function(name, description, properties):
    return {'type': 'function', 'name': name, 'description': description, 'strict': True,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': list(properties), 'additionalProperties': False}}


def tool_schemas():
    date = prop(description='한국 실측 날짜 YYYY-MM-DD. 제공 센서는 2025-04-01~29.')
    equip = prop(nullable=True, description='원본 설비 ID 또는 4호기 같은 설비명. null은 전체.')
    field = {'type': 'object', 'properties': {
        'name': prop(), 'source': prop(enum=['auto', 'user', 'blank']),
        'required': prop('boolean'), 'maps_to': prop(nullable=True)},
        'required': ['name', 'source', 'required', 'maps_to'], 'additionalProperties': False}
    extra = {'type': ['array', 'null'], 'items': {'type': 'object', 'properties': {
        'name': prop(), 'value': prop()}, 'required': ['name', 'value'], 'additionalProperties': False},
        'description': '사용자가 명시한 보고서 항목/값 목록. 추정해서 채우지 않음.'}
    return [
        function('get_equipment_list', '실제 등록 설비 목록·개수와 선택일 상태. AI/센서 기준 판정은 별개.', {}),
        function('get_anomalies', '실측 센서 기준 위반 구간과 AI 이상 구간 조회. 고장 확정 아님.', {'date': date, 'equipment': equip}),
        function('get_measurements', '개별 센서 또는 설비별 묶음 요약. 전체 설비 전압은 equipment="전체", item="phase_voltage"로 한 번 조회. 설비 간 평균을 합치지 않음. 자료 없으면 날짜를 바꾸지 않음.',
                 {'date': date, 'equipment': prop(description='등록된 설비 이름 또는 전체. 전체는 설비별 결과표를 반환.'),
                  'item': prop(enum=MEASUREMENTS + MEASUREMENT_GROUPS,
                               description='phase_voltage=상전압 R/S/T 묶음, line_voltage=선간 RS/ST/TR, power_factor=역률 R/S/T. 개별 상도 지정 가능.')}),
        function('get_power', '실측 전력 kWh·고정단가 요금·탄소 조회.', {'date': date, 'equipment': equip, 'granularity': prop(enum=['hour', 'day'])}),
        function('get_power_ranking', '실측일의 설비별 전력 소비 순위 조회.', {'date': date}),
        function('get_forecast', 'Extra Trees의 2025-05-10~31 00:00 505시간 예측. 월 전체/실측 아님.', {'month': prop(), 'equipment': equip}),
        function('get_thresholds', '변경 가능한 프로젝트 센서 경고 기준과 변경 이력. AI 임계값은 고정.', {}),
        function('set_thresholds', '사용자가 명시적으로 요청한 센서 경고 기준 변경 및 재분석. AI 모델 변경 금지.', {
            'phase_voltage': prop('number', True, minimum=100, maximum=300),
            'line_voltage': prop('number', True, minimum=200, maximum=600),
            'power_factor': prop('number', True, minimum=1, maximum=100), 'reason': prop()}),
        function('get_report_template', '현재 보고서 양식과 필수 입력 항목 조회.', {}),
        function('save_report_template', '새 양식의 확인 대기 초안을 준비. 실제 적용은 화면의 확인 버튼으로 수행.', {
            'fields': prop('array', items=field, minItems=1, maxItems=60),
            'layout': prop(enum=['table']), 'source_file': prop(nullable=True)}),
        function('reset_report_template', '사용자가 요청한 경우 기본 보고서 양식으로 복원.', {}),
        function('create_report', '사용자가 제공한 조치 내용과 실측 자료로 초안 작성. 저장/전송은 하지 않음.', {
            'date': date, 'equipment': prop(), 'action_taken': prop(), 'extra_fields': extra}),
    ]


class AgentTools:
    def __init__(self, data, reports, state):
        self.data, self.reports, self.state = data, reports, state
        state.setdefault('thresholds', deepcopy(DEFAULT_THRESHOLDS))
        state.setdefault('threshold_history', [])
        state.setdefault('view_date', TODAY)
        state.setdefault('selected_equipment', '12(4호기)')

    def schema(self):
        return tool_schemas()

    def execute(self, name, args):
        schema = next((s for s in self.schema() if s['name'] == name), None)
        if schema is None:
            return {'error': '허용되지 않은 도구입니다.'}
        try:
            # UI/local callers may omit nullable optional fields, unlike strict API requests.
            args = dict(args or {})
            defaults = {'equipment': None, 'phase_voltage': None, 'line_voltage': None,
                        'power_factor': None, 'reason': '', 'action_taken': '', 'extra_fields': None,
                        'source_file': None, 'layout': 'table', 'granularity': 'hour'}
            if name == 'create_report' and isinstance(args.get('extra_fields'), dict):
                args['extra_fields'] = [{'name': k, 'value': str(v)} for k, v in args['extra_fields'].items()]
            for key in schema['parameters']['required']:
                if key not in args and key in defaults:
                    args[key] = defaults[key]
            jsonschema.validate(args, schema['parameters'])
            result = getattr(self, name)(**args)
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            return result
        except jsonschema.ValidationError:
            return {'error': '입력 형식이나 값 범위를 확인해 주세요.'}
        except (ValueError, KeyError) as exc:
            return {'error': str(exc)[:220]}
        except Exception:
            return {'error': '자료를 처리하지 못했습니다. 데이터 준비 상태를 확인해 주세요.'}

    def get_equipment_list(self):
        rows = self.data.get_equipment_list(self.state['view_date'], self.state['thresholds'])
        anomalies = self.get_anomalies(self.state['view_date'])
        for row in rows:
            equipment = row['equipment']
            row['channel'] = equipment.split('(', 1)[0]
            row['name'] = equipment.replace('(', ' (', 1)
            row['date'] = self.state['view_date']
            row['status'] = ('unknown' if not row['data_available'] else
                             'abnormal' if row.get('rule_event_count') or row.get('ai_event_count') else 'normal')
            row['abnormal_items'] = sorted({e['item'] for e in anomalies.get('items', [])
                                            if e['equipment'] == equipment})
        return rows

    def get_anomalies(self, date, equipment=None):
        return self.data.get_anomalies(date, equipment, self.state['thresholds'])

    def get_measurements(self, date, equipment, item):
        all_equipment = equipment is None or ''.join(str(equipment).split()).lower() in {'', 'all', '전체', '모든설비', '공장전체', 'factory'}
        if item in MEASUREMENT_GROUPS or all_equipment:
            # One data-service operation reads the date once and summarizes
            # each equipment independently. Never issue 39 individual queries.
            return self.data.get_measurements_group(date, '전체' if all_equipment else equipment,
                                                    item, self.state['thresholds'])
        result = self.data.get_measurements(date, equipment, item, self.state['thresholds'])
        summary = result.get('summary', {})
        # Keep exact five-second extrema distinct from hourly means.
        result['hourly_min'] = result.get('min', [])
        result['hourly_max'] = result.get('max', [])
        result['min'] = summary.get('minimum_observed')
        result['max'] = summary.get('maximum_observed')
        result['minimum_value'] = summary.get('minimum_observed')
        stamp = summary.get('minimum_observed_timestamp')
        result['min_hour'] = datetime.fromisoformat(stamp).hour if stamp else None
        result['min_timestamp'] = stamp
        result['under_threshold_hours'] = [datetime.fromisoformat(row['timestamp']).hour for row in result.get('series', [])
            if row.get('min') is not None and result.get('threshold') is not None
            and row['min'] < result['threshold']]
        result['under_threshold_hours_basis'] = 'hours containing at least one observed five-second value below threshold'
        for row in result.get('series', []):
            row['hour'], row['value'] = datetime.fromisoformat(row['timestamp']).hour, row.get('mean')
        return result

    def get_power(self, date, equipment=None, granularity='hour'):
        result = self.data.get_power(date, equipment, granularity)
        result.update(total_cost_krw=result.get('cost'), total_co2_kg=result.get('co2'),
                      cost_rate_krw_per_kwh=180, co2_factor_kg_per_kwh=.424)
        for row in result.get('series', []):
            row['t'] = row['timestamp']
        return result

    def get_power_ranking(self, date):
        result = self.data.get_power_ranking(date)
        total = result.get('total_kwh')
        for row in result.get('items', []):
            row['share_pct'] = row['kwh'] / total * 100 if total and row.get('kwh') is not None else None
        return result

    def get_forecast(self, month, equipment=None):
        result = self.data.get_forecast(month, equipment)
        result.update(expected_kwh=result.get('total_kwh'), expected_cost_krw=result.get('cost'),
                      expected_co2_kg=result.get('co2'), prev_month_diff_pct=None,
                      mae_kwh_per_hour=result.get('mae'), cost_rate_krw_per_kwh=180,
                      co2_factor_kg_per_kwh=.424)
        return result

    def get_thresholds(self):
        return {**self.state['thresholds'], 'is_default': self.state['thresholds'] == DEFAULT_THRESHOLDS,
                'changed_log': self.state['threshold_history'], 'scope': 'sensor_rules_only',
                'ai_thresholds_fixed': True}

    def set_thresholds(self, phase_voltage=None, line_voltage=None, power_factor=None, reason=''):
        previous = deepcopy(self.state['thresholds'])
        updated = deepcopy(previous)
        bounds = {'phase_voltage': (100, 300), 'line_voltage': (200, 600), 'power_factor': (1, 100)}
        for key, value in [('phase_voltage', phase_voltage), ('line_voltage', line_voltage), ('power_factor', power_factor)]:
            if value is not None:
                lo, hi = bounds[key]
                if isinstance(value, bool) or not math.isfinite(value) or not lo <= value <= hi:
                    raise ValueError('경고 기준 범위를 확인해 주세요.')
                updated[key] = float(value)
        before = self.data.get_anomalies(self.state['view_date'], None, previous)
        after = self.data.get_anomalies(self.state['view_date'], None, updated)
        available = bool(before.get('data_available') and after.get('data_available'))
        if available and before.get('ai_count') != after.get('ai_count'):
            raise ValueError('AI 판정이 바뀌어 기준 변경을 취소했습니다.')
        was = {i['equipment'] for i in before.get('items', [])}
        now = {i['equipment'] for i in after.get('items', [])}
        entry = {'at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),
                 'by': '로컬 사용자', 'before': previous, 'after': updated, 'reason': reason}
        if previous != updated:
            self.state['thresholds'] = updated
            self.state['threshold_history'].append(entry)
            self.state['report_draft'] = None  # Existing analysis/threshold snapshot is stale.
        return {'before': previous, 'after': updated, 'anomaly_count_before': before.get('count') if available else None,
                'anomaly_count_after': after.get('count') if available else None,
                'newly_abnormal': sorted(now - was) if available else None,
                'no_longer_abnormal': sorted(was - now) if available else None, 'ai_count': after.get('ai_count'),
                'analysis_date': self.state['view_date'], 'data_available': available,
                'ai_model_unchanged': True, 'log_entry': entry,
                'note': ('센서 경고 기준만 변경했습니다. 이상 구간 수는 고장 사건 수가 아닙니다.' if available else
                         '기준을 변경했습니다. 선택일의 센서 자료가 없어 재분석 건수는 제공할 수 없습니다. 실측일을 선택하면 변경한 기준이 적용됩니다.')}

    def get_report_template(self):
        return self.reports.get_report_template()

    def save_report_template(self, fields, layout='table', source_file=None):
        self.state['pending_template'] = {'fields': deepcopy(fields), 'layout': layout, 'source_file': source_file}
        return {'requires_confirmation': True, 'fields': fields,
                'note': '화면의 양식 확인·적용 버튼을 눌러야 저장됩니다.'}

    def reset_report_template(self):
        self.state['report_draft'] = None
        self.state['pending_template'] = None
        return self.reports.reset_report_template()

    def create_report(self, date, equipment, action_taken='', extra_fields=None):
        equipment = self.data.normalize_equipment(equipment)
        if equipment is None:
            raise ValueError('보고서를 작성할 설비 한 개를 선택해 주세요.')
        extras = {r['name']: r['value'] for r in extra_fields} if isinstance(extra_fields, list) else dict(extra_fields or {})
        template = self.reports.get_report_template()
        user_fields = {_canonical(f['name']) for f in template['fields'] if f.get('source') == 'user'}
        unknown = [name for name in extras if _canonical(name) not in user_fields]
        if unknown:
            return {'error': '양식에 없는 사용자 항목입니다: ' + ', '.join(unknown),
                    'allowed_user_fields': sorted(user_fields),
                    'note': 'get_report_template에서 실제 항목 이름을 확인하세요. 점검 일시와 조치 일시는 서로 다른 항목입니다.'}
        extras = {_canonical(name): value for name, value in extras.items()}
        previous = self.state.get('report_draft')
        context = self.state.get('report_context')
        if previous and context == (date, equipment):
            old = {r['name']: r.get('value', '') for r in previous.get('rows', []) if r.get('source') == 'user'}
            old.update(extras)
            extras = old
            if not action_taken:
                action_taken = str(extras.get('조치 내용', extras.get('조치내용', '')) or '')
        if action_taken:
            # An explicitly updated action must supersede the previous draft.
            extras['조치 내용'] = action_taken
        readings = {key: self.get_measurements(date, equipment, key) for key in MEASUREMENTS}
        power = self.get_power(date, equipment)
        anomalies = self.get_anomalies(date, equipment)
        draft = self.reports.create_report(date, equipment, action_taken, extras,
                                          measurements=readings, power=power, anomalies=anomalies)
        draft['storage_status'] = 'session_draft_only'
        draft['export_blockers'] = list(draft.get('missing_user_fields', []))
        if draft.get('source_identity_mismatch'):
            draft['export_blockers'].append('자료 출처 불일치')
        draft['export_note'] = '내보내기 제한 사유는 export_blockers뿐입니다. 선택 자동항목의 결측은 원인이 아닙니다. 보고서는 화면의 세션 초안이며 파일로 저장하거나 전송하지 않았습니다.'
        self.state['report_draft'] = draft
        self.state['report_context'] = (date, equipment)
        self.state['selected_equipment'] = equipment
        self.state['view_date'] = date
        if 'template_name' not in draft:
            draft['template_name'] = self.reports.get_report_template().get('name', '기본 점검 보고서')
        return draft
