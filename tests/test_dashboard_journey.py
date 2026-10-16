from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def app():
    result=AppTest.from_file(str(ROOT/'dashboard_app.py'),default_timeout=45).run()
    assert not result.exception
    return result

def page(result,name):
    result.radio(key='nav_page').set_value(name).run()
    if name == '이상탐지':
        result.radio(key='inspection_mode').set_value('전체 탐지 기록').run()
    assert not result.exception
    return result

def test_default_guide_and_task_navigation():
    result=app()
    assert result.session_state['active_tab']=='시작 가이드'
    for name in ['현황·예측','이상탐지','에이전트','시작 가이드']:
        page(result,name)
        assert result.session_state['active_tab']==name

def test_sensor_no_data_and_recovery():
    result=page(app(),'이상탐지')
    result.selectbox(key='context_month').select('2025-05').run()
    result.selectbox(key='context_day').select(31).run()
    assert not result.exception
    assert any('실측이 없어서' in item.value for item in result.info)
    next(b for b in result.button if b.label=='마지막 실측일 · 4월 29일 보기').click().run()
    assert not result.exception
    assert result.session_state['view_date']=='2025-04-29'
    assert any(item.value=='459구간' for item in result.metric)

def test_inspection_to_pinned_report():
    result=page(app(),'이상탐지')
    next(b for b in result.button if b.label=='이 설비로 보고서 시작 →').click().run()
    assert not result.exception
    assert result.session_state['active_tab']=='에이전트'
    draft=result.session_state['report_draft']
    assert draft['equipment']=='13 (3호기)'
    assert draft['date']=='2025-04-29'
    result.selectbox(key='context_day').select(28).run()
    assert not result.exception
    assert result.session_state['report_draft']['date']=='2025-04-29'

def test_hourly_fallback_and_rule_change_remain_honest():
    result=page(app(),'이상탐지')
    result.selectbox(key='context_month').select('2025-03').run()
    assert not result.exception
    assert any('구간 수와 연속 지속 시간' in item.value for item in result.caption)
    assert any('시간대' in item.value for item in result.metric)
    result.selectbox(key='context_month').select('2025-04').run()
    next(x for x in result.number_input if x.label=='상전압 하한(V)').set_value(200)
    next(x for x in result.button if x.label=='설정 기준 적용').click().run()
    assert not result.exception
    assert any(x.value=='459구간' for x in result.metric)
    assert any('구간 수와 연속 지속 시간' in x.value for x in result.caption)


def test_sensor_insight_deep_link_preserves_all_equipment():
    result = AppTest.from_file(str(ROOT/'dashboard_app.py'), default_timeout=45)
    result.query_params['view'] = 'sensor-insights'
    result.run()
    assert not result.exception
    assert result.session_state['active_tab'] == '이상탐지'
    assert result.radio(key='inspection_mode').value == '확인할 정보'
    assert sum(button.label == '보기 →' for button in result.button) == 13
    assert not result.warning


def test_overview_card_opens_selected_machine_without_hiding_other_cards():
    result = page(app(), '현황·예측')
    assert sum(button.label == '보기 →' for button in result.button) == 13
    result.button(key='overview_equipment_13').click().run()
    assert not result.exception
    assert result.session_state['active_tab'] == '이상탐지'
    assert result.session_state['sensor_insight_detail_equipment'] == '13 (3호기)'
    assert sum(button.label == '보기 →' for button in result.button) == 13
    assert any(button.label == '이 설비로 보고서 시작 →' for button in result.button)


def test_sensor_help_opens_guide_with_preserved_criteria_and_report_help():
    result = AppTest.from_file(str(ROOT/'dashboard_app.py'), default_timeout=45)
    result.query_params['view'] = 'sensor-insights'
    result.run()
    result.button(key='sensor_insight_guide').click().run()
    assert not result.exception
    assert result.session_state['active_tab'] == '시작 가이드'
    text = '\n'.join(item.value for item in result.markdown)
    assert '198V 미만' in text
    assert '10분 미만' in text
    assert '초안 생성 시점에 고정' in text
    assert '원본 5초 실측 자료' in text
