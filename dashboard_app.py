"""마가렛트: 처음 사용하는 사람을 위한 전력 현황 · 점검 · 보고서 화면."""
from __future__ import annotations

import calendar
import html
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dashboard_data as D
from monitoring.alert_service import get_alert_summary

PAGES = ['시작 가이드', '현황·예측', '이상탐지', '에이전트']
PAGE_LABELS = {'시작 가이드': '시작 가이드', '현황·예측': '현황·예측', '이상탐지': '설비 점검', '에이전트': '보고서 도우미'}
TEAL, BLUE, AMBER = '#087f8c', '#5267df', '#a96312'

st.set_page_config(page_title='마가렛트 · 전력 관리와 설비 점검', page_icon='🏭', layout='wide', initial_sidebar_state='collapsed')


def init_state():
    defaults = {'active_tab': '시작 가이드', 'selected_year': 2025, 'selected_month': 4,
                'selected_day': 29, 'selected_equipment': '13 (3호기)', 'thresholds': dict(D.DEFAULT_THRESHOLDS),
                'threshold_history': [], 'theme': 'light', 'anomaly_view': 'daily',
                'priority_seconds': 60, 'priority_events': 3}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.query_params.get('view') in ('ai-notifications', 'sensor-insights') and not st.session_state.get('_notification_link_applied'):
        st.session_state.update(active_tab='이상탐지', nav_page='이상탐지', inspection_mode='확인할 정보', _notification_link_applied=True)
    if st.session_state.get('inspection_mode') == 'AI 알림 후보':
        st.session_state['inspection_mode'] = '확인할 정보'
    pending = st.session_state.pop('_pending_tab', None)
    if pending in PAGES:
        st.session_state['active_tab'] = pending
        st.session_state['nav_page'] = pending
    if st.session_state['active_tab'] not in PAGES:
        st.session_state['active_tab'] = PAGES[0]


def navigate(page):
    st.session_state['_pending_tab'] = page
    st.rerun()


def apply_style():
    dark = st.session_state['theme'] == 'dark'
    bg, panel, ink, muted, border = ('#101821', '#192431', '#edf4f7', '#b8c6d4', '#344457') if dark else ('#f6f8fb', '#ffffff', '#182739', '#536477', '#dfe6ee')
    st.markdown(f'''<style>
:root{{--mg-bg:{bg};--mg-panel:{panel};--mg-text:{ink};--mg-text-muted:{muted};--mg-border:{border}}}
.stApp{{background:{bg};color:{ink}}}
.block-container{{max-width:1440px;padding:2rem 2.5rem 3rem}}
[data-testid="stHeader"]{{background:transparent}}
h1,h2,h3,p,label,[data-testid="stMarkdownContainer"]{{color:{ink}}}
[data-testid="stCaptionContainer"] p{{color:{muted};font-size:14px;line-height:1.6}}
h1{{font-size:30px!important;letter-spacing:-.7px}}h2{{font-size:23px!important}}h3{{font-size:19px!important}}
[data-testid="stVerticalBlockBorderWrapper"]>div{{background:{panel};border-color:{border}!important;border-radius:14px}}
[data-testid="stMetricValue"]{{font-size:28px;color:{ink}}}
[data-testid="stMetricLabel"]{{font-size:15px}}
[data-testid="stButton"] button{{min-height:42px;border-radius:9px;font-size:15px}}
button[kind="secondary"],button[kind="secondaryFormSubmit"]{{background:{panel};color:{ink};border:1px solid {border}}}
button[kind="secondary"] *,button[kind="secondaryFormSubmit"] *{{color:{ink}!important}}
[data-testid="stBaseButton-primary"],button[kind="primary"]{{background:{TEAL};border-color:{TEAL};color:white}}
[data-testid="stBaseButton-primary"] *,button[kind="primary"] *{{color:white!important}}
[data-testid="stRadio"] label{{padding:7px 12px;border-radius:8px}}
.brand{{display:flex;align-items:center;gap:12px;font-weight:800;font-size:25px;color:{ink}}}
.brand-mark{{width:36px;height:36px;border-radius:11px;background:{TEAL};color:white;display:grid;place-items:center;font-size:22px}}
.context-line{{color:{muted};font-size:14px;margin:6px 0 18px}}
.scope-badge{{background:{panel};border:1px solid {border};border-radius:20px;padding:6px 12px;display:inline-block;color:{muted};font-size:13px}}
@media(max-width:700px){{.block-container{{padding:1.1rem .9rem 2rem}}h1{{font-size:25px!important}}[data-testid="stRadio"] label{{padding:5px 6px}}}}
</style>''', unsafe_allow_html=True)


def header():
    left, right = st.columns([5, 2])
    with left:
        st.markdown('<div class="brand"><span class="brand-mark">M</span>마가렛트</div>', unsafe_allow_html=True)
    with right:
        with st.expander('화면 설정 · 도움말'):
            st.session_state.setdefault('dashboard_theme_mode', st.session_state['theme'])
            st.radio('화면 모드', ['light', 'dark'], format_func=lambda x: '밝은 화면' if x == 'light' else '어두운 화면',
                     key='dashboard_theme_mode', horizontal=True,
                     on_change=lambda: st.session_state.update(theme=st.session_state['dashboard_theme_mode']))
            if st.button('처음 사용 가이드', use_container_width=True):
                navigate('시작 가이드')
            st.caption('요금 180원/kWh · 탄소 0.424 kgCO₂/kWh를 적용한 추정값입니다.')
    st.markdown('<div class="context-line">과거 데이터 체험 · 실시간 수집 아님 · 자료 기준 2025.04.30 00:00</div>', unsafe_allow_html=True)
    st.session_state.setdefault('nav_page', st.session_state['active_tab'])
    st.radio('작업 선택', PAGES, key='nav_page', format_func=PAGE_LABELS.get,
             horizontal=True, label_visibility='collapsed', on_change=lambda: st.session_state.update(active_tab=st.session_state['nav_page']))
    st.divider()


def current_date():
    ss = st.session_state
    return date(ss['selected_year'], ss['selected_month'], ss['selected_day'])


def context_filters():
    ss = st.session_state
    months = [f'{y:04d}-{m:02d}' for y, m in D.available_months()]
    value = f"{ss['selected_year']:04d}-{ss['selected_month']:02d}"
    if value not in months:
        value = '2025-04'
    ss.setdefault('context_month', value)
    ss.setdefault('context_day', ss['selected_day'])
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 4], vertical_alignment='bottom')
        with c1:
            selected = st.selectbox('조회 월', months, key='context_month', format_func=lambda s: f'{s[:4]}년 {int(s[5:])}월')
            ss['selected_year'], ss['selected_month'] = map(int, selected.split('-'))
        days = list(range(1, calendar.monthrange(ss['selected_year'], ss['selected_month'])[1] + 1))
        # Write the value explicitly so changing month length preserves the chosen day.
        ss['context_day'] = min(int(ss.get('context_day', ss['selected_day'])), days[-1])
        with c2:
            ss['selected_day'] = st.selectbox('조회 일', days, key='context_day', format_func=lambda d: f'{d}일')
        with c3:
            day = current_date()
            if day.isoformat() in D.sensor_dates():
                st.markdown(f'**{day:%Y.%m.%d} · 센서 실측 자료**')
            elif day >= date(2025, 4, 30):
                st.markdown(f'**{day:%Y.%m.%d} · 전력 예측 기간**')
                st.caption('전력·요금·탄소 예측 제공 · 센서 실측 없음')
    ss['view_date'] = current_date().isoformat()
    ss['view_range'] = ss.get('anomaly_view', 'daily')


def summary_for(day=None, equipment=None):
    return get_alert_summary((day or current_date()).isoformat(), dict(st.session_state['thresholds']), equipment, cutoff=D.NOW)


def duration_text(seconds):
    if seconds is None:
        return '확인 불가'
    seconds = int(round(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return ' '.join(f'{v}{unit}' for v, unit in [(h, '시간'), (m, '분'), (s, '초')] if v) or '0초'


def is_review(row):
    seconds = max((row.get(k, {}).get('sample_duration_seconds') or 0) for k in ('rule', 'ai'))
    events = max((row.get(k, {}).get('event_count') or 0) for k in ('rule', 'ai'))
    return seconds >= st.session_state['priority_seconds'] or events >= st.session_state['priority_events']


def has_record(row):
    return any((row.get(k, {}).get('affected_hours') or row.get(k, {}).get('flagged_records') or row.get(k, {}).get('event_count') or 0) > 0 for k in ('rule', 'ai'))


def alert_rows(summary):
    rows = []
    for row in summary.get('equipment_rows', []):
        rule, ai = row.get('rule', {}), row.get('ai', {})
        status = '자료 없음' if not row.get('data_available') else '검토 우선' if is_review(row) else '기록 있음' if has_record(row) else '탐지 기록 없음'
        count = lambda x: f"{x['event_count']:,}구간" if x.get('event_count') is not None else f"{x.get('affected_hours', 0):,}시간대 · 구간 수 미확인"
        rows.append({'설비': row['equipment'], '표시 상태': status, '설정 기준 이탈': count(rule),
                     'AI 탐지': count(ai), 'AI 탐지 표본 누적': duration_text(ai.get('sample_duration_seconds')),
                     '_priority': 0 if status == '검토 우선' else 1 if status == '기록 있음' else 2,
                     '_records': ai.get('flagged_records') or 0})
    return sorted(rows, key=lambda r: (r['_priority'], -r['_records'], r['설비']))


def metric_card(col, label, value, caption=None):
    with col:
        with st.container(border=True):
            st.metric(label, value)
            if caption:
                st.caption(caption)


def finish_chart(fig, ytitle, height=340):
    dark = st.session_state['theme'] == 'dark'
    ink = '#edf4f7' if dark else '#182739'
    fig.update_layout(template='plotly_dark' if dark else 'plotly_white', height=height,
                      margin=dict(l=20, r=20, t=36, b=36), font=dict(size=14, color=ink),
                      legend=dict(orientation='h', y=1.13, x=0, font=dict(color=ink)), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', yaxis_title=ytitle, hovermode='x unified')
    fig.update_xaxes(color=ink)
    fig.update_yaxes(color=ink)
    fig.update_annotations(font_color=ink)
    return fig


def render_overview():
    ss = st.session_state
    st.title('현황·예측')
    context_filters()
    day = current_date()
    hv = D.month_hourly_values(day.year, day.month)
    power = D.month_power_estimate(day.year, day.month)
    previous = pd.Timestamp(day.year, day.month, 1) - pd.offsets.MonthBegin(1)
    prev_power = D.month_power_estimate(previous.year, previous.month)
    days = calendar.monthrange(day.year, day.month)[1]
    prev_days = calendar.monthrange(previous.year, previous.month)[1]
    daily_delta = (power / days / (prev_power / prev_days) - 1) * 100 if power is not None and prev_power else None
    compare = f'일평균 전월 대비 {daily_delta:+.3f}%' if daily_delta is not None else '전월 비교 자료 없음'
    cols = st.columns(3)
    title = f'{day.month}월 합계' if (day.year, day.month) < (2025, 4) else f'{day.month}월 예상 합계'
    metric_card(cols[0], title + ' 요금', D.won(power * 180) if power is not None else '자료 없음', compare)
    metric_card(cols[1], title + ' 전력량', D.kwh(power) if power is not None else '자료 없음')
    metric_card(cols[2], title + ' 탄소배출량', D.kgco2(power * .424) if power is not None else '자료 없음')
    actual_n = int(hv.actual.notna().sum())
    predicted_n = int((hv.actual.isna() & hv.pred.notna()).sum())
    filled_n = int(hv.value.isna().sum())
    if power is None:
        st.caption(f'선택월 {len(hv):,}시간의 자료가 없어 월 합계를 계산할 수 없어요.')
    else:
        st.caption(f'월 합계 구성: 실측 {actual_n:,}시간 + AI 예측 {predicted_n:,}시간' + (f' + 평균 보정 {filled_n:,}시간 (AI 예측 아님)' if filled_n else '') + f' · 전체 {len(hv):,}시간')
    with st.container(border=True):
        top, controls = st.columns([3, 4], vertical_alignment='center')
        top.subheader('실측과 예측 흐름')
        metric = controls.radio('확인할 지표', ['전력량', '전기요금', '탄소배출량'], horizontal=True, label_visibility='collapsed')
        period = st.radio('그래프 기간', ['선택일 · 시간별', '선택월 · 일별'], horizontal=True)
        factor, unit = {'전력량': (1, 'kWh'), '전기요금': (180, '원'), '탄소배출량': (.424, 'kgCO₂')}[metric]
        if period.startswith('선택일'):
            frame = D.hourly_power_series(day)
            x, actual, pred = frame.hour, frame.actual, frame.pred
            xtitle = '시각 (한국시간)'
        else:
            frame = D.monthly_power_series(day.year, day.month)
            x, actual, pred = frame.day, frame.power_actual, frame.power_pred
            xtitle = f'{day.month}월 일자'
        fig = go.Figure()
        fig.add_scatter(x=x, y=actual * factor, name='실측', mode='lines+markers', line=dict(color=TEAL, width=3), connectgaps=False)
        fig.add_scatter(x=x, y=pred * factor, name='AI 예측', mode='lines', line=dict(color=BLUE, width=3, dash='dash'), connectgaps=False)
        fig.update_xaxes(title=xtitle)
        values = np.r_[np.asarray(actual, float), np.asarray(pred, float)] * factor
        values = values[np.isfinite(values)]
        if len(values):
            center = values.mean()
            fig.update_yaxes(range=[min(values.min(), center * .95), max(values.max(), center * 1.05)])
        else:
            st.info('선택한 기간에는 이 그래프를 그릴 자료가 없어요. 다른 날짜를 선택해 주세요.')
        st.plotly_chart(finish_chart(fig, unit), use_container_width=True, config={'displayModeBar': False})
        st.caption('실측: 4월 29일까지 · AI 예측: 4월 30일부터')
    from monitoring.sensor_insights import get_sensor_insights
    from ui_sensor_insights import render_equipment_overview
    insights = get_sensor_insights(day.isoformat(), policy=ss.get('sensor_insight_policy'), cutoff=D.NOW)
    st.subheader('설비 한 번에 보기')
    render_equipment_overview(insights, key_prefix='overview')
    if st.button('설비별 근거 확인 →', type='primary', use_container_width=True):
        ss['inspection_mode'] = '확인할 정보'
        navigate('이상탐지')
    render_validation()


def render_validation():
    with st.expander('AI 예측은 얼마나 맞았나요? · 검증 결과 보기'):
        st.write('**5월 10일 00시 ~ 5월 31일 00시, 505시간**의 저장된 예측과 정답을 같은 구간에서 비교했어요.')
        folder = D.FORECAST_DIR
        try:
            selected = pd.read_csv(folder / 'may_comparison_selected.csv', parse_dates=['id'])
            baseline = pd.read_csv(folder / 'may_comparison_mean.csv', parse_dates=['id'])
            joined = selected.merge(baseline[['id', 'hourly_pow_predicted']], on='id', suffixes=('', '_mean'))
            actual, pred, mean = (joined[c] / 1000 for c in ['hourly_pow_actual', 'hourly_pow_predicted', 'hourly_pow_predicted_mean'])
            score = float((actual - pred).abs().mean())
            base_score = float((actual - mean).abs().mean())
            st.write(f'Extra Trees 평균 절대 오차 **{score:.4f} kWh/시간** · 평균값 기준선 **{base_score:.4f} kWh/시간**')
            st.caption('이 자료에서는 단순 평균 대비 개선이 확인되지 않았어요. 이 수치는 5월 전체 또는 새로운 현장 데이터의 성능을 보장하지 않아요.')
            fig = go.Figure()
            for values, label, color, dash in [(actual, '검증 정답', TEAL, 'solid'), (pred, 'Extra Trees 예측', BLUE, 'solid'), (mean, '평균값 기준선', '#8b95a7', 'dash')]:
                fig.add_scatter(x=joined.id, y=values, name=label, line=dict(color=color, dash=dash), mode='lines')
            st.plotly_chart(finish_chart(fig, 'kWh'), use_container_width=True, config={'displayModeBar': False})
            st.caption('단위 안내: 제공된 정답 수치를 Wh로 해석하여 kWh로 환산한 값이에요. 제공자 단위 확인이 필요하며, 4·5월은 이미 살펴본 자료의 재평가예요.')
        except (OSError, ValueError, KeyError):
            st.info('검증 결과 파일을 준비하지 못했어요. 예측 조회와 별도로 확인이 필요해요.')


def render_anomaly():
    ss = st.session_state
    st.title('설비 점검')
    context_filters()
    inspection_mode = st.radio('확인할 내용', ['확인할 정보', '전체 탐지 기록'], key='inspection_mode', horizontal=True)
    if inspection_mode == '확인할 정보':
        from ui_sensor_insights import render_sensor_insights
        result = render_sensor_insights(current_date().isoformat())
        if result and result.get('data_available'):
            selected = ss.get('sensor_insight_detail_equipment')
            if selected in D.EQUIPMENT_NAMES:
                render_sensor_detail(selected, summary_for(equipment=selected), insights=result)
        elif st.button('마지막 실측일 · 4월 29일 보기', type='primary'):
            ss.update(selected_year=2025, selected_month=4, selected_day=29)
            ss.pop('context_month', None); ss.pop('context_day', None)
            st.rerun()
        return
    summary = summary_for()
    if not summary.get('data_available'):
        st.info('이 날짜에는 전압·역률 실측이 없어서 상태를 판단할 수 없어요.')
        if st.button('마지막 실측일 · 4월 29일 보기', type='primary'):
            ss.update(selected_year=2025, selected_month=4, selected_day=29)
            ss.pop('context_month', None); ss.pop('context_day', None)
            st.rerun()
        return
    rows = alert_rows(summary)
    rule, ai = summary['rule'], summary['ai']
    cols = st.columns(3)
    metric_card(cols[0], '먼저 살펴볼 설비', f"{sum(r['_priority'] == 0 for r in rows)}대 / 13대", '표시 기준에 따른 검토 순서 · 고장 판정 아님')
    event_value = lambda group: f"{group['event_count']:,}구간" if group.get('event_count') is not None else f"{group.get('affected_hours', 0):,}시간대"
    metric_card(cols[1], '설정 기준 이탈', event_value(rule), f"기록이 있는 설비 {rule['equipment_count']}대 · 항목별 중복 포함")
    metric_card(cols[2], 'AI 패턴 탐지', event_value(ai), f"5초 표본 {ai.get('flagged_records') or 0:,}개 · 실제 고장 수가 아니에요")
    st.caption(summary.get('note', ''))
    with st.expander('목록 표시 기준 · 센서 기준 설정'):
        st.write('표시 기준은 검토 순서만 바꿔요. 원기록이나 AI 판정을 지우거나 바꾸지 않아요.')
        a, b = st.columns(2)
        seconds = a.number_input('설정 기준 또는 AI의 표본 누적 시간(초) 이상', min_value=5, max_value=86400, value=int(ss['priority_seconds']), step=5)
        events = b.number_input('설정 기준 또는 AI의 발생 구간 수 이상', min_value=1, max_value=1000, value=int(ss['priority_events']))
        if (seconds, events) != (ss['priority_seconds'], ss['priority_events']):
            ss['priority_seconds'], ss['priority_events'] = seconds, events
            st.rerun()
        st.caption('둘 중 하나를 만족하면 검토 우선으로 보여요. 1분·3구간은 체험용 표시 기본값이며 설비의 안전 기준이 아니에요.')
        with st.form('sensor_thresholds'):
            a, b, c = st.columns(3)
            phase = a.number_input('상전압 하한(V)', min_value=1.0, max_value=1000.0, value=float(ss['thresholds']['phase_voltage']))
            line = b.number_input('선간전압 하한(V)', min_value=1.0, max_value=2000.0, value=float(ss['thresholds']['line_voltage']))
            pf = c.number_input('역률 하한(%)', min_value=1.0, max_value=100.0, value=float(ss['thresholds']['power_factor']))
            st.caption('교육 자료의 기본값을 사용해요. 실제 설비의 허용 기준은 현장 사양을 확인해야 해요. 이 설정은 AI 기준을 바꾸지 않아요.')
            st.caption('기준을 바꾸면 시간별 요약으로 재계산해요. 이때 설정 기준의 실제 구간 수·지속 시간은 확인 불가로 표시돼요.')
            if st.form_submit_button('설정 기준 적용'):
                ss['thresholds'] = dict(phase_voltage=phase, line_voltage=line, power_factor=pf)
                st.rerun()
    with st.container(border=True):
        left, right = st.columns([5, 4])
        with left:
            st.subheader('설비별 기록')
            view = st.radio('표시할 설비', ['전체 설비', '기록 있는 설비', '검토 우선'], horizontal=True)
            visible = [r for r in rows if view == '전체 설비' or (view == '검토 우선' and r['_priority'] == 0) or (view == '기록 있는 설비' and r['_priority'] <= 1)]
            if visible:
                table = pd.DataFrame(visible).drop(columns=['_priority', '_records'])
                st.caption('행을 선택하면 오른쪽에 해당 설비의 기록이 나와요.')
                selection = st.dataframe(table, hide_index=True, use_container_width=True, height=300,
                                         on_select='rerun', selection_mode='single-row', key='equipment_records')
                indices = selection.selection.rows
                selected_key = (current_date().isoformat(), view, tuple(indices))
                if indices and ss.get('_last_row_selection') != selected_key and indices[0] < len(visible):
                    ss['selected_equipment'] = visible[indices[0]]['설비']
                    ss['detail_equipment'] = ss['selected_equipment']
                    ss['_last_row_selection'] = selected_key
            else:
                st.info('현재 표시 기준에 맞는 설비가 없어요. 전체 기록은 계속 보관돼요.')
        with right:
            st.subheader('선택 설비')
            names = D.EQUIPMENT_NAMES
            ss.setdefault('detail_equipment', ss['selected_equipment'] if ss['selected_equipment'] in names else names[0])
            st.selectbox('근거를 확인할 설비', names, key='detail_equipment', on_change=lambda: ss.update(selected_equipment=ss['detail_equipment']))
            ss['selected_equipment'] = ss['detail_equipment']
            eq = ss['selected_equipment']
            detail = summary_for(equipment=eq)
            st.markdown(f'**{eq} · {current_date():%Y.%m.%d}**')
            for key, label in [('rule', '설정 기준'), ('ai', 'AI 탐지')]:
                item = detail[key]
                st.write(f"{label}: **{event_value(item)}** · 표본 누적 **{duration_text(item.get('sample_duration_seconds'))}**")
                if item.get('longest_duration_seconds') is not None:
                    st.caption(f"최장 연속 구간 {duration_text(item['longest_duration_seconds'])}")
            st.caption('표본 누적: 탐지된 5초 기록의 합 · 고장 시간 아님')
            def request_report():
                ss['report_start_requested'] = True
                ss['_pending_tab'] = '에이전트'
            st.button('이 설비로 보고서 시작 →', type='primary', use_container_width=True, on_click=request_report)
    render_sensor_detail(ss['selected_equipment'], detail)


def render_sensor_detail(equipment, detail, insights=None):
    with st.container(border=True):
        st.subheader(f'{equipment} · 시간별 최저값 참고')
        family = st.radio('측정 항목', ['상전압', '선간전압', '역률'], horizontal=True)
        settings = {'상전압': (['voltageR', 'voltageS', 'voltageT'], ['R상', 'S상', 'T상'], 'phase_voltage', 'V'),
                    '선간전압': (['voltageRS', 'voltageST', 'voltageTR'], ['RS', 'ST', 'TR'], 'line_voltage', 'V'),
                    '역률': (['powerFactorR', 'powerFactorS', 'powerFactorT'], ['R상', 'S상', 'T상'], 'power_factor', '%')}
        columns, names, key, unit = settings[family]
        raw, source = D.equipment_daily(current_date())
        data = raw[raw.equipment == equipment] if source == 'sensor' else raw.iloc[:0]
        fig = go.Figure()
        for col, label, color in zip(columns, names, [TEAL, BLUE, '#b96b21']):
            fig.add_scatter(x=data.hour, y=data[col], name=label, mode='lines+markers', line=dict(color=color, width=2), connectgaps=False)
        if insights and family in ('상전압', '역률'):
            policy = insights['policy']
            if family == '상전압':
                lines = [(policy['voltage_low'], '지속 하한'), (policy['voltage_severe_low'], '큰 이탈 하한')]
            else:
                lines = [(policy['pf_low'], '지속 저하'), (policy['pf_sharp_low'], '급락 값 조건')]
            for value, label in lines:
                fig.add_hline(y=value, line_color='#b96b21', line_dash='dash', annotation_text=label, annotation_position='bottom right')
        else:
            fig.add_hline(y=st.session_state['thresholds'][key], line_color='#b96b21', line_dash='dash', annotation_text='설정 하한', annotation_position='bottom right')
        fig.update_xaxes(title='시각 (한국시간)')
        st.plotly_chart(finish_chart(fig, unit), use_container_width=True, config={'displayModeBar': False})
        st.caption('시간별 최저값 · 실제 지속 시간은 개별 기록에서 확인')
    with st.expander('발생 기록 전체 보기 · 기준 이탈과 AI 탐지 구분'):
        records = []
        for source_key, label in [('rule_events', '설정 기준'), ('ai_events', 'AI 탐지')]:
            for event in detail.get(source_key, []):
                records.append({'종류': label, '항목': event.get('item'), '시작': event.get('timestamp'), '끝': event.get('end'),
                                '연속 탐지 시간': duration_text(event.get('duration_seconds')),
                                '측정값/점수': event.get('value'), '단위': event.get('unit'),
                                '집계': '5초 기록 기반' if event.get('duration_seconds') is not None else '관측된 시간대 · 실제 지속 미확인'})
        if records:
            st.caption(f'전체 {len(records):,}행 · 센서별 구간은 서로 겹칠 수 있어요.')
            st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)
        else:
            st.write('해당 설비의 탐지 기록이 없어요.')


def render_report():
    st.title('보고서 도우미')
    context_filters()
    from agent.agent_ui import render_agent_tab
    render_agent_tab(st.session_state)


def main():
    init_state()
    apply_style()
    header()
    page = st.session_state['active_tab']
    if page == '시작 가이드':
        from ui_guide import render_guide
        render_guide()
    elif page == '현황·예측':
        render_overview()
    elif page == '이상탐지':
        render_anomaly()
    else:
        render_report()
    st.divider()
    st.caption('마가렛트 · 박정열 · 김민찬 · 최경철')


if __name__ == '__main__':
    main()
