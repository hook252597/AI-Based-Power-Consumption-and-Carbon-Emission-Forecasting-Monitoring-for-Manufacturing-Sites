"""Readable measurement evidence alongside preserved AI detections; no alert sending."""
from __future__ import annotations

from copy import deepcopy
from typing import Any
import re

import pandas as pd
import streamlit as st

import dashboard_data as D

from monitoring.alert_service import EQUIPMENT_LIST
from monitoring.sensor_insights import DEFAULT_POLICY, get_sensor_insights, normalize_policy

POLICY_KEY = "sensor_insight_policy"
DETAIL_KEY = "sensor_insight_detail_equipment"
_ITEM_NAMES = {"voltageR": "R상 전압", "voltageS": "S상 전압", "voltageT": "T상 전압",
               "powerFactorR": "R상 역률", "powerFactorS": "S상 역률", "powerFactorT": "T상 역률"}
_KIND_NAMES = {"voltage_severe": "큰 전압 이탈", "voltage_sustained": "전압 이탈 지속",
               "pf_sharp": "역률 급락", "pf_sustained": "낮은 역률 지속"}


def _number(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "확인 불가"
    return f"{float(value):,.{digits}f}"


def _duration(value: Any) -> str:
    if value is None or pd.isna(value):
        return "확인 불가"
    seconds = int(value)
    minutes, remainder = divmod(seconds, 60)
    return f"{minutes}분 {remainder}초" if minutes and remainder else f"{minutes}분" if minutes else f"{seconds}초"


def _count(value: Any, unit: str) -> str:
    return "확인 불가" if value is None or pd.isna(value) else f"{int(value):,}{unit}"


def _range(low: Any, high: Any, unit: str) -> str:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "자료 없음"
    return f"{_number(low)}–{_number(high)} {unit}"


def _unit(item: str) -> str:
    return "V" if str(item).startswith("voltage") else "%"


def _direction(value: Any) -> str:
    return {"low": "낮아짐", "high": "높아짐", "down": "낮아짐", "up": "높아짐"}.get(str(value), "")


def _condition(kind: str, direction: Any, policy: dict) -> str:
    if kind == "voltage_severe":
        return f"상전압 {policy['voltage_severe_low']:g}V 미만 또는 {policy['voltage_severe_high']:g}V 초과"
    if kind == "voltage_sustained":
        return f"상전압 {policy['voltage_low']:g}V 미만 또는 {policy['voltage_high']:g}V 초과가 {_duration(policy['voltage_duration_seconds'])} 지속"
    if kind == "pf_sharp":
        return f"역률 {policy['pf_sharp_low']:g}% 미만이고 급락 구간 시작 전 {_duration(policy['baseline_seconds'])} 중앙값(구간 동안 고정)보다 {policy['pf_drop_pp']:g}%p 이상 하락"
    if kind == "pf_sustained":
        return f"역률 {policy['pf_low']:g}% 미만이 {_duration(policy['pf_duration_seconds'])} 지속"
    return "확인 조건 미상"


def policy_table(policy: dict) -> pd.DataFrame:
    return pd.DataFrame([{"확인 항목": name, "측정값 조건": _condition(kind, None, policy)}
                         for kind, name in _KIND_NAMES.items()])


def _apply_policy() -> None:
    values = {key: st.session_state[f"sensor_cfg_{key}"] for key in DEFAULT_POLICY}
    try:
        st.session_state[POLICY_KEY] = normalize_policy(values)
        st.session_state.pop("_sensor_policy_error", None)
    except (ValueError, TypeError):
        st.session_state["_sensor_policy_error"] = "기준을 적용하지 못했어요. 전압은 큰 이탈 하한 < 지속 하한 < 지속 상한 < 큰 이탈 상한 순서여야 하고, 역률 급락 기준은 지속 기준보다 낮아야 해요. 시간은 5초 단위로 입력해주세요."


def _reset_policy() -> None:
    st.session_state[POLICY_KEY] = normalize_policy(DEFAULT_POLICY)
    for key, value in DEFAULT_POLICY.items():
        st.session_state[f"sensor_cfg_{key}"] = value
    st.session_state.pop("_sensor_policy_error", None)


def _controls(*, policy_editable: bool = True) -> dict:
    try:
        policy = normalize_policy(st.session_state.get(POLICY_KEY, DEFAULT_POLICY))
    except (ValueError, TypeError):
        policy = normalize_policy(DEFAULT_POLICY)
    st.session_state[POLICY_KEY] = dict(policy)
    for key, value in policy.items():
        st.session_state.setdefault(f"sensor_cfg_{key}", value)
    with st.expander("확인 기준 조정"):
        if not policy_editable:
            st.caption("기본 기준 결과 · 기준 변경은 원본 자료 연결 후 가능")
        with st.form("sensor_insight_policy_form"):
            st.markdown("**상전압 · R/S/T상**")
            left, right = st.columns(2)
            left.number_input("큰 전압 이탈 하한(V)", min_value=1.0, step=1.0, key="sensor_cfg_voltage_severe_low", disabled=not policy_editable)
            right.number_input("큰 전압 이탈 상한(V)", min_value=1.0, step=1.0, key="sensor_cfg_voltage_severe_high", disabled=not policy_editable)
            left, middle, right = st.columns(3)
            left.number_input("지속 이탈 하한(V)", min_value=1.0, step=1.0, key="sensor_cfg_voltage_low", disabled=not policy_editable)
            middle.number_input("지속 이탈 상한(V)", min_value=1.0, step=1.0, key="sensor_cfg_voltage_high", disabled=not policy_editable)
            right.number_input("전압 이탈 지속 시간(초)", min_value=5, step=5, key="sensor_cfg_voltage_duration_seconds", disabled=not policy_editable)
            st.markdown("**역률 · R/S/T상**")
            left, middle, right = st.columns(3)
            left.number_input("급락을 확인할 역률(%) 미만", min_value=0.0, max_value=100.0, step=1.0, key="sensor_cfg_pf_sharp_low", disabled=not policy_editable)
            middle.number_input("평소보다 하락한 폭(%p) 이상", min_value=0.1, max_value=100.0, step=1.0, key="sensor_cfg_pf_drop_pp", help="역률이 90%에서 70%가 되면 20%p 하락입니다.", disabled=not policy_editable)
            right.number_input("평소와 비교할 직전 구간(초)", min_value=5, step=5, key="sensor_cfg_baseline_seconds", help="급락 구간 시작 전 연속 기록의 중앙값입니다. 이 기준선은 구간 동안 고정됩니다.", disabled=not policy_editable)
            left, middle, right = st.columns(3)
            left.number_input("낮은 역률 기준(%) 미만", min_value=0.0, max_value=100.0, step=1.0, key="sensor_cfg_pf_low", disabled=not policy_editable)
            middle.number_input("낮은 역률 지속 시간(초)", min_value=5, step=5, key="sensor_cfg_pf_duration_seconds", disabled=not policy_editable)
            right.number_input("같은 유형의 정보를 묶을 간격(분)", min_value=0, step=1, key="sensor_cfg_merge_minutes", disabled=not policy_editable)
            left, right = st.columns(2)
            left.form_submit_button("이 기준 적용", type="primary", on_click=_apply_policy, use_container_width=True, disabled=not policy_editable)
            right.form_submit_button("과제용 기본값 복원", on_click=_reset_policy, use_container_width=True)
        st.caption("현재 화면에 적용 · AI 모델과 원기록 유지")
    if st.session_state.get("_sensor_policy_error"):
        st.warning(st.session_state["_sensor_policy_error"])
    return dict(st.session_state[POLICY_KEY])


def _select_equipment(name: str) -> None:
    st.session_state.update(selected_equipment=name, detail_equipment=name, sensor_insight_detail_equipment=name)
    if st.session_state.get("active_tab") == "현황·예측":
        st.session_state.update(_pending_tab="이상탐지", inspection_mode="확인할 정보")


def _detail_changed() -> None:
    name = st.session_state.get(DETAIL_KEY)
    if name and name != "전체 설비":
        st.session_state.update(selected_equipment=name, detail_equipment=name)


def render_equipment_overview(result: dict, key_prefix: str = "sensor_overview") -> None:
    """Keep every registered machine visible, even when there are no observations."""
    by_name = {row.get("equipment"): row for row in result.get("per_equipment", [])}
    for offset in range(0, len(EQUIPMENT_LIST), 4):
        columns = st.columns(4, gap="small")
        for column, (code, name) in zip(columns, EQUIPMENT_LIST[offset:offset + 4]):
            row = by_name.get(name, {})
            available = bool(result.get("data_available", True)) and bool(row.get("data_available"))
            with column:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    if available:
                        st.caption(f"전압 {_range(row.get('voltage_min'), row.get('voltage_max'), 'V')}  \n역률 {_range(row.get('pf_min'), row.get('pf_max'), '%')}  \nAI 원기록 {_count(row.get('original_event_count'), '구간')}")
                        if row.get("group_count") is None:
                            st.write("확인 정보 집계 불가")
                        elif row["group_count"]:
                            st.write(f"**확인할 정보 {int(row['group_count']):,}묶음**")
                        else:
                            st.write("추가 확인 정보 없음")
                    else:
                        st.write("실측 자료 없음")
                    st.button("보기 →", key=f"{key_prefix}_equipment_{code}",
                              on_click=_select_equipment, args=(name,), use_container_width=True,
                              help=f"{name}의 확인 정보와 원래 AI 기록을 봐요.")
    st.caption("선택일 R/S/T상 전체 범위 · 정보 없음은 정상 판정 아님")


def _log_id(row: dict) -> str:
    return str(row.get("event_id", row.get("id", "")))


def unlinked_ai_logs(result: dict) -> list[dict]:
    explicit = result.get("log_only")
    if explicit is not None:
        return list(explicit)
    related = {str(identity) for event in result.get("events", []) for identity in event.get("source_ai_event_ids", [])}
    return [row for row in result.get("all_ai_logs", []) if _log_id(row) not in related]


def groups_table(groups: list[dict], events: list[dict], policy: dict) -> pd.DataFrame:
    event_by_id = {str(row.get("id", row.get("event_id"))): row for row in events}
    rows = []
    for group in groups:
        members = [event_by_id[str(identity)] for identity in group.get("event_ids", []) if str(identity) in event_by_id]
        members.sort(key=lambda row: str(row.get("trigger_at", "")))
        first = members[0] if members else {}
        item, kind = group.get("item", ""), group.get("kind", "")
        first_value = group.get("first_value_at_trigger", first.get("value_at_trigger"))
        rows.append({"설비": group.get("equipment"), "항목": _ITEM_NAMES.get(item, item),
            "확인 내용": _KIND_NAMES.get(kind, kind) + (" · " + _direction(group.get("direction")) if _direction(group.get("direction")) else ""),
            "적용 조건": _condition(kind, group.get("direction"), policy),
            "첫 확인 값": f"{_number(first_value, 2)} {_unit(item)}", "구간 대표값": f"{_number(group.get('extreme_value'), 2)} {_unit(item)}",
            "첫 확인 시각": group.get("first_trigger_at"), "마지막 확인 시각": group.get("last_trigger_at"),
            "발생 구간 수": group.get("event_count"),
            "연결 AI 원구간 수": len({str(i) for event in members for i in event.get("source_ai_event_ids", [])}),
            "묶음 ID": group.get("id", group.get("group_id")),
            "근거 구간 ID": " | ".join(map(str, group.get("event_ids", [])))})
    return pd.DataFrame(rows)


def events_table(events: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"설비": row.get("equipment"), "항목": _ITEM_NAMES.get(row.get("item"), row.get("item")),
        "확인 내용": _KIND_NAMES.get(row.get("kind"), row.get("kind")), "방향": _direction(row.get("direction")),
        "센서 조건 시작": row.get("start"), "종료(미포함)": row.get("end"), "정보 확인 시각": row.get("trigger_at"),
        "그 시점 값": row.get("value_at_trigger"), "단위": _unit(row.get("item", "")),
        "대표값": row.get("extreme_value"), "센서 조건 지속(초)": row.get("duration_seconds"),
        "시작 전 기준선(구간 동안 고정)": row.get("baseline"), "역률 하락폭(%p)": row.get("drop_pp"),
        "연결 AI 원기록 ID": " | ".join(map(str, row.get("source_ai_event_ids", []))),
        "구간 ID": row.get("id", row.get("event_id"))} for row in events])


def ai_log_table(logs: list[dict], *, log_only: bool = False) -> pd.DataFrame:
    return pd.DataFrame([{"설비": row.get("equipment"), "탐지 시작": row.get("start", row.get("timestamp")),
        "종료(미포함)": row.get("end"), "AI 연속 탐지(초)": row.get("duration_seconds"),
        "AI 표본 수": row.get("ai_records", row.get("flagged_records")),
        "최대 AI 점수": row.get("score_max", row.get("score", row.get("value"))),
        **({"보관 이유": "측정값·지속 조건을 함께 충족한 정보에 연결되지 않음"} if log_only else {}),
        "원기록 ID": _log_id(row)} for row in logs])


def _show_download(frame: pd.DataFrame, label: str, name: str, key: str, *, compact: bool = False) -> None:
    hidden = [col for col in frame if col.endswith(" ID")]
    if compact:
        hidden += ["적용 조건", "연결 AI 원구간 수"]
    visible = frame.drop(columns=hidden, errors="ignore")
    st.dataframe(visible, hide_index=True, use_container_width=True, height=300)
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"), name, "text/csv", key=key)


def _start_report(name: str, result: dict) -> None:
    row = next((row for row in result.get("per_equipment", []) if row.get("equipment") == name), {})
    context = {"day": result.get("day"), "date": result.get("day"), "equipment": name,
        "policy": result.get("policy", {}), "summary": row,
        "source": result.get("source", {}), "coverage": result.get("coverage", {}),
        "data_available": result.get("data_available"), "status": result.get("status"),
        "events": [event for event in result.get("events", []) if event.get("equipment") == name],
        "groups": [group for group in result.get("groups", []) if group.get("equipment") == name]}
    st.session_state.update(selected_equipment=name, detail_equipment=name, report_start_requested=True,
        _pending_tab="에이전트", sensor_insight_report_context=deepcopy(context))


def _safe_note(value: Any) -> str:
    text = str(value or "실측 자료의 준비·검증 상태를 확인해주세요.")
    # Keep the diagnostic cause while hiding local filesystem locations.
    text = re.sub(r"[A-Za-z]:[\\/][^'\"\r\n]+", "[로컬 파일]", text)
    text = re.sub(r"\\\\[^'\"\r\n]+", "[로컬 파일]", text)
    return text


def _open_guide() -> None:
    st.session_state["_pending_tab"] = "시작 가이드"


def render_sensor_insights(day: Any, equipment: str | None = None) -> dict | None:
    try:
        requested_policy = normalize_policy(st.session_state.get(POLICY_KEY, DEFAULT_POLICY))
    except (ValueError, TypeError):
        requested_policy = normalize_policy(DEFAULT_POLICY)
    try:
        result = get_sensor_insights(day, policy=requested_policy, cutoff=D.NOW)
    except (OSError, ValueError, KeyError):
        result = None
    controls, criteria, guide = st.columns([2, 2, 1])
    with controls:
        policy = _controls(policy_editable=result.get("policy_editable", True) if result else False)
    with criteria:
        with st.expander("현재 적용 중인 기준", expanded=False):
            st.caption("220V 상전압 가정 · 과제용 점검 기준")
            st.table(policy_table(policy).set_index("확인 항목"))
            if policy['merge_minutes'] == 0:
                st.caption("반복 묶기 사용 안 함")
            else:
                st.caption(f"동일 유형 반복 묶기: {policy['merge_minutes']}분 미만")
    with guide:
        st.button("기준·집계 설명", key="sensor_insight_guide", on_click=_open_guide, use_container_width=True)
    day_text = pd.Timestamp(day).strftime("%Y-%m-%d")
    st.caption("과제용 점검 정보 · 실제 고장 판정 아님")
    if result is None:
        st.warning("확인 정보 자료를 준비하지 못했어요. 지금은 상태를 판단할 수 없어요.")
        render_equipment_overview({"per_equipment": []})
        return None
    if result.get("status") != "ok" or not result.get("data_available"):
        st.info("검증된 5초 실측 자료를 사용할 수 없어요. 자료 없음은 정상이나 0건을 뜻하지 않아요.")
        with st.expander("자료를 사용할 수 없는 이유"):
            st.caption(_safe_note(result.get("note")))
        render_equipment_overview(result)
        return result

    summary = result.get("summary", {})
    columns = st.columns(3)
    columns[0].metric("원래 AI 탐지", _count(summary.get("original_event_count"), "구간"), help="모델이 탐지한 연속 5초 기록을 설비별 구간으로 셉니다. AI 원기록은 모두 보존합니다.")
    columns[1].metric("확인할 정보", _count(summary.get("group_count"), "묶음"), help="AI 근거와 센서 조건을 충족한 기록을 항목별로 묶은 수입니다. 원래 AI 구간 수나 고장 건수와는 다릅니다.")
    columns[2].metric("정보가 있는 설비", f"{sum(bool(row.get('group_count')) for row in result.get('per_equipment', []))}대 / 13대")
    st.markdown("**전체 설비 한 번에 보기**")
    render_equipment_overview(result)

    names = [name for _, name in EQUIPMENT_LIST]
    options = ["전체 설비", *names]
    if st.session_state.get(DETAIL_KEY) not in options:
        st.session_state[DETAIL_KEY] = equipment if equipment in names else "전체 설비"
    selected = st.selectbox("상세 근거를 확인할 설비", options, key=DETAIL_KEY, on_change=_detail_changed, help="아래 상세 기록만 좁혀 봅니다. 전체 설비 카드는 그대로 유지됩니다.")
    def selected_rows(rows):
        return [row for row in rows if selected == "전체 설비" or row.get("equipment") == selected]
    groups = selected_rows(result.get("groups", []))
    events = selected_rows(result.get("events", []))
    all_logs = selected_rows(result.get("all_ai_logs", []))
    log_only = selected_rows(unlinked_ai_logs(result))
    if selected != "전체 설비":
        st.markdown(f"**{selected} · 확인 정보 {len(groups):,}묶음 · AI 원기록 {len(all_logs):,}구간**")
        st.button("이 설비로 보고서 시작 →", key="sensor_insight_start_report", on_click=_start_report, args=(selected, result))
    information, original, retained = st.tabs(["확인할 정보", "AI 원기록", "기록만 보관"])
    suffix = day_text.replace("-", "") + ("_전체" if selected == "전체 설비" else "_설비" + selected.split(" ", 1)[0])
    with information:
        if groups:
            _show_download(groups_table(groups, events, policy), "확인 정보 CSV 내려받기", f"확인정보_{suffix}.csv", "sensor_download_groups", compact=True)
            with st.expander("발생 구간별 측정값과 지속 시간"):
                _show_download(events_table(events), "발생 구간 CSV 내려받기", f"확인정보_근거_{suffix}.csv", "sensor_download_events")
        else:
            st.info("현재 기준의 추가 확인 정보가 없어요. AI 원기록은 아래 탭에서 확인하세요.")
    with original:
        st.caption("모델의 전체 탐지 기록 · 점수는 고장 확률이 아님")
        if all_logs:
            _show_download(ai_log_table(all_logs), "AI 원기록 CSV 내려받기", f"AI_원기록_{suffix}.csv", "sensor_download_ai_logs")
        else:
            st.write("이 실측 범위에는 원래 AI 탐지 구간이 없어요.")
    with retained:
        st.caption("확인 정보에 연결되지 않은 AI 기록 · 원기록 보존")
        if log_only:
            _show_download(ai_log_table(log_only, log_only=True), "기록만 보관한 구간 CSV 내려받기", f"AI_기록보관_{suffix}.csv", "sensor_download_retained")
        else:
            st.write("현재 범위에는 확인 정보와 연결되지 않은 AI 원기록이 없어요.")
    return result
