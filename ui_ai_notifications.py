"""Review preserved AI records and preview notification policies without sending alerts."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from monitoring.ai_notifications import DEFAULT_POLICY, get_ai_notifications, normalize_policy

_REASON_LABELS = {
    "duration": "연속 시간 충족", "sustained": "연속 시간 충족",
    "persistence": "연속 시간 충족", "repeat": "반복 횟수 충족",
    "repetition": "반복 횟수 충족", "repeated": "반복 횟수 충족",
    "score_below_cutoff": "추가 점수 기준 미달", "below_score_cutoff": "추가 점수 기준 미달",
    "no_qualifying_samples": "추가 점수 기준 미달", "below_score_margin": "추가 점수 기준 미달",
    "insufficient_duration_and_repetition": "지속·반복 조건 미달",
    "duration_and_repeat_not_met": "지속·반복 조건 미달",
    "insufficient_evidence": "지속·반복 조건 미달",
    "not_notified": "지속·반복 조건 미달",
}


def _reason(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return " · ".join(dict.fromkeys(_reason(item) for item in value)) or "확인 불가"
    if value is None:
        return "확인 불가"
    return _REASON_LABELS.get(str(value), str(value))


def _time(value: Any) -> str:
    if value is None or pd.isna(value):
        return "확인 불가"
    return pd.Timestamp(value).strftime("%m-%d %H:%M:%S")


def _duration(value: Any) -> str:
    if value is None or pd.isna(value):
        return "확인 불가"
    seconds = int(value)
    minutes, remainder = divmod(seconds, 60)
    if minutes and remainder:
        return f"{minutes}분 {remainder}초"
    return f"{minutes}분" if minutes else f"{seconds}초"


def _count(value: Any, unit: str) -> str:
    return "확인 불가" if value is None or pd.isna(value) else f"{int(value):,}{unit}"


def log_table(rows: list[dict]) -> pd.DataFrame:
    """Keep identifiers and full timestamps in exports so records remain traceable."""
    return pd.DataFrame([{
        "원기록 ID": row.get("event_id"), "설비": row.get("equipment"),
        "원래 탐지 시작": row.get("start"), "종료(미포함)": row.get("end"),
        "연속 탐지 시간(초)": row.get("duration_seconds"), "AI 표본 수": row.get("ai_records"),
        "최대 AI 점수": row.get("score_max"), "기존 모델 기준": row.get("base_threshold"),
        "추가 점수 기준 통과 표본": row.get("qualified_records"),
        "점수 통과 최장 연속(초)": row.get("max_qualifying_duration_seconds"),
        "분류": "알림 후보에 포함" if row.get("notification_candidate") else "기록만 보관",
        "기록만 남긴 이유": "—" if row.get("notification_candidate") else _reason(row.get("log_only_reason")),
    } for row in rows])


def notification_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "알림 묶음 ID": row.get("notification_id"), "설비": row.get("equipment"),
        "첫 조건 충족 시각": row.get("first_trigger_at"), "마지막 조건 충족 시각": row.get("last_trigger_at"),
        "알림 근거": _reason(row.get("reasons")), "묶은 조건 충족 수": row.get("candidate_count"),
        "포함 원기록 수": len(set(row.get("source_event_ids", []))),
        "포함 원기록 ID": " | ".join(map(str, row.get("source_event_ids", []))),
        "조건 충족 때 최대 점수": row.get("score_max_at_triggers"),
    } for row in rows])


def candidate_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "설비": row.get("equipment"), "조건 충족 시각": row.get("trigger_at"),
        "알림 근거": _reason(row.get("reasons")), "통과 연속 시간(초)": row.get("duration_seconds"),
        "반복 원구간 수": row.get("repeat_event_count"), "그 시점 점수": row.get("score_at_trigger"),
        "추가 점수 기준": row.get("score_cutoff"),
        "근거 원기록 ID": " | ".join(map(str, row.get("supporting_event_ids", []))),
    } for row in rows])


def equipment_table(rows: list[dict]) -> pd.DataFrame:
    result = []
    for row in rows:
        available = row.get("data_available", False)
        result.append({
            "설비": row.get("equipment"),
            "원래 AI 탐지": _count(row.get("original_event_count") if available else None, "구간"),
            "알림 후보": _count(row.get("notification_count") if available else None, "묶음"),
            "후보 포함 원기록": _count(row.get("notified_original_event_count") if available else None, "구간"),
            "기록만 보관": _count(row.get("log_only_event_count") if available else None, "구간"),
        })
    return pd.DataFrame(result)


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def _show_table(frame: pd.DataFrame, *, height: int = 320) -> None:
    st.dataframe(frame, hide_index=True, use_container_width=True, height=height)


def _controls() -> dict:
    try:
        current = normalize_policy(st.session_state.get("ai_notification_policy", DEFAULT_POLICY))
    except (ValueError, TypeError):
        current = normalize_policy(DEFAULT_POLICY)
        st.info("알림 비교 설정을 기본값으로 되돌렸어요. 원래 AI 기록은 그대로예요.")
    st.session_state["ai_notification_policy"] = dict(current)
    with st.expander("알림 후보 기준 조정 · 체험용 설정"):
        st.write("기존 AI가 탐지한 표본 중 추가 점수 기준을 통과한 표본으로 지속·반복을 확인해요.")
        with st.form("ai_notification_policy_form"):
            a, b = st.columns(2)
            score_options = sorted(set([0.0, 0.02, 0.05, float(current["score_margin"])]))
            margin = a.selectbox("기존 모델 점수 기준에 더할 값", score_options,
                                 index=score_options.index(float(current["score_margin"])),
                                 format_func=lambda number: f"+{number:.2f}", key="ai_policy_score_margin")
            duration_options = sorted(set([60, 180, 300, int(current["duration_seconds"])]))
            duration = b.selectbox("연속 시간 조건", duration_options,
                                   index=duration_options.index(int(current["duration_seconds"])),
                                   format_func=_duration, key="ai_policy_duration")
            a, b, c = st.columns(3)
            repeats = a.number_input("반복할 원래 AI 구간 수", min_value=2, max_value=20,
                                     value=int(current["repeat_count"]), step=1, key="ai_policy_repeat_count")
            window = b.number_input("반복을 살펴볼 시간(분)", min_value=1, max_value=120,
                                    value=int(current["repeat_window_minutes"]), step=1, key="ai_policy_repeat_window")
            merge = c.number_input("같은 설비의 알림을 묶을 시간(분)", min_value=0, max_value=120,
                                   value=int(current["merge_window_minutes"]), step=1, key="ai_policy_merge_window")
            st.caption("+0.02는 점수에 0.02를 더한다는 뜻이에요. 2% 증가나 전압·역률의 변화율이 아니며, 점수는 고장 확률이 아니에요.")
            if st.form_submit_button("이 설정으로 후보 다시 보기", type="primary"):
                try:
                    current = normalize_policy(dict(score_margin=float(margin), duration_seconds=int(duration),
                        repeat_count=int(repeats), repeat_window_minutes=int(window), merge_window_minutes=int(merge)))
                    st.session_state["ai_notification_policy"] = dict(current)
                except (ValueError, TypeError):
                    st.warning("선택한 값으로 비교할 수 없어요. 설정 범위를 다시 확인해주세요.")
        st.caption("이 화면에서 비교한 설정만 적용돼요. 전압·역률 규칙, 원래 AI 판정과 학습 모델은 바뀌지 않아요.")
    return dict(current)


def _start_report(equipment: str) -> None:
    st.session_state.update(selected_equipment=equipment, detail_equipment=equipment,
                            report_start_requested=True, _pending_tab="에이전트")


def render_ai_notifications(day: Any, equipment: str | None = None) -> dict | None:
    """Render an auditable, local-only preview. Return the exact result for a caller."""
    st.subheader("AI 기록과 알림 후보")
    st.write("AI가 찾은 기록은 모두 보관하고, 점수·지속·반복 조건을 만족한 경우를 알림 후보로 따로 확인해요.")
    policy = _controls()
    st.caption(
        f"비교 중: 기존 모델 기준 +{policy['score_margin']:.2f} · "
        f"{_duration(policy['duration_seconds'])} 연속 또는 {policy['repeat_window_minutes']}분 안에 "
        f"서로 다른 원래 AI 구간 {policy['repeat_count']}회 · 같은 설비 {policy['merge_window_minutes']}분 묶음"
    )
    period = st.radio("확인할 기간", ["선택일", "전체 실측기간"], horizontal=True, key="ai_policy_period")
    selected_day = None if period == "전체 실측기간" else day
    # The overview always keeps all equipment; filtering applies only to detail.
    selected_equipment = None
    period_label = "2025-04-01 ~ 2025-04-29" if selected_day is None else pd.Timestamp(day).strftime("%Y-%m-%d")
    st.caption(f"조회 범위: {period_label} · {selected_equipment or '전체 설비'} · 과거 5초 실측 자료 · 시각은 한국시간")
    try:
        result = get_ai_notifications(day=selected_day, equipment=selected_equipment, policy=policy)
    except (OSError, ValueError, KeyError):
        st.warning("알림 후보 비교 자료를 확인할 수 없어요. 원래 AI 탐지 기록은 위에서 확인할 수 있어요.")
        return None
    if result.get("status") != "ok" or not result.get("data_available"):
        st.info("이 범위의 5초 실측 자료 또는 AI 점수 비교 자료가 없어 알림 후보를 판단할 수 없어요. 확인 가능한 실측 기간은 2025년 4월 1~29일이에요. 자료 없음은 정상이나 알림 0건을 뜻하지 않아요.")
        return result

    summary = result.get("summary", {})
    a, b = st.columns(2)
    a.metric("원래 AI 탐지", _count(summary.get("original_event_count"), "구간"))
    b.metric("알림 후보", _count(summary.get("notification_count"), "묶음"))
    a, b = st.columns(2)
    a.metric("후보에 포함된 원기록", _count(summary.get("notified_original_event_count"), "구간"))
    b.metric("기록만 보관", _count(summary.get("log_only_event_count"), "구간"))
    st.caption("원래 AI 탐지 구간 = 후보에 포함된 원기록 + 기록만 보관한 원기록이에요. 반복 판정에 함께 쓰인 원기록도 포함하며, 같은 원기록은 한 번만 세요. 알림 묶음 수는 원기록 수와 달라요.")
    sustained = summary.get("sustained_candidate_count")
    repeated = summary.get("repeated_candidate_count")
    if sustained is not None and repeated is not None:
        st.write(f"후보가 생긴 이유: **연속 시간 충족 {sustained:,}회 · 반복 횟수 충족 {repeated:,}회**")
        st.caption("이유별 조건 충족 횟수는 중복될 수 있으며 알림 묶음 수와 달라요.")
        if sustained == 0:
            longest = max((row.get("max_qualifying_duration_seconds") or 0 for row in result.get("all_logs", [])), default=0)
            st.caption(f"추가 점수 기준을 통과한 최장 연속 구간은 {_duration(longest)}예요. 설정한 {_duration(policy['duration_seconds'])} 연속 조건을 만족한 후보가 없다는 뜻이며 정상 판정은 아니에요.")
    st.info("체험용 알림 후보예요. 실제 알림은 발송하지 않으며, 후보가 없다고 설비가 정상이라는 뜻은 아니에요. 점수는 고장 확률이 아니고 전압·역률의 규칙 기준을 사용한 알림도 아니에요.")

    per_equipment = result.get("per_equipment", [])
    if per_equipment:
        st.markdown("**설비 한 번에 보기**")
        _show_table(equipment_table(per_equipment), height=493 if len(per_equipment) > 10 else 220)

    detail_names = [row["equipment"] for row in per_equipment if row.get("equipment")]
    detail_options = ["전체 설비", *dict.fromkeys(detail_names)]
    detail_key = "ai_notification_detail_equipment"
    if st.session_state.get(detail_key) not in detail_options:
        st.session_state[detail_key] = equipment if equipment in detail_names else "전체 설비"
    chosen_equipment = st.selectbox("상세 기록을 확인할 설비", detail_options, key=detail_key)
    st.caption("위의 전체 현황과 설비 목록은 유지하고, 아래 기록만 선택한 설비로 좁혀 보여요.")
    detail_equipment = None if chosen_equipment == "전체 설비" else chosen_equipment
    def detail_rows(key):
        return [row for row in result.get(key, []) if detail_equipment is None or row.get("equipment") == detail_equipment]
    notifications, all_logs, log_only = (detail_rows(key) for key in ("notifications", "all_logs", "log_only"))
    candidates = detail_rows("trigger_candidates")
    if detail_equipment:
        st.markdown(f"**{detail_equipment} · 원래 AI 탐지 {len(all_logs):,}구간 · 알림 후보 {len(notifications):,}묶음**")
        if selected_day is not None:
            st.button("이 설비로 보고서 시작 →", key="ai_policy_start_report",
                      on_click=_start_report, args=(detail_equipment,))
        else:
            st.caption("보고서를 만들려면 ‘선택일’로 돌아가 날짜와 해당 설비의 기록을 확인하세요.")
    notification_view, log_view, excluded_view = st.tabs(["알림 후보", "모든 AI 기록", "기록만 남긴 이유"])
    suffix = "전체기간" if selected_day is None else pd.Timestamp(day).strftime("%Y%m%d")
    if detail_equipment:
        suffix += "_설비" + detail_equipment.split(" ", 1)[0]
    with notification_view:
        st.caption("첫 조건 충족 시각부터 후보가 돼요. 처음 이상을 탐지한 시각으로 소급하지 않으며, 같은 설비의 마지막 후보 이후 설정한 시간 동안 새 후보가 없으면 다음 알림 묶음으로 나눠요. 묶음의 시작부터 끝까지 이상이 계속됐다는 뜻은 아니에요.")
        if notifications:
            frame = notification_table(notifications)
            _show_table(frame)
            st.download_button("알림 후보 CSV 내려받기", _csv(frame), f"AI_알림후보_{suffix}.csv", "text/csv", key="ai_policy_download_notifications")
            with st.expander("각 후보가 조건을 충족한 근거"):
                _show_table(candidate_table(candidates))
                st.caption("반복은 추가 점수 조건을 통과한 서로 다른 원래 AI 구간을 세요. 한 구간 안에 표본이 여러 개 있어도 여러 회로 세지 않아요.")
        else:
            st.write("현재 조건을 만족한 알림 후보가 없어요. 탐지 기록은 ‘모든 AI 기록’에서 그대로 확인할 수 있어요.")
    with log_view:
        st.caption("새 기준으로 제외된 기록도 지우지 않아요. 아래 시간은 원래 AI 구간의 시간이며, 점수 조건을 통과한 연속 시간과 구분해요.")
        if all_logs:
            frame = log_table(all_logs)
            _show_table(frame)
            st.download_button("원래 AI 기록 CSV 내려받기", _csv(frame), f"AI_원기록_{suffix}.csv", "text/csv", key="ai_policy_download_logs")
        else:
            st.write("이 실측 범위에서 기존 모델이 표시한 AI 탐지 구간이 없어요.")
    with excluded_view:
        st.caption("추가 점수 기준을 통과하지 못했거나, 지속·반복 조건을 만족하지 않은 기록이에요. 정상으로 재분류한 것이 아니에요.")
        if log_only:
            frame = log_table(log_only)
            _show_table(frame)
            st.download_button("기록만 보관한 구간 CSV 내려받기", _csv(frame), f"AI_기록만보관_{suffix}.csv", "text/csv", key="ai_policy_download_log_only")
        else:
            st.write("현재 범위에는 기록만 보관하는 원래 AI 구간이 없어요.")
    with st.expander("지속 시간·반복 횟수는 어떻게 계산하나요?"):
        st.write("같은 설비에서 5초 표본이 실제로 이어지고 추가 점수 기준을 계속 통과할 때만 연속 시간을 늘려요. 점수가 낮아지거나 실측이 끊기면 그 연속 구간은 끝나요.")
        st.write("연속 시간과 반복 횟수는 둘 중 하나를 만족하면 돼요. 반복은 서로 다른 원래 AI 구간이 정한 시간창 안에서 추가 점수 조건을 충족했는지 확인해요.")
        st.caption("기존 구간의 최대 점수와 전체 지속 시간을 곱하거나, 시간별 요약값으로 연속 시간을 추정하지 않아요. 알림 후보를 줄이는 것은 모델 정확도나 고장 탐지 성능의 개선을 뜻하지 않아요.")
    return result
