"""Optional Streamlit tab fragment; imports never start a page or server.

Pass data/reports/state to embed it in an existing app. The conversation core
is independent of Streamlit; legacy calendar keys are an opt-in UI adapter.
"""
from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from .agent_config import APP, PROJECT, DEFAULT_THRESHOLDS, load_settings
from .agent_core import run_turn
from .agent_data import DataService
from .agent_reports import ReportService
from .agent_tools import AgentTools


TODAY = "2025-05-31"
SECTIONS = ((1, "문서 정보"), (4, "1. 점검 개요"), (9, "2. 점검 대상 설비"),
            (15, "3. 전기 안전 점검 결과"), (28, "4. 전력 사용 현황"),
            (37, "5. 이상 발생 내역"), (44, "6. 조치 사항"),
            (52, "7. 종합 의견 및 후속 조치"), (59, "8. 확인"))


@st.cache_resource
def get_services():
    return DataService(PROJECT), ReportService(APP)


def _filled(value):
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _invalidate(state):
    state["report_draft"] = None
    state["report_context"] = None
    state.pop("export_bytes", None)
    state.pop("export_fingerprint", None)
    state.pop("agent_report_status", None)
    state.pop("agent_report_status_revision", None)
    state.pop("_last_measurement_query", None)
    state["_agent_report_snapshot"] = {}
    state["_agent_recent_fields"] = {}


def sync_legacy_calendar(state):
    """Optional bridge for hosts that use selected_year/month/day controls.

    A changed canonical view_date wins if both representations changed. This
    helper does not import a host dashboard or create its filter widgets.
    """
    last = state.get("_agent_seen_dashboard_date")
    try:
        legacy = date(int(state["selected_year"]), int(state["selected_month"]), int(state["selected_day"])).isoformat()
    except (KeyError, TypeError, ValueError):
        legacy = None
    requested = state.get("view_date")
    if not requested or (last and requested == last and legacy and legacy != last):
        requested = legacy or TODAY
    selected = date.fromisoformat(str(requested)[:10])
    state["view_date"] = selected.isoformat()
    state["selected_year"], state["selected_month"], state["selected_day"] = selected.year, selected.month, selected.day
    state["_agent_seen_dashboard_date"] = selected.isoformat()


def _init_state(state, data, app_dir=APP, *, legacy_calendar=False):
    """Initialize shared agent state, independently from any host navigation."""
    state["_agent_legacy_calendar"] = bool(legacy_calendar)
    if legacy_calendar:
        sync_legacy_calendar(state)
    state.setdefault("view_date", TODAY)
    defaults = {
        "selected_equipment": next(iter(getattr(data, "modules", [])), "12(4호기)"),
        "thresholds": dict(DEFAULT_THRESHOLDS), "threshold_history": [], "messages": [],
        "report_draft": None, "report_context": None, "pending_template": None,
        "trace": [], "forecast_month": "2025-05", "agent_template_panel_open": False,
    }
    for key, value in defaults.items():
        state.setdefault(key, copy.deepcopy(value))
    view_day = date.fromisoformat(str(state["view_date"])[:10])
    state["view_date"] = view_day.isoformat()
    if hasattr(data, "normalize_equipment"):
        normalized = data.normalize_equipment(state.get("selected_equipment"))
        if normalized:
            state["selected_equipment"] = normalized
    signature = (state["view_date"], state.get("selected_equipment"), _fingerprint(state["thresholds"]))
    previous = state.get("_agent_view_signature")
    if previous and previous != signature:
        draft = state.get("report_draft")
        if draft and (draft.get("date"), draft.get("equipment")) != signature[:2]:
            _invalidate(state)
    state["_agent_view_signature"] = signature
    if "template_asked" not in state:
        try:
            stored = json.loads((Path(app_dir) / "runtime/agent_state.json").read_text(encoding="utf-8"))
            state["template_asked"] = stored.get("template_asked") is True
        except (OSError, ValueError, AttributeError):
            state["template_asked"] = False


def _mark_template_asked(state, reports):
    state["template_asked"] = True
    path = reports.runtime / "agent_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({"template_asked": True}), encoding="utf-8")
    temp.replace(path)


def _opening_message(state, tools):
    if state.get("_agent_greeted"):
        return
    query = {"date": state["view_date"], "equipment": None}
    result = tools.execute("get_anomalies", query)
    state["_agent_anomaly_query"] = query
    equipment = state.get("selected_equipment") or "공장 전체"
    if result.get("data_available") is True:
        text = (f"선택일 {state['view_date']}의 공장 전체 기록을 확인했어요. "
                f"센서 기준 미달 {result.get('count', 0):,}구간, AI 탐지 {result.get('ai_count', 0):,}구간입니다. "
                f"현재 선택한 설비는 {equipment}예요. 어느 설비의 기록을 살펴볼까요?")
    else:
        text = (f"선택일 {state['view_date']}에는 센서 실측 기록이 없어 이상 여부를 판단할 수 없어요. "
                "4월 1–29일을 선택하면 전압·역률을 확인할 수 있고, 5월 10–31일의 전력 예측도 조회할 수 있어요.")
    state["messages"].append({"role": "assistant", "content": text, "mode": "summary"})
    state["_agent_greeted"] = True


def _snapshot(state):
    current = {r["name"]: r.get("value") for r in (state.get("report_draft") or {}).get("rows", [])}
    before = state.get("_agent_report_snapshot", {})
    now = time.time()
    recent = {key: stamp for key, stamp in state.get("_agent_recent_fields", {}).items() if now - stamp < 12}
    for key, value in current.items():
        if _filled(value) and value != before.get(key):
            recent[key] = now
    state["_agent_report_snapshot"] = current
    state["_agent_recent_fields"] = recent


def ordered_rows(rows, is_default=True):
    if not is_default:
        return list(rows)
    def position(pair):
        index, row = pair
        cell = str(row.get("cell", ""))
        match = re.fullmatch(r"([A-Z]+)(\d+)", cell)
        if match:
            return int(match.group(2)), match.group(1), index
        if row.get("name") == "이상 발생 내역":
            return 38, "A", index
        return 70, "A", index
    return [row for _, row in sorted(enumerate(rows), key=position)]


def checklist(rows):
    manual = [r for r in rows if r.get("source") == "user"]
    return manual, sum(_filled(r.get("value")) for r in manual)


def _checklist_panel(rows):
    manual, complete = checklist(rows)
    with st.expander(f"일일 점검 체크리스트  ·  {complete} / {len(manual)} 완료", expanded=False):
        required = [r for r in rows if r.get("required")]
        done = sum(_filled(r.get("value")) for r in required)
        st.progress(done / len(required) if required else 1.0, text=f"필수 항목 {done} / {len(required)} 완료")
        for row in manual:
            mark = "✓" if _filled(row.get("value")) else "□"
            required_tag = " · 필수" if row.get("required") else " · 선택"
            st.write(f"{mark} {row['name']}{required_tag}")


def _wants_template_panel(text):
    return "양식" in text and bool(re.search(r"바꿀|바꾸|바꿔|변경|등록|업로드|올릴|다른|새로", text)) and not bool(re.search(r"바꾸지\s*마|변경하지\s*마", text))


def _threshold_changes(before, after):
    descriptions = []
    for key, label, unit in (("phase_voltage", "상전압", "V"), ("line_voltage", "선간전압", "V"), ("power_factor", "역률", "%")):
        old, new = before.get(key), after.get(key)
        if old != new and old is not None and new is not None:
            descriptions.append(f"{label} {old:g}{unit} → {new:g}{unit}")
    return descriptions


def threshold_change_summary(result):
    """Card text comes only from returned values; missing counts remain unknown."""
    if result.get("error"):
        return {"error": str(result["error"])}
    changes = _threshold_changes(result.get("before", {}), result.get("after", {}))
    before, after = result.get("anomaly_count_before"), result.get("anomaly_count_after")
    valid = all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (before, after))
    available = valid and result.get("data_available") is not False
    return {
        "changes": changes,
        "counts": f"센서 기준 위반 구간 {before:,.0f} → {after:,.0f}개" if available else "선택일 실측 자료가 없어 재분석 건수를 확인할 수 없습니다.",
        "newly": result.get("newly_abnormal") if available else None,
        "removed": result.get("no_longer_abnormal") if available else None,
        "date": result.get("analysis_date"),
    }


def _threshold_change_card(state):
    entries = [entry for entry in state.get("trace", []) if entry.get("name") == "set_thresholds" and isinstance(entry.get("result"), dict)]
    if not entries:
        return
    summary = threshold_change_summary(entries[-1]["result"])
    with st.container(border=True):
        if summary.get("error"):
            st.error(summary["error"])
            return
        st.markdown("**센서 기준 변경 결과**")
        st.write(" · ".join(summary["changes"]) if summary["changes"] else "기준값은 그대로입니다.")
        st.write(summary["counts"])
        if summary["newly"]:
            st.caption("새로 포함된 설비: " + ", ".join(summary["newly"]))
        if summary["removed"]:
            st.caption("변경 기준에서 제외된 설비: " + ", ".join(summary["removed"]))
        st.caption("AI 판정 기준은 바뀌지 않습니다.")


def _threshold_controls(tools, state):
    with st.expander("센서 기준·변경 이력", expanded=False):
        current = state["thresholds"]
        st.caption(f"상전압 {current['phase_voltage']:g}V · 선간전압 {current['line_voltage']:g}V · 역률 {current['power_factor']:g}%")
        history = state.get("threshold_history", [])
        if not history:
            st.caption("아직 변경 이력이 없습니다.")
        for entry in reversed(history):
            if not isinstance(entry, dict):
                st.write(str(entry))
                continue
            changes = _threshold_changes(entry.get("before", {}), entry.get("after", {}))
            st.write(" · ".join(changes) if changes else str(entry.get("change", "기준 변경")))
            details = [str(entry.get(key, "")) for key in ("at", "reason") if entry.get(key)]
            st.caption(" · ".join(details))
        if st.button("센서 기준을 기본값으로 복원", key="agent_reset_thresholds", disabled=current == DEFAULT_THRESHOLDS, use_container_width=True):
            args = {**DEFAULT_THRESHOLDS, "reason": "사용자가 화면에서 기본 기준 복원"}
            result = tools.execute("set_thresholds", args)
            state["trace"] = [{"name": "set_thresholds", "args": args, "result": result}]
            if result.get("error"):
                st.error(result["error"])
            else:
                state.pop("export_bytes", None)
                state.pop("export_fingerprint", None)
                state.pop("_last_measurement_query", None)
                _snapshot(state)
                st.rerun()


def _respond(prompt, state, tools, settings):
    if _wants_template_panel(prompt):
        state["agent_template_panel_open"] = True
    result = run_turn(prompt, state, tools, api_key=settings.api_key or None, model=settings.model)
    state["trace"] = result.get("trace", [])
    state["last_mode"] = result.get("mode", "local")
    state["last_notice"] = result.get("error", "")
    if result.get("report_status"):
        state["agent_report_status"] = str(result["report_status"])
        state["agent_report_status_revision"] = _fingerprint(state.get("report_draft"))
    if state.get("_agent_legacy_calendar"):
        sync_legacy_calendar(state)
    # Core owns insertion; annotate the newest assistant turn without duplicating it.
    for message in reversed(state.get("messages", [])):
        if message.get("role") == "assistant":
            message["mode"] = state["last_mode"]
            break
    _snapshot(state)
    return result


def _template_controls(reports, state):
    if state.get("report_draft") and not state["template_asked"]:
        st.info("회사에서 따로 쓰는 보고서 양식이 있나요? 기본 양식으로 계속하거나 다른 양식을 등록해 주세요.")
        a, b = st.columns(2)
        if a.button("기본 양식으로 계속", key="agent_accept_default_template", use_container_width=True):
            _mark_template_asked(state, reports)
            st.rerun()
        if b.button("다른 양식 등록", key="agent_open_template", use_container_width=True):
            state["agent_template_panel_open"] = True
            st.rerun()
    expanded = state["agent_template_panel_open"] or bool(state.get("pending_template"))
    with st.expander("보고서 양식 변경·복원", expanded=expanded):
        uploaded = st.file_uploader("새 양식 (.xlsx / .docx)", type=["xlsx", "docx"], key="agent_template_upload")
        if uploaded and st.button("파일의 항목 확인", key="agent_parse_template"):
            try:
                state["pending_template"] = reports.parse_template(uploaded.name, uploaded.getvalue())
                state["agent_template_panel_open"] = True
                state.pop("agent_template_preview", None)
                st.rerun()
            except (ValueError, OSError, KeyError):
                st.error("양식 항목을 읽지 못했습니다. 아래에 항목을 직접 입력해 주세요.")
        manual = st.text_input("항목 직접 등록 · 쉼표로 구분", key="agent_manual_template", placeholder="점검 일시, 설비명, 점검자, 조치 내용, 종합 판정")
        st.caption("업로드 양식은 확인한 항목을 간단한 Excel 표로 내보냅니다. 원래 디자인은 복제하지 않습니다. HWP는 항목을 직접 등록해 주세요.")
        if manual and st.button("입력한 항목 확인", key="agent_parse_manual"):
            names = [name.strip() for name in re.split(r"[,，\n]", manual) if name.strip()]
            known = {f["name"].replace(" ", ""): f for f in reports.default["fields"]}
            fields = []
            for name in names:
                base = known.get(name.replace(" ", ""), {})
                fields.append({"name": base.get("name", name), "source": base.get("source", "user"), "required": base.get("required", False), "maps_to": base.get("maps_to")})
            state["pending_template"] = {"fields": fields, "layout": "table", "source_file": None}
            state["agent_template_panel_open"] = True
            state.pop("agent_template_preview", None)
            st.rerun()
        pending = state.get("pending_template")
        if pending:
            st.write("채우는 방식과 필수 여부를 확인해 주세요. 적용 버튼을 눌러야 저장됩니다.")
            frame = pd.DataFrame(pending.get("fields", []))
            edited = st.data_editor(frame, hide_index=True, use_container_width=True,
                                    column_config={"name": "항목", "source": st.column_config.SelectboxColumn("채우는 방식", options=["auto", "user", "blank"], required=True), "required": "필수"},
                                    disabled=[c for c in ("cell", "maps_to", "formula") if c in frame.columns], key="agent_template_preview")
            a, b = st.columns(2)
            if a.button("확인한 양식 적용", key="agent_apply_template", type="primary", use_container_width=True):
                try:
                    records = edited.where(pd.notna(edited), None).to_dict("records")
                    reports.save_report_template(records, pending.get("layout", "table"), pending.get("source_file"))
                    state["pending_template"] = None
                    state["agent_template_panel_open"] = False
                    _mark_template_asked(state, reports)
                    _invalidate(state)
                    st.rerun()
                except (ValueError, TypeError):
                    st.error("항목 이름·입력 방식·필수 여부를 확인해 주세요.")
            if b.button("변경 취소", key="agent_cancel_template", use_container_width=True):
                state["pending_template"] = None
                state["agent_template_panel_open"] = False
                st.rerun()
        if st.button("기본 보고서 양식 복원", key="agent_reset_template"):
            reports.reset_report_template()
            state["pending_template"] = None
            state["agent_template_panel_open"] = False
            _mark_template_asked(state, reports)
            _invalidate(state)
            st.rerun()


def _preview_html(rows, recent, default=True):
    parts, previous_section = ['<div class="agent-report-preview">'], None
    for row in ordered_rows(rows, default):
        match = re.search(r"(\d+)$", str(row.get("cell", "")))
        row_num = int(match.group(1)) if match else 38 if row.get("name") == "이상 발생 내역" else 70
        section = next((label for start, label in reversed(SECTIONS) if row_num >= start), "") if default else "보고서 항목"
        if section != previous_section:
            parts.append(f'<div class="agent-report-section">{html.escape(section)}</div>')
            previous_section = section
        source = row.get("source", "user")
        value = row.get("value")
        blank = source == "blank"
        text = "(공란 유지)" if blank else "(미입력)" if not _filled(value) else str(value)
        missing_class = " missing" if not _filled(value) else ""
        fresh_class = " recent" if row["name"] in recent else ""
        tag = {"auto": "자동", "user": "입력", "blank": "공란"}.get(source, "입력")
        name = html.escape(row["name"]) + (" <span class='required'>*</span>" if row.get("required") else "")
        parts.append(f'<div class="agent-report-row {source}{missing_class}{fresh_class}"><span class="agent-source">{tag}</span><span class="agent-field-name">{name}</span><span class="agent-field-value">{html.escape(text)}</span></div>')
    parts.append("</div>")
    return "".join(parts)


def _manual_inputs(tools, state, draft):
    manual = [r for r in draft.get("rows", []) if r.get("source") == "user"]
    with st.expander("보고서 입력 · 대화로 알려줘도 반영됩니다", expanded=bool(draft.get("missing_user_fields"))):
        include_optional = st.checkbox("선택 항목도 입력 · 설비 정보와 비고", key="agent_show_optional_fields")
        with st.form("agent_report_input_form"):
            inputs = {r["name"]: str(r.get("value")) if r.get("value") is not None else "" for r in manual}
            revision = _fingerprint([(r["name"], r.get("value")) for r in manual])[:12]
            def field(row):
                value = row.get("value")
                inputs[row["name"]] = st.text_input(row["name"] + (" *" if row.get("required") else ""), value=str(value) if value is not None else "", key="agent_report_input_" + revision + "_" + _fingerprint(row["name"])[:12])
            for row in manual:
                if row.get("required") or row["name"] == "조치 내용":
                    field(row)
            if include_optional:
                st.caption("선택 입력 · 설비 정보와 비고")
                for row in manual:
                    if not row.get("required") and row["name"] != "조치 내용":
                        field(row)
            submit = st.form_submit_button("입력 내용을 보고서에 반영", use_container_width=True)
        if submit:
            context = state.get("report_context") or (state["view_date"], state["selected_equipment"])
            result = tools.execute("create_report", {"date": context[0], "equipment": context[1], "extra_fields": inputs})
            if result.get("error"):
                st.error(result["error"])
            else:
                state.pop("export_bytes", None)
                _snapshot(state)
                st.rerun()


def _report_panel(tools, reports, state, template, rows):
    draft = state.get("report_draft")
    ready = bool(draft and draft.get("ready_to_export") and state.get("template_asked"))
    head, export = st.columns([3, 1.2])
    head.markdown(f"**{template['name']}**")
    if export.button("내보내기", disabled=not ready, key="agent_prepare_export", use_container_width=True):
        try:
            state["export_bytes"] = reports.export_xlsx(draft)
            state["export_fingerprint"] = _fingerprint(draft)
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("파일을 준비하지 못했습니다. 입력 항목과 양식을 확인해 주세요.")
    if ready and state.get("export_bytes") and state.get("export_fingerprint") == _fingerprint(draft):
        st.download_button("완성된 Excel 다운로드", data=state["export_bytes"], file_name=f"마가렛트_점검보고서_{draft['date']}_{draft['equipment']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="agent_download_report", use_container_width=True)
    st.caption("청록: 자동 자료 · 노랑: 사용자 입력 · * 필수 · 문서번호·결재·서명은 공란")
    _template_controls(reports, state)
    if not draft:
        st.caption("보고서 작성을 요청하면 선택한 날짜·설비의 자료로 아래 양식을 채웁니다.")
    else:
        st.caption(f"{draft['date']} · {draft['equipment']}")
        if state.get("agent_report_status") and state.get("agent_report_status_revision") == _fingerprint(draft):
            st.caption(state["agent_report_status"])
        if draft.get("missing_user_fields"):
            st.caption("필수 미입력: " + ", ".join(draft["missing_user_fields"]))
        if draft.get("missing_auto_fields"):
            with st.expander(f"자동 자료 없는 항목 · {len(draft['missing_auto_fields'])}개", expanded=False):
                st.write(", ".join(draft["missing_auto_fields"]))
                st.caption("없는 수치는 만들지 않습니다. 선택 항목은 빈칸으로 남기며 필수 자료가 없으면 내보낼 수 없습니다.")
        if draft.get("source_identity_mismatch"):
            st.error("자료의 출처가 서로 달라 내보낼 수 없습니다. 보고서를 다시 생성해 주세요.")
    with st.container(height=350, border=True, key="agent_report_preview"):
        st.markdown(_preview_html(rows, state.get("_agent_recent_fields", {}), template.get("is_default", False)), unsafe_allow_html=True)
    if draft:
        _manual_inputs(tools, state, draft)
        with st.expander("자료 출처와 작성 기준"):
            if draft.get("source_period"):
                st.write(draft["source_period"])
            if draft.get("data_id"):
                st.caption("자료 ID: " + str(draft["data_id"]))
            for note in draft.get("notes", []):
                st.caption(note)


def render_agent_tab(data=None, reports=None, state=None, *, legacy_calendar=False):
    """Render a tab fragment; the host owns navigation and page configuration.

    ``state`` may be a persistent dict or Streamlit session-state mapping. When
    omitted, st.session_state is used. A host's old calendar keys are touched
    only if ``legacy_calendar=True`` is explicitly passed.
    """
    if data is None and reports is None:
        data, reports = get_services()
    elif data is None:
        data = DataService(PROJECT)
    elif reports is None:
        reports = ReportService(APP)
    if state is None:
        state = st.session_state
    _init_state(state, data, reports.app_dir, legacy_calendar=legacy_calendar)
    tools = AgentTools(data, reports, state)
    settings = load_settings()
    _opening_message(state, tools)
    _snapshot(state)
    template = reports.get_report_template()
    draft = state.get("report_draft")
    rows = draft.get("rows", []) if draft else [{**field, "value": None} for field in template["fields"]]
    st.markdown("""<style>
      .agent-report-section{font-weight:700;padding:9px 7px;margin-top:7px;border-bottom:1px solid #c8d2d1}
      .agent-report-row{display:grid;grid-template-columns:38px minmax(110px,1fr) minmax(100px,1.1fr);gap:7px;padding:7px 8px;margin:3px 0;border-radius:5px;font-size:13px;color:#172a2a}
      .agent-report-row.auto{background:#eaf3f2}.agent-report-row.user{background:#fff9e0}.agent-report-row.blank{background:#f2f5f5}
      .agent-source{font-size:11px;color:#536262}.agent-field-value{white-space:pre-wrap;overflow-wrap:anywhere}.missing .agent-field-value{color:#7a8585}.required{color:#b33a25}
      @keyframes agent-field-flash{from{box-shadow:inset 0 0 0 2px #40a99b;background:#d9f2cf}to{box-shadow:none}}
      .agent-report-row.recent{animation:agent-field-flash 5s ease-out}.st-key-agent_chat_input{position:sticky;bottom:0;z-index:2}
      .st-key-agent_conversation [data-testid="stChatMessage"]{padding:.6rem}
      @media(max-width:1100px){
        .st-key-agent_workspace [data-testid="stHorizontalBlock"]:has(> :is([data-testid="column"],[data-testid="stColumn"]) .st-key-agent_conversation){flex-direction:column!important;align-items:stretch!important;gap:1.25rem!important}
        .st-key-agent_workspace [data-testid="stHorizontalBlock"]:has(> :is([data-testid="column"],[data-testid="stColumn"]) .st-key-agent_conversation)>:is([data-testid="column"],[data-testid="stColumn"]){width:100%!important;min-width:0!important;flex:1 1 auto!important}
      }
      </style>""", unsafe_allow_html=True)
    with st.container(key="agent_workspace"):
        left, right = st.columns([45, 55], gap="large")
    with left:
        _checklist_panel(rows)
        st.caption(f"{state['view_date']} · {state.get('selected_equipment') or '공장 전체'}")
        if not settings.api_key:
            st.caption("데모 모드 · API 키가 없어 실제 자료를 로컬 도구로 조회합니다.")
        elif state.get("last_mode") == "local":
            st.caption(f"데모 모드 · 마지막 API 응답을 완료하지 못했습니다. 다음 질문은 {settings.model}로 다시 시도합니다.")
        else:
            st.caption(f"{settings.model} · 질문에 맞춰 실제 자료를 조회합니다.")
        if settings.note:
            st.warning(settings.note)
        with st.container(height=270, border=True, key="agent_conversation"):
            for message in state["messages"]:
                role = message.get("role")
                if role not in {"user", "assistant"}:
                    continue
                spaces = st.columns([.10, .90] if role == "user" else [.90, .10])
                with spaces[1 if role == "user" else 0]:
                    with st.chat_message(role):
                        mode = message.get("mode")
                        if role == "assistant" and mode:
                            st.caption("실측 요약" if mode == "summary" else settings.model if mode == "openai" else "데모 모드")
                        st.markdown(str(message.get("content", "")))
            _threshold_change_card(state)
        prompt = st.chat_input("예: 4호기 전압 어때? · 그럼 보고서 작성해줘", key="agent_chat_input", max_chars=8000)
        if prompt:
            with st.spinner("자료를 확인하고 있어요…"):
                _respond(prompt, state, tools, settings)
            st.rerun()
        if st.button("선택한 설비로 보고서 시작", key="agent_create_report", use_container_width=True):
            result = tools.execute("create_report", {"date": state["view_date"], "equipment": state["selected_equipment"]})
            if result.get("error"):
                st.error(result["error"])
            else:
                _snapshot(state)
                st.rerun()
        _threshold_controls(tools, state)
        if state.get("trace"):
            with st.expander("이번 답변의 조회 근거"):
                st.json(state["trace"], expanded=False)
    with right:
        _report_panel(tools, reports, state, template, rows)
