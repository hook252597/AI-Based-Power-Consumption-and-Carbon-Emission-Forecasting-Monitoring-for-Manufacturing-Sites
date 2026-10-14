# -*- coding: utf-8 -*-
"""마가렛트 에이전트 — 대시보드 '에이전트' 탭 화면 (agent/agent.md 4~7절)

필요 패키지 : streamlit==1.39.0, openai, pandas, pyarrow, openpyxl, python-docx
               pip install streamlit==1.39.0 openai pandas pyarrow openpyxl python-docx==1.1.2
환경변수    : OPENAI_API_KEY (st.secrets 우선, 없으면 환경변수. 둘 다 없으면 데모 모드)
              OPENAI_MODEL   (선택, 기본 gpt-5-mini)
실행        : streamlit run dashboard_app.py  → '에이전트' 탭
              (데이터 캐시 준비: python tools/prepare_dashboard_cache.py)

대시보드와 공유하는 상태(st.session_state): view_date, selected_equipment, view_range(anomaly_view),
thresholds, threshold_history, report_draft. 에이전트 전용: messages, agent_history, agent_trace,
pending_template, template_upload_open, checklist.
"""
from __future__ import annotations

import html
import os
import re
import time

import pandas as pd
import streamlit as st

from .agent_core import DEFAULT_MODEL, opening_summary, run_turn
from .agent_tools import (DEFAULT_THRESHOLDS, TODAY, AgentTools, TemplateStore, export_xlsx, parse_docx,
                          parse_xlsx)


@st.cache_resource
def get_templates() -> TemplateStore:
    """앱 시작 시 기본 양식(BasicSafetyReportForm.xlsx)을 파싱해 report_template.json 으로 등록."""
    store = TemplateStore()
    store.ensure_default()
    return store


def load_settings() -> tuple[str, str]:
    """API 키·모델: st.secrets → 환경변수. 키는 코드에 적지 않는다."""
    key, model = "", ""
    try:
        key = str(st.secrets.get("OPENAI_API_KEY", "") or "")
        model = str(st.secrets.get("OPENAI_MODEL", "") or "")
    except Exception:  # noqa: BLE001 — secrets.toml이 없을 때
        pass
    return (key or os.environ.get("OPENAI_API_KEY", "")).strip(), (model or os.environ.get("OPENAI_MODEL", "") or DEFAULT_MODEL).strip()


CSS = """<style>
.ag-bubble{padding:8px 11px;border-radius:10px;margin:3px 0;font-size:14px;line-height:1.5;white-space:pre-wrap;
  overflow-wrap:anywhere;max-width:92%;display:inline-block;color:#111827}
.ag-user{background:#dcfce7;float:right;clear:both}
.ag-bot{background:#f1f5f9;float:left;clear:both}
.ag-row{overflow:hidden}
.ag-mode{font-size:11px;color:#6b7280;margin:0 2px}
.ag-section{font-weight:700;font-size:13px;margin:10px 0 3px;padding-bottom:3px;border-bottom:1px solid #d1d5db}
.ag-field{display:grid;grid-template-columns:40px minmax(120px,1fr) minmax(110px,1.2fr);gap:6px;padding:5px 8px;
  margin:2px 0;border-radius:5px;font-size:13px;color:#172a2a}
.ag-field.auto{background:#eaf3f2}.ag-field.user{background:#fff9e0}.ag-field.blank{background:#f2f5f5}
.ag-field.formula{background:#f2f5f5}
.ag-tag{font-size:11px;color:#536262}.ag-val{white-space:pre-wrap;overflow-wrap:anywhere}
.ag-field.empty .ag-val{color:#9ca3af}.ag-req{color:#b91c1c}
@keyframes ag-flash{from{box-shadow:inset 0 0 0 2px #16a34a;background:#d9f2cf}to{box-shadow:none}}
.ag-field.fresh{animation:ag-flash 4s ease-out}
.ag-card{border:1px solid #d1d5db;border-radius:8px;padding:8px 10px;margin:6px 0;font-size:13px;background:#fff;color:#111827}
</style>"""


def _filled(v) -> bool:
    return v not in (None, "", []) and not (isinstance(v, str) and not v.strip())


# -----------------------------------------------------------------------------
# 상태 준비
# -----------------------------------------------------------------------------
def _ensure_summary(state: dict, tools: AgentTools) -> None:
    """화면을 처음 열 때와 조회 날짜가 바뀔 때, 에이전트가 먼저 그 날의 이상 요약을 말한다."""
    if state.get("_agent_summary_date") == state["view_date"]:
        return
    lead = f"조회 날짜를 {state['view_date']}로 바꾸셨네요. " if state.get("_agent_summary_date") else ""
    state.setdefault("messages", []).append(
        {"role": "assistant", "content": lead + opening_summary(state, tools), "mode": "summary"})
    state["_agent_summary_date"] = state["view_date"]


def _template_rows(state: dict, template: dict) -> list[dict]:
    draft = state.get("report_draft")
    if draft and draft.get("template_name") == template.get("name"):
        return draft["rows"]
    return [{**f, "value": None} for f in template.get("fields", [])]


def _fresh_fields(state: dict, rows: list[dict]) -> set:
    """방금 채워진 항목(직전 화면 대비 값이 바뀐 항목)을 잠깐 강조한다."""
    current = {r["name"] + "|" + str(i): str(r.get("value")) for i, r in enumerate(rows)}
    previous = state.get("_agent_prev_values", {})
    now = time.time()
    fresh = {k: t for k, t in state.get("_agent_fresh", {}).items() if now - t < 4}
    for k, v in current.items():
        if previous and _filled(v) and v != "None" and previous.get(k) != v:
            fresh[k] = now
    state["_agent_prev_values"] = current
    state["_agent_fresh"] = fresh
    return set(fresh)


# -----------------------------------------------------------------------------
# 왼쪽: 체크리스트 · 대화 · 입력
# -----------------------------------------------------------------------------
def _checklist(state: dict, rows: list[dict]) -> None:
    manual = [r for r in rows if r.get("source") == "user"]
    done = sum(1 for r in manual if _filled(r.get("value")))
    state["checklist"] = {"done": done, "total": len(manual)}
    with st.expander(f"일일 점검 체크리스트  ·  {done} / {len(manual)} 완료", expanded=False):
        required = [r for r in manual if r.get("required")]
        if required:
            req_done = sum(1 for r in required if _filled(r.get("value")))
            st.progress(req_done / len(required), text=f"필수 {req_done} / {len(required)}")
        for r in manual:
            mark = "✅" if _filled(r.get("value")) else "⬜"
            st.markdown(f"{mark} {html.escape(r['name'])}{' · 필수' if r.get('required') else ''}")


def _conversation(state: dict, model: str, api_key: str) -> None:
    with st.container(height=330, border=True):
        for m in state.get("messages", []):
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            css = "ag-user" if role == "user" else "ag-bot"
            label = ""
            if role == "assistant":
                label = {"summary": "요약", "demo": "데모 모드", "openai": model}.get(m.get("mode"), "")
                label = f'<div class="ag-mode">{html.escape(label)}</div>' if label else ""
            st.markdown(f'<div class="ag-row">{label}<div class="ag-bubble {css}">{html.escape(str(m["content"]))}</div></div>',
                        unsafe_allow_html=True)
        _threshold_card(state)


def _threshold_card(state: dict) -> None:
    last = next((t for t in reversed(state.get("agent_trace") or []) if t["name"] == "set_thresholds"
                 and isinstance(t.get("result"), dict) and t["result"].get("changed")), None)
    if not last:
        return
    r = last["result"]
    counts = (f"이상 구간 {r['anomaly_count_before']}건 → {r['anomaly_count_after']}건" if r.get("data_available")
              else "선택일 실측이 없어 건수 변화는 확인할 수 없음")
    added = ", ".join(r.get("newly_abnormal") or []) or "없음"
    st.markdown(f'<div class="ag-card"><b>임계치 변경</b> · {html.escape(r["log_entry"]["change"])}<br>'
                f'{html.escape(counts)} ({html.escape(str(r.get("analysis_date")))})<br>'
                f'새로 포함된 설비: {html.escape(added)}<br>'
                f'<span style="color:#6b7280;font-size:12px">이상탐지 탭 가이드선에 반영 · AI 판정 기준은 그대로</span></div>',
                unsafe_allow_html=True)


def _template_upload(state: dict, tools: AgentTools) -> None:
    """'양식 바꿀게' 요청 시 대화창 안에 업로드 위젯을 띄우고, 확인받은 뒤에만 저장한다."""
    if not (state.get("template_upload_open") or state.get("pending_template")):
        return
    with st.container(border=True):
        st.markdown("**보고서 양식 등록**")
        file = st.file_uploader("양식 파일 (.xlsx / .docx / .hwp)", type=["xlsx", "docx", "hwp"], key="ag_template_file")
        if file is not None and state.get("_ag_parsed_file") != (file.name, file.size):
            state["_ag_parsed_file"] = (file.name, file.size)
            content = file.getvalue()
            if file.name.lower().endswith(".hwp"):
                state["messages"].append({"role": "assistant", "mode": "summary",
                                          "content": "HWP 파일은 읽을 수 없어요. 항목을 말로 불러주세요. 예: '발생일시, 설비, 원인, 조치, 확인자 이렇게 넣어줘'"})
            else:
                try:
                    parsed = parse_xlsx(content) if file.name.lower().endswith(".xlsx") else parse_docx(content)
                    if not parsed["fields"]:
                        raise ValueError("항목을 찾지 못했습니다.")
                    path = tools.templates.store_upload(file.name, content)
                    state["pending_template"] = {"fields": parsed["fields"], "source_file": file.name,
                                                 "source_path": str(path) if file.name.lower().endswith(".xlsx") else None,
                                                 "sheet": parsed.get("sheet"), "layout": "xlsx" if path.suffix == ".xlsx" else "table"}
                    groups = {s: [f["name"] for f in parsed["fields"] if f["source"] == s] for s in ("auto", "user", "blank")}
                    state["messages"].append({"role": "assistant", "mode": "summary", "content":
                        "양식을 확인했습니다. 항목을 이렇게 채우면 될까요?\n"
                        f"  자동으로 채울 항목 : {', '.join(groups['auto'][:12]) or '없음'}{' …' if len(groups['auto']) > 12 else ''}\n"
                        f"  알려주셔야 할 항목 : {', '.join(groups['user'][:12]) or '없음'}{' …' if len(groups['user']) > 12 else ''}\n"
                        f"  비워둘 항목        : {', '.join(groups['blank']) or '없음'}\n"
                        "아래 표에서 고친 뒤 '이렇게 적용'을 눌러 주세요."})
                except Exception:  # noqa: BLE001
                    state["messages"].append({"role": "assistant", "mode": "summary",
                                              "content": "양식 항목을 읽지 못했습니다. 항목을 말로 불러주시면 그대로 등록하겠습니다."})
            st.rerun()
        pending = state.get("pending_template")
        if pending:
            frame = pd.DataFrame([{"항목": f["name"], "채우는 방식": f["source"], "필수": bool(f.get("required"))}
                                  for f in pending["fields"]])
            edited = st.data_editor(frame, hide_index=True, use_container_width=True, key="ag_template_editor",
                                    column_config={"채우는 방식": st.column_config.SelectboxColumn(
                                        options=["auto", "user", "blank", "formula"], required=True)},
                                    disabled=["항목"], height=220)
            a, b = st.columns(2)
            if a.button("이렇게 적용", type="primary", use_container_width=True, key="ag_template_apply"):
                fields = [{**f, "source": row["채우는 방식"], "required": bool(row["필수"])}
                          for f, (_, row) in zip(pending["fields"], edited.iterrows())]
                r = tools.execute("save_report_template", {"fields": fields, "source_file": pending.get("source_file")})
                state["template_upload_open"] = False
                state.pop("_ag_parsed_file", None)
                state["messages"].append({"role": "assistant", "mode": "summary", "content":
                    r.get("error") or f"저장했습니다. 앞으로 보고서는 '{r['name']}' 양식으로 작성합니다. 기본 양식으로 돌아가시려면 말씀해 주세요."})
                st.rerun()
            if b.button("취소", use_container_width=True, key="ag_template_cancel"):
                state["pending_template"] = None
                state["template_upload_open"] = False
                state.pop("_ag_parsed_file", None)
                st.rerun()
        elif st.button("닫기", key="ag_template_close"):
            state["template_upload_open"] = False
            st.rerun()


def _threshold_controls(state: dict, tools: AgentTools) -> None:
    th = tools.thresholds
    history = state.get("threshold_history") or []
    with st.expander(f"임계치 · 상전압 {th['phase_voltage']:g}V / 선간 {th['line_voltage']:g}V / 역률 {th['power_factor']:g}%",
                     expanded=False):
        if not history:
            st.caption("변경 이력이 없습니다.")
        for e in reversed(history[-10:]):
            if isinstance(e, dict):
                st.markdown(f"- {html.escape(str(e.get('at', '')))} · {html.escape(str(e.get('change', '')))}"
                            + (f" — {html.escape(str(e['reason']))}" if e.get("reason") else ""))
        if st.button("기본값으로 되돌리기", disabled=th == DEFAULT_THRESHOLDS, use_container_width=True, key="ag_th_reset"):
            r = tools.execute("set_thresholds", {**DEFAULT_THRESHOLDS, "reason": "화면에서 기본값 복원"})
            state["agent_trace"] = [{"name": "set_thresholds", "args": DEFAULT_THRESHOLDS, "result": r}]
            state["messages"].append({"role": "assistant", "mode": "summary",
                                      "content": r.get("error") or "임계치를 기본값(상전압 210V · 선간전압 360V · 역률 85%)으로 되돌렸습니다."})
            st.rerun()


# -----------------------------------------------------------------------------
# 오른쪽: 보고서 미리보기 · 내보내기
# -----------------------------------------------------------------------------
def _value_text(row: dict) -> str:
    v = row.get("value")
    if row.get("source") == "blank":
        return "(비워 둠)"
    if row.get("kind") == "table" and isinstance(v, list):
        return "\n".join(" · ".join(str(x) for x in rec.values()) for rec in v) or "(미입력)"
    return str(v) if _filled(v) else "(미입력)"


def _report_panel(state: dict, tools: AgentTools) -> None:
    template = tools.templates.get()
    rows = _template_rows(state, template)
    draft = state.get("report_draft") if state.get("report_draft", {}) and \
        state["report_draft"].get("template_name") == template.get("name") else None
    ready = bool(draft and draft.get("ready_to_export"))
    head, btn = st.columns([3, 1.3], vertical_alignment="center")
    head.markdown(f"**{html.escape(template.get('name', '보고서'))}**"
                  + ("" if template.get("is_default") else " <span style='font-size:11px;color:#6b7280'>사용자 양식</span>"),
                  unsafe_allow_html=True)
    data = export_xlsx(draft, template) if ready else b""
    btn.download_button("내보내기", data=data or b" ", disabled=not ready, use_container_width=True, key="ag_export",
                        file_name=f"점검보고서_{draft['date']}_{re.sub(r'[^0-9A-Za-z가-힣]+', '_', draft['equipment'] or '전체')}.xlsx" if draft else "report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if draft and not tools.templates.template_asked():
        st.info("사내에서 따로 쓰시는 보고서 양식이 있나요? 없으면 기본 양식으로 작성합니다.")
        a, b = st.columns(2)
        if a.button("기본 양식으로 계속", use_container_width=True, key="ag_tpl_default"):
            tools.templates.mark_asked()
            state["_template_question_open"] = False
            st.rerun()
        if b.button("다른 양식 등록", use_container_width=True, key="ag_tpl_upload"):
            state["template_upload_open"] = True
            st.rerun()
    if draft:
        missing = draft.get("missing_user_fields") or []
        st.caption(f"{draft['date']} · {draft['equipment']}"
                   + (f" · 필수 미입력: {', '.join(missing)}" if missing else " · 필수 항목 완료"))
    else:
        st.caption("조치 내용을 말씀하시거나 '보고서 작성해줘'라고 하면 선택한 날짜·설비 자료로 채웁니다.")
    st.caption("청록 = 자동 채움 · 노랑 = 직접 입력 · 회색 = 비워 둠/엑셀 자동 계산 · * 필수")
    fresh = _fresh_fields(state, rows)
    parts, section = [], None
    for i, r in enumerate(rows):
        if r.get("section") and r["section"] != section:
            section = r["section"]
            parts.append(f'<div class="ag-section">{html.escape(section)}</div>')
        value = _value_text(r)
        cls = r.get("source", "user") + (" empty" if value in ("(미입력)", "(비워 둠)", "(엑셀에서 자동 계산)") else "") \
            + (" fresh" if f"{r['name']}|{i}" in fresh else "")
        tag = {"auto": "자동", "user": "입력", "blank": "공란", "formula": "계산"}.get(r.get("source"), "입력")
        req = " <span class='ag-req'>*</span>" if r.get("required") else ""
        parts.append(f'<div class="ag-field {cls}"><span class="ag-tag">{tag}</span>'
                     f'<span>{html.escape(r["name"])}{req}</span><span class="ag-val">{html.escape(value)}</span></div>')
    with st.container(height=430, border=True):
        st.markdown("".join(parts), unsafe_allow_html=True)
    if draft:
        _manual_form(state, tools, draft)
    with st.expander("보고서 양식 변경 · 복원", expanded=False):
        if st.button("다른 양식 등록하기", use_container_width=True, key="ag_tpl_open"):
            state["template_upload_open"] = True
            st.rerun()
        if st.button("기본 양식으로 복원", disabled=bool(template.get("is_default")), use_container_width=True, key="ag_tpl_reset"):
            r = tools.execute("reset_report_template", {})
            state["messages"].append({"role": "assistant", "mode": "summary",
                                      "content": r.get("error") or "기본 양식(제조현장 안전 점검 보고서)으로 되돌렸습니다."})
            st.rerun()


def _manual_form(state: dict, tools: AgentTools, draft: dict) -> None:
    manual = [r for r in draft["rows"] if r.get("source") == "user"]
    with st.expander("직접 입력 · 대화로 말씀하셔도 반영됩니다", expanded=False):
        with st.form("ag_manual_form"):
            values = {}
            for r in manual:
                values[r["name"]] = st.text_input(r["name"] + (" *" if r.get("required") else ""),
                                                  value=str(r.get("value") or ""), placeholder=r.get("hint") or "",
                                                  key=f"ag_in_{draft['date']}_{draft['equipment']}_{r['name']}")
            if st.form_submit_button("보고서에 반영", use_container_width=True):
                changed = {k: v for k, v in values.items() if v != str(next((r.get("value") or "") for r in manual if r["name"] == k))}
                if changed:
                    tools.execute("create_report", {"date": draft["date"], "equipment": draft["equipment"], "extra_fields": changed})
                st.rerun()


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------
def render_agent_tab(state: dict | None = None) -> None:
    """대시보드 본문 최상위에서 호출한다(내부에서 좌우 열을 만든다)."""
    state = st.session_state if state is None else state
    state.setdefault("view_date", TODAY.isoformat())
    tools = AgentTools(state, get_templates())
    api_key, model = load_settings()
    _ensure_summary(state, tools)
    template = tools.templates.get()
    rows = _template_rows(state, template)
    st.markdown(CSS, unsafe_allow_html=True)

    left, right = st.columns([45, 55], gap="large")
    with left:
        _checklist(state, rows)
        st.caption(f"{state['view_date']} · {state.get('selected_equipment') or '공장 전체'} · "
                   + (f"{model}" if api_key else "데모 모드 · API 키가 없어 같은 데이터를 규칙으로 조회합니다"))
        _conversation(state, model, api_key)
        _template_upload(state, tools)
        prompt = st.chat_input("예: 오늘 문제 있는 곳 있어? · 4호기 전압 어때? · 콘덴서 교체했어", key="ag_chat_input")
        if prompt:
            with st.spinner("자료를 확인하고 있어요…"):
                result = run_turn(prompt, state, tools, api_key=api_key or None, model=model)
            if result.get("notice") and result["mode"] == "demo" and api_key:
                state["_agent_notice"] = result["notice"]
            st.rerun()
        if state.get("_agent_notice"):
            st.caption(state["_agent_notice"])
        if state.get("agent_trace"):
            with st.expander("이번 답변의 조회 근거 (디버깅용)", expanded=False):
                st.json(state["agent_trace"], expanded=False)
        _threshold_controls(state, tools)
    with right:
        _report_panel(state, tools)
