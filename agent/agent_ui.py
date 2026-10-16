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
from pathlib import Path

import pandas as pd
import streamlit as st

from .agent_core import DEFAULT_MODEL, opening_summary, run_turn
from .agent_tools import (DEFAULT_THRESHOLDS, TODAY, AgentTools, TemplateStore, export_xlsx, parse_docx,
                          parse_xlsx)


def get_templates(state: dict | None = None) -> TemplateStore:
    """현재 브라우저 세션에서만 양식과 업로드를 유지한다."""
    store = TemplateStore(state=st.session_state if state is None else state)
    store.ensure_default()
    return store


def _configured_secret_files() -> list:
    try:
        return list(st.get_option("secrets.files") or [])
    except Exception:
        return []


def load_settings() -> tuple[str, str]:
    """있는 설정 파일만 조용히 읽는다. 파일 부재·파싱 오류를 화면에 노출하지 않는다."""
    key, model = "", ""
    configured = _configured_secret_files()
    # st.secrets는 파일이 없으면 예외 전에 st.error로 내부 경로를 표시한다.
    # 공개 설정의 파일 목록을 직접 읽어 같은 순서로 덮어쓰되 원문은 출력하지 않는다.
    for filename in configured:
        try:
            path = Path(filename).expanduser()
            if not path.is_file():
                continue
            try:
                import tomllib
                with path.open("rb") as handle:
                    values = tomllib.load(handle)
            except ImportError:  # Python 3.10 배포 환경
                import toml
                values = toml.loads(path.read_text(encoding="utf-8"))
            if "OPENAI_API_KEY" in values:
                key = str(values.get("OPENAI_API_KEY") or "")
            if "OPENAI_MODEL" in values:
                model = str(values.get("OPENAI_MODEL") or "")
        except (OSError, ValueError, TypeError):
            continue
    return (key or os.environ.get("OPENAI_API_KEY", "")).strip(), (model or os.environ.get("OPENAI_MODEL", "") or DEFAULT_MODEL).strip()


# 에이전트 탭 색상. 대시보드 테마 CSS의 `.stApp div {color: 테마 글자색}` 규칙보다 우선하도록
# 모든 선택자 앞에 `.stApp`을 붙인다(그러지 않으면 다크 모드에서 밝은 배경 위에 밝은 글자가 된다).
AGENT_PALETTES = {
    "light": {
        "user_bg": "#dcfce7", "user_text": "#111827", "bot_bg": "#f1f5f9", "bot_text": "#111827",
        "muted": "#6b7280", "line": "#d1d5db", "field_text": "#172a2a", "tag": "#536262",
        "auto_bg": "#eaf3f2", "input_bg": "#fff9e0", "blank_bg": "#f2f5f5",
        "empty": "#5b6370", "req": "#b91c1c", "flash": "#d9f2cf",
        "card_bg": "#ffffff", "card_text": "#111827",
    },
    "dark": {
        "user_bg": "#14532d", "user_text": "#dcfce7", "bot_bg": "#21262d", "bot_text": "#e6edf3",
        "muted": "#9da7b3", "line": "#30363d", "field_text": "#e6edf3", "tag": "#9da7b3",
        "auto_bg": "#11302c", "input_bg": "#3a3312", "blank_bg": "#1f242b",
        "empty": "#9da7b3", "req": "#f87171", "flash": "#1f4d2b",
        "card_bg": "#161b22", "card_text": "#e6edf3",
    },
}


def agent_css(theme: str = "light") -> str:
    c = AGENT_PALETTES.get(theme, AGENT_PALETTES["light"])
    return f"""<style>
.stApp .ag-bubble{{padding:8px 11px;border-radius:10px;margin:3px 0;font-size:15px;line-height:1.65;white-space:pre-wrap;
  overflow-wrap:anywhere;max-width:92%;display:inline-block}}
.stApp .ag-user{{background:{c['user_bg']};color:{c['user_text']};float:right;clear:both}}
.stApp .ag-bot{{background:{c['bot_bg']};color:{c['bot_text']};float:left;clear:both}}
.stApp .ag-row{{overflow:hidden}}
.stApp .ag-mode{{font-size:11px;color:{c['muted']};margin:0 2px}}
.stApp .ag-section{{font-weight:700;font-size:14px;margin:10px 0 3px;padding-bottom:3px;border-bottom:1px solid {c['line']}}}
.stApp .ag-field{{display:grid;grid-template-columns:40px minmax(120px,1fr) minmax(110px,1.2fr);gap:6px;padding:5px 8px;
  margin:2px 0;border-radius:5px;font-size:14px;color:{c['field_text']}}}
.stApp .ag-field *{{color:inherit}}
.stApp .ag-field.auto{{background:{c['auto_bg']}}}.stApp .ag-field.user{{background:{c['input_bg']}}}
.stApp .ag-field.blank,.stApp .ag-field.formula{{background:{c['blank_bg']}}}
.stApp .ag-field .ag-tag{{font-size:11px;color:{c['tag']}}}.stApp .ag-val{{white-space:pre-wrap;overflow-wrap:anywhere}}
.stApp .ag-field.empty .ag-val{{color:{c['empty']}}}.stApp .ag-field .ag-req{{color:{c['req']}}}
@keyframes ag-flash{{from{{box-shadow:inset 0 0 0 2px #16a34a;background:{c['flash']}}}to{{box-shadow:none}}}}
.stApp .ag-field.fresh{{animation:ag-flash 4s ease-out}}
.stApp .ag-card{{border:1px solid {c['line']};border-radius:8px;padding:8px 10px;margin:6px 0;font-size:14px;
  background:{c['card_bg']};color:{c['card_text']}}}
.stApp .ag-card *{{color:inherit}}
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
    counts = (f"설정 기준 발생 구간 {r['anomaly_count_before']}개 → {r['anomaly_count_after']}개" if r.get("data_available") and r.get("anomaly_count_before") is not None
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
    draft = state.get("report_draft") or None
    current_date = str(state["view_date"])
    current_eq = state.get("selected_equipment") or "전체"
    st.markdown("### 점검 보고서")
    st.caption("1 대상 확인 → 2 조치 기록 → 3 필수 입력·담당자 판정 → 4 Excel 다운로드")
    st.markdown(f"**현재 선택: {html.escape(current_date)} · {html.escape(current_eq)}**")
    if draft:
        st.info(f"작성 중인 보고서: {draft['date']} · {draft['equipment']} — 화면의 날짜·설비를 바꿔도 이 초안은 유지됩니다.")
        with st.expander("현재 선택으로 새 보고서 시작", expanded=bool(state.get("report_start_requested"))):
            st.caption("작성 중인 내용을 유지하려면 그대로 닫으세요. 새로 시작하면 이전 초안은 이번 세션에서 한 번 복원할 수 있습니다.")
            if st.button("현재 선택으로 새 보고서", key="ag_new_report", use_container_width=True):
                tools.start_new_report(current_date, current_eq)
                tools.templates.mark_asked()
                state.pop("report_start_requested", None)
                state["_ag_form_generation"] = state.get("_ag_form_generation", 0) + 1
                st.rerun()
    else:
        st.caption("선택한 날짜·설비의 자료를 채운 뒤 점검 내용과 담당자 정보를 입력합니다.")
        if st.button("선택한 대상으로 보고서 시작", type="primary", key="ag_start_report", use_container_width=True):
            tools.start_new_report(current_date, current_eq)
            tools.templates.mark_asked()
            state.pop("report_start_requested", None)
            st.rerun()
    if state.get("previous_report_draft"):
        if st.button("이전 초안 복원", key="ag_restore_report"):
            previous = state.pop("previous_report_draft")
            state["previous_report_draft"] = draft
            state["report_draft"] = previous
            state["_ag_form_generation"] = state.get("_ag_form_generation", 0) + 1
            st.rerun()
    if draft:
        required = [r for r in draft["rows"] if r.get("required")]
        done = sum(_filled(r.get("value")) for r in required)
        st.progress(done / len(required) if required else 1.0, text=f"필수 입력 {done} / {len(required)} 완료")
        if not draft.get("data_available"):
            st.warning("선택한 보고서 대상의 실측 자료가 없습니다. 날짜·설비를 확인한 뒤 새 보고서를 시작하세요.")
        _manual_form(state, tools, draft)
        st.markdown("**3. 담당자 종합판정**")
        st.caption("센서 기준 판정과 별개입니다. 실제 점검 결과를 확인한 담당자가 선택하세요. 내용을 수정하면 다시 확인해야 합니다.")
        revision = draft.get("revision", 0)
        options = ["판정을 선택하세요", "적합", "추가 점검 필요", "조치 후 재확인 필요"]
        judgment = st.selectbox("점검 결과", options, index=options.index(draft["judgment"]) if draft.get("judgment") in options else 0,
            key=f"ag_judgment_{revision}_{state.get('_ag_form_generation', 0)}")
        if st.button("담당자로서 이 판정 확인", disabled=judgment == options[0], key="ag_confirm_judgment", use_container_width=True):
            tools.confirm_judgment(judgment)
            st.rerun()
        if draft.get("judgment_confirmed"):
            st.success(f"담당자 확인 완료: {draft['judgment']}")
        with st.expander("자동으로 채운 자료·전체 양식 미리보기", expanded=False):
            st.caption("설정 기준 이탈과 AI 탐지는 고장을 확정하지 않습니다. 전체 탐지 내역은 Excel 별첨에 포함됩니다.")
            for row in draft["rows"]:
                if row.get("kind") == "table":
                    records = row.get("value") or []
                    st.markdown(f"**{row['name']} · 대표 {min(3, len(records))} / 전체 {len(records)}개**")
                    if records:
                        st.dataframe(pd.DataFrame(records[:3]), hide_index=True, use_container_width=True)
                elif row.get("source") != "user":
                    st.markdown(f"**{row['name']}**: {_value_text(row)}")
        st.markdown("**4. Excel 다운로드**")
        blockers = draft.get("export_blockers", [])
        if blockers:
            st.info("다운로드 전 남은 항목: " + ", ".join(blockers))
        else:
            st.caption("본문 대표 내역과 설정 기준·AI 전체 내역 별첨을 함께 저장합니다.")
        ready = bool(draft.get("ready_to_export") and draft.get("judgment_confirmed"))
        data = export_xlsx(draft, template) if ready else b" "
        st.download_button("점검 보고서 Excel 다운로드", data=data, disabled=not ready, use_container_width=True,
            key="ag_export", file_name=f"점검보고서_{draft['date']}_{re.sub(r'[^0-9A-Za-z가-힣]+', '_', draft['equipment'] or '전체')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with st.expander("보고서 양식 설정", expanded=False):
        st.caption(f"현재 양식: {template.get('name')} · 변경 사항은 현재 브라우저 세션에만 적용됩니다.")
        if st.button("다른 양식 등록하기", key="ag_tpl_open", use_container_width=True):
            state["template_upload_open"] = True
            st.rerun()
        if st.button("기본 양식으로 복원", disabled=bool(template.get("is_default")), key="ag_tpl_reset", use_container_width=True):
            tools.execute("reset_report_template", {})
            st.rerun()


def _manual_form(state: dict, tools: AgentTools, draft: dict) -> None:
    manual = [r for r in draft["rows"] if r.get("source") == "user" and re.sub(r"\s", "", r["name"]) != "종합판정"]
    generation = state.get("_ag_form_generation", 0)
    revision = draft.get("revision", 0)
    st.markdown("**2. 조치 기록과 필수 정보**")
    st.caption("별표(*) 항목부터 입력하세요. 입력값을 지우고 반영하면 보고서에서도 지워집니다. 실제로 수행하지 않은 조치는 적지 않아도 됩니다.")
    with st.form("ag_manual_form"):
        values = {}
        visible = [r for r in manual if r.get("required") or "조치내용" in re.sub(r"\s", "", r["name"])]
        rest = [r for r in manual if r not in visible]
        def render_field(r):
            key = f"ag_in_{generation}_{revision}_{draft['date']}_{draft['equipment']}_{r['name']}"
            label = r["name"] + (" *" if r.get("required") else "")
            return st.text_area(label, value=str(r.get("value") or ""), height=80, key=key) if "내용" in r["name"] else st.text_input(label, value=str(r.get("value") or ""), placeholder=r.get("hint") or "", key=key)
        for r in visible:
            values[r["name"]] = render_field(r)
        with st.expander(f"추가 입력 항목 {len(rest)}개", expanded=False):
            for r in rest:
                values[r["name"]] = render_field(r)
        if st.form_submit_button("입력 내용을 보고서에 반영", use_container_width=True):
            changed = {k: v for k, v in values.items() if v != str(next((r.get("value") or "") for r in manual if r["name"] == k))}
            if changed:
                tools.execute("create_report", {"date": draft["date"], "equipment": draft["equipment"], "extra_fields": changed})
            st.rerun()


def render_agent_tab(state: dict | None = None) -> None:
    state = st.session_state if state is None else state
    state.setdefault("view_date", TODAY.isoformat())
    tools = AgentTools(state, get_templates(state))
    if state.get("report_start_requested") and not state.get("report_draft"):
        # 점검 화면에서 사용자가 누른 '이 설비로 보고서 시작'을 한 번만 실행한다.
        tools.start_new_report(str(state["view_date"]), state.get("selected_equipment"))
        tools.templates.mark_asked()
        state.pop("report_start_requested", None)
    api_key, model = load_settings()
    _ensure_summary(state, tools)
    st.markdown(agent_css(state.get("theme", "light")), unsafe_allow_html=True)
    st.caption("자료를 질문으로 확인하고 점검 보고서를 작성하세요. 보고서 시작·직접 입력·다운로드는 대화 없이도 사용할 수 있습니다.")
    left, right = st.columns([4, 6], gap="large")
    with left:
        with st.expander("AI에게 자료 질문하기", expanded=False):
            st.markdown("### 자료에 질문하기")
            st.caption(f"{state['view_date']} · {state.get('selected_equipment') or '공장 전체'} · " + ("AI 대화" if api_key else "간단 조회 모드"))
            _conversation(state, model, api_key)
            st.caption("아래 질문을 누르면 바로 전송합니다.")
            quick = None
            for title, text in (("선택일 점검할 설비", "선택한 날짜에 점검할 설비를 요약해줘"),
                                ("선택 설비 전압 확인", f"{state.get('selected_equipment') or '전체 설비'} 전압은 어때?"),
                                ("이번 달 예상 요금", "선택한 날짜의 달 예상 요금을 알려줘")):
                if st.button(title, key="ag_quick_" + title, use_container_width=True):
                    quick = text
            _template_upload(state, tools)
            prompt = st.chat_input("예: 3호기 전압은 어때?", key="ag_chat_input")
            if prompt or quick:
                with st.spinner("자료를 확인하고 있어요…"):
                    result = run_turn(prompt or quick, state, tools, api_key=api_key or None, model=model)
                if result.get("notice") and result["mode"] == "demo" and api_key:
                    state["_agent_notice"] = "AI 연결이 원활하지 않아 간단 조회로 안내했습니다. 잠시 후 다시 시도하세요."
                st.rerun()
            if state.get("_agent_notice"):
                st.caption(state["_agent_notice"])
        _threshold_controls(state, tools)
    with right:
        _report_panel(state, tools)
