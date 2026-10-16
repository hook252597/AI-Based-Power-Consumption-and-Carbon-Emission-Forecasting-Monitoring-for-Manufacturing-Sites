# -*- coding: utf-8 -*-
"""마가렛트 에이전트 — 모델 호출과 function calling 루프 (agent/agent.md 2·3·4·8절)

필요 패키지 : openai (상용 API), streamlit 없이도 동작
환경변수    : OPENAI_API_KEY (없으면 데모 모드), OPENAI_MODEL (기본 gpt-5-mini)
              — 키 읽기는 agent_ui.load_settings 가 st.secrets → 환경변수 순으로 담당한다.
실행        : 단독 실행하지 않는다. agent_ui.render_agent_tab 이 run_turn 을 호출한다.

구성
- call_model(messages, tools, ...) : 제공사별 분기는 이 함수 안에만 둔다(현재 OpenAI).
- run_turn(text, state, tools, ...): 사용자 메시지 → 모델 → 도구 실행 → 모델 … (최대 5회) → 답변.
- 시스템 프롬프트는 agentSystem.md 를 읽고, 매 턴 [현재 상태] 블록을 새로 만들어 끝에 붙인다
  (지난 턴의 상태 블록은 대화 기록에 남기지 않는다).
- 대화 기록은 최근 10턴만 유지한다. 보고서 초안은 state["report_draft"]에 따로 둔다.
- API 키가 없거나 호출이 실패하면 데모 모드: 같은 도구를 규칙으로 호출해 답한다(숫자는 도구 결과만 사용).
"""
from __future__ import annotations

import copy
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .agent_tools import (APP_DIR, DEFAULT_THRESHOLDS, EQUIPMENT_LIST, FAMILY_LABEL, ITEMS, NOW, TODAY,
                          TOOL_SCHEMAS, AgentTools, _norm, duration_text, forecast_period, resolve_equipment, sensor_period)

SYSTEM_PATH = APP_DIR / "agentSystem.md"
DEFAULT_MODEL = "gpt-5-mini"
MAX_ROUNDS = 5            # 한 사용자 메시지당 모델 호출 최대 횟수(무한 루프·요금 방지)
MAX_TURNS = 10            # 대화 기록 유지 턴 수
TIMEOUT_SECONDS = 20
MAX_TOOL_CHARS = 12_000   # 모델에 넘기는 도구 결과 길이 제한(토큰 절약)

DELAY_TEXT = "응답이 지연되고 있습니다. 다시 시도할까요?"
FAIL_TEXT = "확인에 실패했습니다. 질문을 조금 더 구체적으로 다시 말씀해 주세요."


class ModelTimeout(Exception):
    """모델 응답 시간 초과."""


class ModelError(Exception):
    """모델 호출 실패(키 오류·네트워크·제공사 오류 등)."""


# -----------------------------------------------------------------------------
# 1) 모델 호출 — 제공사를 바꿀 때는 이 함수만 고친다
# -----------------------------------------------------------------------------
def call_model(messages: list[dict], tools: list[dict], *, api_key: str, model: str = DEFAULT_MODEL,
               timeout: float = TIMEOUT_SECONDS) -> dict:
    """반환: {"content": str | None, "tool_calls": [{"id", "name", "arguments": dict}]}"""
    try:
        from openai import APITimeoutError, OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ModelError("openai 패키지가 설치되지 않았습니다.") from exc
    try:
        client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        response = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto")
    except APITimeoutError as exc:
        raise ModelTimeout(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — 인증·요금·네트워크 오류 모두 데모 모드로
        raise ModelError(f"{type(exc).__name__}") from exc
    message = response.choices[0].message
    calls = []
    for call in message.tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except ValueError:
            arguments = {}
        calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})
    return {"content": message.content, "tool_calls": calls}


# -----------------------------------------------------------------------------
# 2) 시스템 프롬프트 + 매 턴 상태 블록
# -----------------------------------------------------------------------------
def system_prompt() -> str:
    try:
        return SYSTEM_PATH.read_text(encoding="utf-8")
    except OSError:
        return "너는 제조 현장 에너지 관리 대시보드의 AI 에이전트다. 모든 수치는 도구 결과에서만 가져온다."


def state_block(state: dict, tools: AgentTools) -> str:
    th = tools.thresholds
    t = tools.templates.get()
    draft = state.get("report_draft") or {}
    sp, fp = sensor_period(), forecast_period()
    view_range = {"daily": "일일", "monthly": "한달"}.get(state.get("view_range") or state.get("anomaly_view"), "일일")
    report = (f"{draft.get('date')} · {draft.get('equipment')} · 남은 필수 항목: "
              f"{', '.join(draft.get('missing_user_fields') or []) or '없음'}") if draft else "없음"
    return "\n".join([
        "[현재 상태]",
        f"과거 데이터 체험 기준: {NOW:%Y-%m-%d %H:%M}. 마지막 실측일: {TODAY}. 실제 오늘 날짜와 다름.",
        f"조회 중인 날짜: {state.get('view_date')}",
        f"선택된 설비: {state.get('selected_equipment') or '공장 전체'}",
        f"이상현황 보기 기준: {view_range}",
        f"현재 임계치: 상전압 {th['phase_voltage']:g}V / 선간전압 {th['line_voltage']:g}V / 역률 {th['power_factor']:g}%"
        + ("" if th == DEFAULT_THRESHOLDS else " (기본값에서 변경됨)"),
        f"보고서 양식: 등록됨 ({t.get('name')}{' · 기본 양식' if t.get('is_default') else ' · 사용자 양식'}) · "
        f"최초 양식 질문: {'완료' if tools.templates.template_asked() else '아직 안 함'}",
        f"작성 중인 보고서: {report}",
        f"데이터 범위: 설비 실측 {sp[0] + ' ~ ' + sp[1] if sp else '없음'} · AI 전력 예측 {fp[0] + ' ~ ' + fp[1] if fp else '없음'}",
        "",
        "[대시보드 연동 규칙 — 위 지침과 충돌하면 이 규칙을 따른다]",
        "- 기간을 말하지 않으면 '조회 중인 날짜'(월 질문은 그 날짜의 달) 기준으로 답하고, 어느 날짜인지 밝힌다.",
        "- 도구 인수는 제공된 함수 스키마만 쓴다(지침 예시의 start/end 인수는 쓰지 않는다). date를 생략하면 조회 중인 날짜가 쓰인다.",
        "- 기준 시각 이후(4/30, 5월)는 실측이 없다. get_power·get_forecast의 data_kind가 forecast면 반드시 '예측'이라고 말한다.",
        "- basis와 집계 단위를 반드시 따른다. event_count는 설비·항목별 연속 구간 수이며 고장 사건 수가 아니다. flagged_records는 5초 표본 수다. 서로 더하지 않는다.",
        "- duration_seconds가 없으면 실제 지속 시간은 확인 불가다. 관측 시간대 수를 시간 단위 지속 시간으로 바꾸지 않는다. 누적 탐지 시간과 최장 연속 시간도 구분한다.",
        "- AI 이상탐지는 고정 모델 판정이며 규칙 임계치를 바꿔도 변하지 않는다. 원인을 단정하지 않는다.",
        "- 설비 13개의 전력 사용량은 거의 같다. 순위를 말할 때 차이가 매우 작다는 점을 함께 밝힌다.",
        "- 보고서는 create_report 결과만 근거로 말한다. 미입력 항목과 export_blockers를 안내한다. 담당자 종합판정은 화면에서 사용자가 직접 확인해야 하며 AI는 확정할 수 없다.",
        "- 작성 중인 초안과 날짜·설비가 다르면 기존 초안을 유지한다. 새 보고서는 화면에서 시작하도록 안내한다. 계약전력·최대수요전력은 추정해서 채우지 않는다.",
    ])


def build_messages(state: dict, tools: AgentTools, history: list[dict]) -> list[dict]:
    return [{"role": "system", "content": system_prompt()}, *history,
            {"role": "system", "content": state_block(state, tools)}]


def _trim(history: list[dict]) -> list[dict]:
    """최근 MAX_TURNS개의 사용자 턴만 남긴다(턴 경계에서 자르므로 도구 메시지 짝이 깨지지 않음)."""
    users = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(users) <= MAX_TURNS:
        return history
    return history[users[-MAX_TURNS]:]


def _compact(result: Any) -> str:
    """보고서 상태·전체 건수를 보존하고 대표 자료만 전송한다. JSON을 중간에서 자르지 않는다."""
    small = copy.deepcopy(result)
    if isinstance(small, dict):
        if isinstance(small.get("rows"), list):
            small["rows"] = [{k: row.get(k) for k in ("name", "source", "required", "value")}
                              for row in small["rows"] if row.get("required") or row.get("source") == "user" or row.get("value") not in (None, "", [])]
        for key in ("items", "ai_items"):
            if isinstance(small.get(key), list):
                small[key] = small[key][:6]
                small[f"shown_{key}"] = len(small[key])
    text = json.dumps(small, ensure_ascii=False, default=str)
    if len(text) > MAX_TOOL_CHARS and isinstance(small, dict):
        if isinstance(small.get("rows"), list):
            small["rows"] = [r for r in small["rows"] if r.get("required") or r.get("source") == "user"]
        small.pop("series", None)
        small["detail_note"] = "상세 표는 화면에서 확인하세요. 전체 건수와 필수 입력 상태는 보존했습니다."
        text = json.dumps(small, ensure_ascii=False, default=str)
    return text


# -----------------------------------------------------------------------------
# 3) 변경 도구 안전장치 — 모델의 제안만으로 화면 상태를 바꾸지 않는다
# -----------------------------------------------------------------------------
def _reset_requested(text: str) -> bool:
    return bool(re.search(r"되돌려|되돌리|복원|원래(?:대로|\s*값|\s*기준)?|기본\s*(?:값|기준)?\s*(?:으로|로)", text))


def mutation_allowed(name: str, args: dict, text: str, state: dict) -> bool:
    if name == "set_thresholds":
        return bool(re.search(r"역률|전압|임계|기준", text)) and bool(re.search(r"\d", text) or _reset_requested(text)) \
            or (_reset_requested(text) and bool(state.get("threshold_history")))
    if name == "save_report_template":
        return "양식" in text or "항목" in text or bool(state.get("pending_template")) and \
            bool(re.search(r"^(응|네|예|맞아|좋아|그렇게|적용)", text.strip()))
    if name == "reset_report_template":
        return "양식" in text and _reset_requested(text)
    return True


# -----------------------------------------------------------------------------
# 4) 한 턴 실행
# -----------------------------------------------------------------------------
def run_turn(text: str, state: dict, tools: AgentTools, *, api_key: str | None = None,
             model: str = DEFAULT_MODEL) -> dict:
    """반환: {"text", "mode": "openai"|"demo", "trace": [{name, args, result}], "notice"}"""
    text = str(text or "").strip()[:4000]
    state.setdefault("messages", [])
    history = state.setdefault("agent_history", [])
    state["messages"].append({"role": "user", "content": text})
    snapshot = copy.deepcopy(history)
    trace: list[dict] = []
    result = None
    notice = "API 키가 없어 데모 모드로 답합니다."
    if api_key:
        try:
            result = _run_model(text, state, tools, history, trace, api_key, model)
        except ModelTimeout:
            state["agent_history"] = snapshot + [{"role": "user", "content": text},
                                                 {"role": "assistant", "content": DELAY_TEXT}]
            result = {"text": DELAY_TEXT, "mode": "openai", "notice": "timeout"}
        except ModelError as exc:
            # 도중에 실패하면 이번 턴의 도구 호출 기록을 버리고(짝이 깨진 기록 방지) 데모 모드로 답한다
            state["agent_history"] = snapshot
            trace.clear()
            notice = f"API 호출 실패({exc}) — 데모 모드로 답합니다."
    if result is None:
        answer = demo_reply(text, state, tools, trace)
        state["agent_history"] = _trim(state.get("agent_history", []) + [
            {"role": "user", "content": text}, {"role": "assistant", "content": answer}])
        result = {"text": answer, "mode": "demo", "notice": notice}
    else:
        state["agent_history"] = _trim(state["agent_history"])
    result["trace"] = trace
    state["messages"].append({"role": "assistant", "content": result["text"], "mode": result["mode"]})
    state["agent_trace"] = trace
    return result


def _run_model(text: str, state: dict, tools: AgentTools, history: list[dict], trace: list[dict],
               api_key: str, model: str) -> dict:
    history.append({"role": "user", "content": text})
    for _ in range(MAX_ROUNDS):
        reply = call_model(build_messages(state, tools, history), TOOL_SCHEMAS, api_key=api_key, model=model)
        if not reply["tool_calls"]:
            content = (reply["content"] or "").strip() or FAIL_TEXT
            history.append({"role": "assistant", "content": content})
            return {"text": content, "mode": "openai", "notice": ""}
        history.append({"role": "assistant", "content": reply["content"], "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
            for c in reply["tool_calls"]]})
        for call in reply["tool_calls"]:
            if not mutation_allowed(call["name"], call["arguments"], text, state):
                output = {"error": "사용자가 명시적으로 요청한 변경이 아니어서 실행하지 않았습니다."}
            else:
                output = tools.execute(call["name"], call["arguments"])
            trace.append({"name": call["name"], "args": call["arguments"], "result": output})
            history.append({"role": "tool", "tool_call_id": call["id"], "content": _compact(output)})
    history.append({"role": "assistant", "content": FAIL_TEXT})
    return {"text": FAIL_TEXT, "mode": "openai", "notice": "tool_round_limit"}


# -----------------------------------------------------------------------------
# 5) 첫 화면 요약 (선택 날짜 기준)
# -----------------------------------------------------------------------------
def _alert_text(r: dict) -> str:
    parts = []
    for key, label in (("rule", "설정 기준"), ("ai", "AI 탐지")):
        metric = r.get(key, {})
        count = metric.get("event_count")
        parts.append(f"{label} {metric.get('equipment_count', 0)}대 · " +
                     (f"발생 구간 {count:,}개" if count is not None else f"이상 기록이 관측된 설비·시간 조합 {metric.get('affected_hours', 0):,}개(구간 수 확인 불가)"))
    return "; ".join(parts)


def opening_summary(state: dict, tools: AgentTools) -> str:
    view = state.get("view_date")
    r = tools.execute("get_anomalies", {"date": view})
    state.setdefault("_agent_focus", {}).update(anomaly_date=view, equipment=None)
    if r.get("error"):
        return f"{view} 기록을 확인하지 못했습니다: {r['error']}"
    if not r.get("data_available"):
        return f"{view}에는 설비 실측 기록이 없어 상태를 판단할 수 없습니다. 실측이 있는 날짜를 선택하거나 전력 예측을 조회하세요."
    if not r.get("affected_equipment"):
        return f"{view}에는 탐지 기록이 없습니다. 실제 고장 여부는 현장 확인이 필요합니다."
    return f"{view} 기록 요약: {_alert_text(r)}입니다. 탐지 구간은 고장 사건 수가 아닙니다. 설비를 골라 근거를 확인하세요."


# -----------------------------------------------------------------------------
# 6) 데모 모드 — 미리 준비한 흐름을 같은 도구로 재생(숫자는 도구 결과만)
# -----------------------------------------------------------------------------
def _call(tools: AgentTools, trace: list[dict], name: str, **args) -> Any:
    out = tools.execute(name, args)
    trace.append({"name": name, "args": args, "result": out})
    return out


def _date_in_text(text: str, state: dict) -> str:
    view = date.fromisoformat(str(state.get("view_date") or TODAY.isoformat())[:10])
    m = re.search(r"(20\d{2})[-./](\d{1,2})[-./](\d{1,2})", text)
    if m:
        return date(*map(int, m.groups())).isoformat()
    m = re.search(r"(?:(20\d{2})년\s*)?(\d{1,2})월\s*(\d{1,2})일", text)
    if m:
        return date(int(m.group(1) or view.year), int(m.group(2)), int(m.group(3))).isoformat()
    if "어제" in text:
        return (TODAY - timedelta(days=1)).isoformat()
    if "오늘" in text or "지금" in text:
        return TODAY.isoformat()
    return view.isoformat()


def _equipment_in_text(text: str) -> str | None:
    compact = re.sub(r"\s+", "", text)
    for _, name in EQUIPMENT_LIST:
        label = name.split("(", 1)[1].rsplit(")", 1)[0]
        pattern = r"(?<!\d)" + re.escape(re.sub(r"\s+", "", label)) if re.match(r"\d+호기", label) else re.escape(label)
        if re.search(pattern, compact):
            return name
    m = re.search(r"(\d+)\s*번\s*(?:설비|채널)", text)
    if m:
        try:
            return resolve_equipment(m.group(1))
        except ValueError:
            return None
    return None


def _won(v: float | None) -> str:
    if v is None:
        return "-"
    return f"약 {v / 10_000:,.0f}만 원" if v >= 100_000 else f"{v:,.0f}원"


def _user_fields_in_text(text: str, tools: AgentTools) -> dict:
    out = {}
    for f in tools.templates.get().get("fields", []):
        if f["source"] != "user":
            continue
        chars = re.sub(r"\s+", "", f["name"])
        pattern = r"\s*".join(map(re.escape, chars)) + r"\s*(?:은|는|:|=|을|를)?\s*([^,\n]+?)(?:\s*(?:으로|로)?\s*(?:해줘|넣어줘|해|입니다|이야))?\s*(?:,|$)"
        m = re.search(pattern, text)
        if m and m.group(1).strip() and not re.match(r"(누구|뭐|언제|어떻게|[?？])", m.group(1).strip()):
            out[f["name"]] = m.group(1).strip()
    return out


def demo_reply(text: str, state: dict, tools: AgentTools, trace: list[dict]) -> str:
    d = _date_in_text(text, state)
    eq = _equipment_in_text(text)
    focus = state.setdefault("_agent_focus", {})

    # 보고서 양식
    if "양식" in text:
        if _reset_requested(text):
            r = _call(tools, trace, "reset_report_template")
            return f"기본 양식({r.get('name')})으로 되돌렸습니다." if not r.get("error") else r["error"]
        if re.search(r"바꿀|바꿔|변경|다른|새|올릴|등록|우리\s*회사", text):
            state["template_upload_open"] = True
            return "네, 양식 파일을 올려주세요(.xlsx / .docx). HWP는 항목을 말로 불러주시면 됩니다."
    if state.get("_template_question_open") and re.search(r"없어|기본|그걸로|그냥|괜찮", text):
        tools.templates.mark_asked()
        state["_template_question_open"] = False
        return "기본 양식으로 작성하겠습니다. 앞으로 다시 여쭙지 않겠습니다."

    # 임계치
    if re.search(r"되돌려|되돌리|복원|원래대로|기본값", text) and state.get("threshold_history") \
            and not re.search(r"양식|보고서", text):
        r = _call(tools, trace, "set_thresholds", **DEFAULT_THRESHOLDS, reason="사용자 요청: 기본값 복원")
        if r.get("error"):
            return r["error"]
        if not r.get("changed"):
            return "임계치가 이미 기본값입니다(상전압 210V · 선간전압 360V · 역률 85%)."
        return ("임계치를 기본값으로 되돌렸습니다. " + _threshold_counts(r))
    m = re.search(r"(역률|상전압|선간\s*전압|선간)[^\d]{0,12}(\d+(?:\.\d+)?)\s*(%|V|볼트)?", text)
    if m and re.search(r"올려|낮춰|내려|바꿔|변경|설정|맞춰|로\s*해|으로|로$|로\s*$", text):
        key = {"역률": "power_factor", "상전압": "phase_voltage"}.get(m.group(1), "line_voltage")
        r = _call(tools, trace, "set_thresholds", **{key: float(m.group(2))}, reason=f"사용자 요청: {text[:60]}")
        if r.get("error"):
            return r["error"]
        return f"{r['log_entry']['change']}{'%' if key == 'power_factor' else 'V'}로 바꿨습니다. " + _threshold_counts(r)

    # 보고서: 조치 보고 / 항목 입력 / 작성 요청
    fields = _user_fields_in_text(text, tools)
    action = re.search(r"(교체|점검|수리|청소|조치|정비|확인)\s*(했|완료|끝냈|마쳤)", text) and not re.search(r"[?？]|했나|했니", text)
    if action or fields or re.search(r"보고서.*(작성|만들|써)", text):
        draft = state.get("report_draft") or {}
        if draft and not action and not re.search(r"보고서", text):
            # 항목 값만 알려준 경우: 작성 중인 초안(날짜·설비)을 그대로 갱신한다
            target_date, target = draft["date"], draft["equipment"]
        else:
            target = eq or focus.get("equipment") or state.get("selected_equipment")
            target_date = d if re.search(r"\d+월\s*\d+일", text) else focus.get("anomaly_date") or state.get("view_date")
        r = _call(tools, trace, "create_report", date=target_date,
                  equipment=target, action_taken=text if action else "", extra_fields=fields or None)
        if r.get("error"):
            return r["error"]
        lines = [f"{r['equipment']} 건으로 보고서 초안을 {'갱신' if fields and not action else '작성'}했습니다({r['date']})."]
        if fields:
            lines.append("반영: " + ", ".join(f"{k} = {v}" for k, v in fields.items()))
        if not tools.templates.template_asked():
            state["_template_question_open"] = True
            lines.append("그 전에 하나만 여쭙겠습니다. 사내에서 따로 쓰시는 보고서 양식이 있나요? 있으면 파일로 올려주세요. "
                         "없으면 기본 양식(제조현장 안전 점검 보고서)으로 작성하겠습니다.")
        missing = r.get("missing_user_fields") or []
        lines.append(("남은 필수 항목: " + ", ".join(missing) + ". 말씀해 주시면 바로 채우겠습니다.") if missing
                     else "필수 입력을 반영했습니다. 화면에서 담당자 종합판정을 확인하면 Excel 다운로드가 가능합니다.")
        return "\n".join(lines)

    # 작년 비교
    if "작년" in text:
        last = (date.fromisoformat(d) - timedelta(days=365)).isoformat()
        r = _call(tools, trace, "get_power", date=last)
        return "비교할 작년 자료가 없어 작년 대비는 확인할 수 없습니다." if not r.get("data_available") else \
            f"작년 같은 날({last}) 전력은 {r['total_kwh']:,.1f} kWh입니다."

    # 이번 달 요금·예측
    if re.search(r"요금|청구|예상|예측|이번\s*달|이달|탄소", text) and not re.search(r"순위|제일|가장", text):
        month = d[:7]
        r = _call(tools, trace, "get_forecast", month=month, equipment=eq)
        if not r.get("data_available"):
            return r.get("note") or f"{month} 자료가 없습니다."
        kind = {"forecast": "AI 예측", "actual": "실측", "actual+forecast": "실측+AI 예측"}[r["data_kind"]]
        diff = r.get("prev_month_diff_pct")
        parts = [f"{month} {('' if not eq else eq + ' ')}전기요금은 {_won(r['expected_cost_krw'])}으로 "
                 f"{'예상됩니다' if r['data_kind'] != 'actual' else '집계됩니다'}({kind}, 180원/kWh 기준).",
                 f"전력 {r['expected_kwh']:,.0f} kWh, 탄소 약 {r['expected_co2_kg'] / 1000:,.1f} tCO2"
                 + (f", 전월 대비 {diff:+.1f}%" if diff is not None else "") + "입니다."]
        if r["data_kind"] == "actual+forecast":
            parts.append(f"(실측 {r['actual_kwh_so_far']:,.0f} kWh + 예측 {r['forecast_kwh']:,.0f} kWh"
                         + (f", 자료 없는 {r['filled_hours_by_average']}시간은 평균으로 보정" if r['filled_hours_by_average'] else "") + ")")
        return " ".join(parts)

    # 전력 순위
    if re.search(r"제일|가장|많이\s*쓰|top\s*\d|순위|상위", text, re.I):
        r = _call(tools, trace, "get_power_ranking", date=d)
        if not r.get("data_available"):
            return r.get("note", "해당 날짜 자료가 없습니다.")
        kind = "AI 예측" if r.get("data_kind") == "forecast" else "실측"
        top = "\n".join(f"{i['rank']}. {i['equipment']} — {i['kwh']:,.2f} kWh ({i['share_pct']}%)" for i in r["items"][:3])
        return (f"{r['date']} 전력 사용량 상위 3개 설비입니다({kind}, 실시간 순위 아님).\n{top}\n"
                "설비 간 차이는 매우 작습니다.")

    # 전압·역률 측정
    if re.search(r"전압|역률", text):
        target = eq or state.get("selected_equipment")
        fam = "line_voltage" if "선간" in text else "power_factor" if "역률" in text else "phase_voltage"
        keys = [k for k, v in ITEMS.items() if v[3] == fam]
        rows = [_call(tools, trace, "get_measurements", date=d, equipment=target, item=k) for k in keys]
        if any(r.get("error") for r in rows):
            return next(r["error"] for r in rows if r.get("error"))
        if not rows[0].get("data_available"):
            return rows[0].get("note", "해당 날짜 자료가 없습니다.")
        focus.update(equipment=rows[0]["equipment"], date=d)
        unit, th = rows[0]["unit"], rows[0]["threshold"]
        bad = [r for r in rows if r["under_threshold_hours"]]
        lines = [f"{'⚠ ' if bad else ''}{rows[0]['equipment']} {d} {FAMILY_LABEL[fam]}은 "
                 + (f"기준({th:g}{unit}) 미만 기록이 관측된 시간대가 있습니다." if bad else f"모두 기준({th:g}{unit}) 이상입니다.")]
        for r in rows:
            lines.append(f"· {r['label']}: 최저 {r['min']:g}{unit} ({r['min_hour']:02d}시)"
                         + (f", 기준 미만 기록이 있는 시간대 {len(r['under_threshold_hours'])}개(지속 시간 아님)" if r["under_threshold_hours"] else ""))
        if bad:
            lines.append("원인을 단정할 수는 없어 현장 점검을 권합니다.")
        return "\n".join(lines)

    # 후속 질문: 어디야?
    if re.search(r"어디|어느\s*설비|어떤\s*설비|어디야", text) and focus.get("anomaly_date"):
        r = _call(tools, trace, "get_anomalies", date=focus["anomaly_date"])
        return _where(r)

    # 전력 사용량
    if re.search(r"전력|전기|사용량|소비량|썼", text):
        r = _call(tools, trace, "get_power", date=d, equipment=eq, granularity="day")
        if not r.get("data_available"):
            return r.get("note", "해당 날짜 자료가 없습니다.")
        kind = "AI 예측" if r.get("data_kind") == "forecast" else "실측"
        return (f"{d} {r['equipment']} 전력 사용량은 {r['total_kwh']:,.1f} kWh, 요금 {r['total_cost_krw']:,.0f}원, "
                f"탄소 {r['total_co2_kg']:,.1f} kgCO2입니다({kind}, 180원/kWh · 0.424 kgCO2/kWh 기준). {r.get('note', '')}").strip()

    # 이상 여부
    if re.search(r"문제|이상|괜찮|어때|상태|점검", text) or eq or re.search(r"\d+월\s*\d+일", text):
        r = _call(tools, trace, "get_anomalies", date=d, equipment=eq)
        if r.get("error"):
            return r["error"]
        if not r.get("data_available"):
            return r.get("note", "해당 기간에는 자료가 없습니다.")
        focus.update(anomaly_date=d, equipment=eq)
        scope = eq or "공장 전체"
        if not r.get("affected_equipment"):
            return f"{d} {scope}에는 탐지 기록이 없습니다."
        return f"{d} {scope}: {_alert_text(r)}입니다. 설정 기준 이탈과 AI 탐지는 별개이며 실제 고장을 확정하지 않습니다."

    return ("설비 이상 여부, 전압·역률, 전력 사용량·요금 예측, 설비별 순위, 보고서 작성을 도와드릴 수 있어요. "
            "예: '오늘 문제 있는 곳 있어?', '4호기 전압 어때?', '이번 달 전기요금 얼마나 나와?'")


def _threshold_counts(r: dict) -> str:
    if not r.get("data_available"):
        return f"{r.get('analysis_date')}에는 실측 자료가 없어 재분석 건수는 확인할 수 없습니다."
    before, after = r.get("anomaly_count_before"), r.get("anomaly_count_after")
    count_text = f"설정 기준 발생 구간 {before}개 → {after}개" if before is not None and after is not None else "시간대 집계만 있어 실제 발생 구간 수는 확인 불가"
    return f"{r['analysis_date']} 기준 {count_text}입니다. AI 판정은 그대로입니다. 기준 변경은 실제 설비 상태가 바뀌었다는 뜻이 아닙니다."


def _where(r: dict) -> str:
    if r.get("error"):
        return r["error"]
    if not r.get("data_available"):
        return r.get("note", "해당 기간에는 자료가 없습니다.")
    affected = r.get("affected_equipment") or []
    if not affected:
        return f"{r['date']}에는 탐지 기록이 없습니다."
    lines = [f"{r['date']} 탐지 기록이 있는 설비입니다."]
    worst = {}
    for e in r.get("items", []):
        worst.setdefault(e["equipment"], e)
    for a in affected[:6]:
        w = worst.get(a["equipment"])
        detail = (f" · {w['item']} {w['value']:g}{w.get('unit', '')} (기준 {w['threshold']:g}{w.get('unit', '')}, "
                  f"지속 {duration_text(w)})") if w else ""
        lines.append(f"· {a['equipment']} — {', '.join(a['items']) or 'AI 탐지'}{detail}")
    lines.append("이 목록은 고장 확인 결과가 아닙니다. 상세 기록과 현장 상태를 함께 확인하세요.")
    return "\n".join(lines)
