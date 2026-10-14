"""All API responses are in-memory mocks; no key file or network is accessed."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_runtime import agent_core as core


class FakeTools:
    def __init__(self, state):
        self.state = state
        self.calls = []

    def schema(self):
        names = ["get_equipment_list", "get_anomalies", "get_measurements", "get_power", "get_power_ranking",
                 "get_forecast", "get_thresholds", "set_thresholds", "create_report"]
        return [{"type": "function", "name": name, "description": name, "strict": True,
                 "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}
                for name in names]

    def execute(self, name, args):
        self.calls.append((name, copy.deepcopy(args)))
        if name == "get_equipment_list":
            return [{"name": "12(4호기)"}, {"name": "13(3호기)"}]
        if name == "get_anomalies":
            if not args["date"].startswith("2025-04"):
                return {"available": False, "note": "센서 자료 없음"}
            return {"date": args["date"], "ai_count": 4, "rule_count": 7,
                    "items": [{"equipment": "12(4호기)", "item": "역률 R", "value": 80, "unit": "%"}]}
        if name == "get_measurements":
            return {**args, "min": 208.5, "threshold": 210, "unit": "V", "series": [{"hour": 1, "value": 208.5}]}
        if name == "get_forecast":
            if args["month"] != "2025-05":
                return {"available": False, "note": "예측 자료 없음"}
            return {"expected_kwh": 100, "expected_cost_krw": 18000, "expected_co2_kg": 42.4}
        if name == "get_power_ranking":
            return {"items": [{"equipment": f"장비{i}", "kwh": 100-i} for i in range(1, 5)]}
        if name == "set_thresholds":
            self.state.setdefault("thresholds", dict(core.DEFAULT_THRESHOLDS)).update(
                {k: v for k, v in args.items() if k in core.DEFAULT_THRESHOLDS})
            self.state.setdefault("threshold_history", []).append(copy.deepcopy(args))
            return {"after": copy.deepcopy(self.state["thresholds"]), "anomaly_count_before": 7, "anomaly_count_after": 9}
        if name == "create_report":
            self.state["report_draft"] = copy.deepcopy(args)
            return {"missing_user_fields": ["조치자", "종합 판정"], "ready_to_export": False}
        return {"note": "조회 완료"}


class FakeClient:
    def __init__(self, *responses):
        self.items = list(responses)
        self.requests = []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        result = self.items.pop(0)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


def call(name="get_anomalies", args=None, ident="c1"):
    return {"type": "function_call", "call_id": ident, "name": name,
            "arguments": json.dumps(args or {"date": "2025-04-29", "equipment": None})}


def response(*items, text=""):
    output = list(items)
    if text:
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    return {"output": output, "output_text": text}


@pytest.fixture
def state():
    return {"messages": [], "view_date": "2025-04-29", "selected_equipment": "12(4호기)",
            "thresholds": dict(core.DEFAULT_THRESHOLDS), "threshold_history": [], "report_draft": {}}


def test_responses_round_trip_preserves_reasoning_multiple_calls_and_fresh_state(state):
    reasoning = {"type": "reasoning", "id": "r1", "summary": [], "encrypted_content": "opaque-test"}
    client = FakeClient(response(reasoning, call(), call("set_thresholds", {"power_factor": 90}, "c2")), response(text="변경 결과입니다."))
    tools = FakeTools(state)
    state["api_key"] = "never-send-this-test-value"
    result = core.run_turn("역률 90으로 변경하고 오늘 이상 확인", state, tools, client=client)
    assert result["mode"] == "openai"
    second = client.requests[1]
    assert reasoning in second["input"]
    assert {x["call_id"] for x in second["input"] if x.get("type") == "function_call_output"} == {"c1", "c2"}
    assert '"power_factor":90' in second["instructions"]
    assert "never-send-this-test-value" not in json.dumps(client.requests)
    assert all(r["model"] == "gpt-5-mini" and r["store"] is False for r in client.requests)
    assert all(r["include"] == ["reasoning.encrypted_content"] for r in client.requests)
    assert all(not x.get("role") in ("system", "developer") for turn in state["_agent_turns"] for x in turn)


def test_sdk_constructed_with_timeout_and_no_retries_without_real_key(state, monkeypatch):
    captured = {}
    client = FakeClient(response(text="연결 확인"))
    def factory(**kwargs):
        captured.update(kwargs)
        return client
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    core.run_turn("안녕", state, FakeTools(state), api_key="unit-test-placeholder")
    assert captured == {"api_key": "unit-test-placeholder", "timeout": 20.0, "max_retries": 0}


def test_api_failure_after_mutation_does_not_replay_and_redacts_exception(state):
    client = FakeClient(response(call("set_thresholds", {"power_factor": 90})), RuntimeError("sensitive-error-body"))
    tools = FakeTools(state)
    result = core.run_turn("역률 90", state, tools, client=client)
    assert [name for name, _ in tools.calls] == ["set_thresholds"]
    assert result["mode"] == "local" and result["error"] == "api_unavailable"
    assert "다시 실행하지" in result["text"]
    assert "sensitive-error-body" not in json.dumps(result)
    assert len(state["threshold_history"]) == 1


def test_tool_that_mutates_then_raises_is_not_retried_after_api_error(state):
    class PartiallyFailedTools(FakeTools):
        def execute(self, name, args):
            super().execute(name, args)
            raise RuntimeError("private-tool-details")
    tools = PartiallyFailedTools(state)
    client = FakeClient(response(call("set_thresholds", {"power_factor": 90})), TimeoutError("private-api-details"))
    result = core.run_turn("역률 90", state, tools, client=client)
    assert len(tools.calls) == 1 and len(state["threshold_history"]) == 1
    assert "private-" not in json.dumps(result)
    assert result["mode"] == "local"


def test_identical_mutations_with_new_ids_are_not_reapplied(state):
    client = FakeClient(response(call("set_thresholds", {"power_factor": 90}, "one"),
                                 call("set_thresholds", {"power_factor": 90}, "two")), response(text="완료"))
    tools = FakeTools(state)
    core.run_turn("역률 90", state, tools, client=client)
    assert len(tools.calls) == 1
    assert len([i for i in client.requests[1]["input"] if i.get("type") == "function_call_output"]) == 2


def test_failure_before_tools_uses_real_local_tools_and_unknown_question_never_writes(state):
    tools = FakeTools(state)
    result = core.run_turn("오늘 문제 있어?", state, tools, client=FakeClient(TimeoutError("private")))
    assert result["mode"] == "local" and tools.calls[0][0] == "get_anomalies"
    tools.calls.clear()
    core.run_turn("점심 뭐 먹어", state, tools)
    assert not tools.calls and not state["report_draft"]


def test_ten_complete_turns_and_tool_pairs_survive_history_trimming(state):
    tools = FakeTools(state)
    for n in range(12):
        client = FakeClient(response(call(ident=f"c{n}")), response(text=f"완료 {n}"))
        core.run_turn(f"오늘 {n}", state, tools, client=client)
    assert len(state["messages"]) == 20 and len(state["_agent_turns"]) == 10
    for turn in state["_agent_turns"]:
        assert turn[0]["role"] == "user"
        assert {i["call_id"] for i in turn if i.get("type") == "function_call"} == {i["call_id"] for i in turn if i.get("type") == "function_call_output"}
    assert state["messages"][0]["content"] == "오늘 2"


def test_context_budget_drops_whole_old_turns(state, monkeypatch):
    state["_agent_turns"] = [[{"role": "user", "content": "x" * 100_000}, {"role": "assistant", "content": "옛 응답"}]]
    monkeypatch.setattr(core, "MAX_INPUT_CHARS", 20_000)
    client = FakeClient(response(text="새 응답"))
    core.run_turn("오늘", state, FakeTools(state), client=client)
    assert client.requests[0]["input"] == [{"role": "user", "content": "오늘"}]


def test_round_limit_is_five_and_reports_already_executed_results(state):
    client = FakeClient(*(response(call(ident=f"c{n}")) for n in range(5)))
    result = core.run_turn("오늘 문제", state, FakeTools(state), client=client)
    assert len(client.requests) == 5
    assert result["error"] == "tool_round_limit" and result["mode"] == "local"


@pytest.mark.parametrize("text,expected,date_value", [
    ("오늘 문제 있어?", "get_anomalies", "2025-05-31"),
    ("6월 3일은 어때?", "get_anomalies", "2025-06-03"),
    ("작년은 어때", "get_anomalies", "2024-05-31"),
    ("전력 top3", "get_power_ranking", "2025-04-29"),
])
def test_local_date_and_rank_intents(state, text, expected, date_value):
    tools = FakeTools(state)
    result = core.run_turn(text, state, tools)
    assert tools.calls[-1] == (expected, {"date": date_value, **({"equipment": None} if expected == "get_anomalies" else {})})
    if date_value != "2025-04-29":
        assert "판단할 수 없습니다" in result["text"] and "정상" not in result["text"]


def test_local_equipment_three_phases_and_context_followup(state):
    tools = FakeTools(state)
    core.run_turn("4호기 전압 알려줘", state, tools)
    assert [args["item"] for name, args in tools.calls if name == "get_measurements"] == ["phase_voltage_R", "phase_voltage_S", "phase_voltage_T"]
    core.run_turn("오늘 문제 있어", state, tools)
    core.run_turn("어디야?", state, tools)
    assert tools.calls[-1] == ("get_anomalies", {"date": "2025-05-31", "equipment": None})


def test_local_forecast_report_fields_threshold_change_and_reset(state):
    tools = FakeTools(state)
    result = core.run_turn("이번달 요금", state, tools)
    assert tools.calls[-1] == ("get_forecast", {"month": "2025-05", "equipment": None})
    assert "505시간" in result["text"] and "18,000원" in result["text"]
    core.run_turn("콘덴서 교체했어", state, tools)
    core.run_turn("점검자는 홍길동", state, tools)
    assert state["report_draft"]["action_taken"] == "콘덴서 교체했어"
    assert state["report_draft"]["extra_fields"] == {"점검자": "홍길동"}
    core.run_turn("역률 90으로 바꿔", state, tools)
    assert state["thresholds"]["power_factor"] == 90
    core.run_turn("되돌려", state, tools)
    assert state["thresholds"] == core.DEFAULT_THRESHOLDS


def test_threshold_question_does_not_mutate_and_control_is_not_action(state):
    tools = FakeTools(state)
    core.run_turn("역률 90이면 어떤가?", state, tools)
    core.run_turn("4호기 꺼줘", state, tools)
    core.run_turn("보고서가 뭔데?", state, tools)
    core.run_turn("역률 90으로 바꾸지 마", state, tools)
    core.run_turn("기본값이 뭐야?", state, tools)
    core.run_turn("점검자는 누구야?", state, tools)
    core.run_turn("역률 90으로 변경하면 어떻게 돼?", state, tools)
    core.run_turn("역률 90으로 안 바꿔", state, tools)
    core.run_turn("기본값으로 복원하면 어떻게 돼?", state, tools)
    assert not any(name in core.MUTATIONS for name, _ in tools.calls)


def test_polite_change_question_is_authorized_but_hypothetical_is_not(state):
    tools = FakeTools(state)
    core.run_turn("역률 90으로 바꿔줄래?", state, tools)
    assert state["thresholds"]["power_factor"] == 90
    assert core._mutation_allowed("set_thresholds", {"power_factor": 90}, "역률 90으로 바꿔줄래?")
    assert not core._mutation_allowed("set_thresholds", {"power_factor": 80}, "역률 90으로 바꿔줄래?")


@pytest.mark.parametrize("name,args,text", [
    ("set_thresholds", {"power_factor": 90}, "현재 역률 90이야?"),
    ("set_thresholds", {"power_factor": 90}, "역률 90으로 바꾸지 마"),
    ("create_report", {"date": "2025-04-29", "equipment": "12(4호기)", "action_taken": "교체 완료"}, "오늘 문제 있어?"),
])
def test_model_cannot_mutate_without_matching_user_intent(state, name, args, text):
    tools = FakeTools(state)
    client = FakeClient(response(call(name, args)), response(text="변경하지 않았습니다."))
    result = core.run_turn(text, state, tools, client=client)
    assert not tools.calls and "error" in result["trace"][0]["result"]


def test_report_request_does_not_allow_fabricated_action_or_person(state):
    assert not core._mutation_allowed("create_report", {"action_taken": "모터 교체 완료"}, "보고서 만들어줘", state)
    assert not core._mutation_allowed("create_report", {"extra_fields": [{"name": "점검자", "value": "허구의 담당자"}]}, "보고서 만들어줘", state)
    assert core._mutation_allowed("create_report", {"action_taken": "콘덴서 교체했어"}, "콘덴서 교체했어", state)
    state["report_draft"] = {"rows": [{"name": "조치 내용", "source": "user", "value": "콘덴서 교체"}]}
    assert core._mutation_allowed("create_report", {"action_taken": "콘덴서 교체", "extra_fields": [{"name": "점검자", "value": "홍길동"}]}, "점검자는 홍길동", state)


def test_large_report_never_erases_current_state_snapshot(state):
    state["report_draft"] = {"evidence": "x" * 200_000,
                             "rows": [{"name": f"필드{n}", "source": "user", "value": "y" * 20_000} for n in range(100)]}
    instructions = core._instructions(state)
    assert '"view_date":"2025-04-29"' in instructions
    assert '"power_factor":85.0' in instructions
    assert len(instructions) < 20_000


def test_large_anomaly_payload_retains_full_counts_and_marks_sampled_items():
    item = {"equipment": "12(4호기)", "item": "voltageR", "timestamp": "2025-04-29T00:00:00",
            "end_timestamp": "2025-04-29T00:00:05", "last_observed_at": "2025-04-29T00:00:00",
            "value": 204.1234, "threshold": 210, "deviation_pct": 2.8, "duration_min": 1/12,
            "duration_seconds": 5, "record_count": 1, "source": "rule", "unit": "V", "note": "관측 정보 " * 100}
    result = {"count": 1647, "ai_count": 459, "items": [item]*1647, "ai_items": [item]*459,
              "record_count": 224640, "data_available": True}
    output = core._tool_output(result)
    parsed = json.loads(output)
    assert len(output) <= core.MAX_TOOL_CHARS
    payload = parsed.get("result", parsed)
    assert payload["sampled_intervals"] is True and payload["count"] == 1647
    assert payload["ai_count"] == 459
    assert 0 < len(payload["items"]) <= 6
    assert payload["affected_equipment"] == [{"equipment": "12(4호기)", "equipment_name": "4호기", "rule_intervals": 1647, "ai_intervals": 459}]
    assert len(result["items"]) == 1647


def test_successful_observation_syncs_view_and_invalidates_old_draft(state):
    state["report_draft"] = {"equipment": "12(4호기)"}
    state["export_bytes"] = b"old-report"
    core._sync_view(state, "get_measurements", {"date": "2025-04-28", "equipment": "13(3호기)", "data_available": True})
    assert state["selected_equipment"] == "13(3호기)" and state["view_date"] == "2025-04-28"
    assert state["report_draft"] is None and "export_bytes" not in state
    core._sync_view(state, "get_measurements", {"date": "2025-06-03", "data_available": False})
    assert state["view_date"] == "2025-04-28"


def test_unknown_equipment_never_uses_selected_equipment_silently(state):
    tools = FakeTools(state)
    result = core.run_turn("9호기 전압 알려줘", state, tools)
    assert [name for name, _ in tools.calls] == ["get_equipment_list"]
    assert "확인할 수 없습니다" in result["text"]
    assert state["selected_equipment"] == "12(4호기)"


def test_actual_service_shapes_for_event_counts_hourly_minimum_and_forecast(state):
    anomaly = core._format("get_anomalies", {"date": "2025-04-29", "data_available": True,
                          "count": 2, "items": [], "ai_count": 3, "ai_items": []})
    assert "AI 탐지 3구간" in anomaly and "설정 하한 미달 2구간" in anomaly
    measurements = core._format("get_measurements", {"min": [None, 212, 209], "threshold": 210, "unit": "V"})
    assert "209.00 V" in measurements
    forecast = core._format("get_forecast", {"total_kwh": 100, "cost": 18000, "co2": 42.4,
                           "period_start": "2025-05-10T00:00:00", "period_end": "2025-05-31T00:00:00", "hours": 505})
    assert "18,000원" in forecast and "505시간" in forecast


def test_last_year_month_does_not_select_current_year_forecast(state):
    tools = FakeTools(state)
    result = core.run_turn("작년 5월 요금", state, tools)
    assert tools.calls[-1] == ("get_forecast", {"month": "2024-05", "equipment": None})
    assert "예측 자료가 없습니다" in result["text"]


def test_invalid_json_and_unknown_function_are_returned_without_execution(state):
    invalid = call()
    invalid["arguments"] = "not-json-secret"
    client = FakeClient(response(invalid, call("unregistered", {}, "c2")), response(text="확인 필요"))
    tools = FakeTools(state)
    result = core.run_turn("오늘", state, tools, client=client)
    assert not tools.calls
    assert "not-json-secret" not in json.dumps(result["trace"])
    assert all("error" in json.loads(item["output"]) for item in client.requests[1]["input"] if item.get("type") == "function_call_output")


def test_report_draft_survives_ten_turn_history(state):
    tools = FakeTools(state)
    core.run_turn("콘덴서 교체했어", state, tools)
    for _ in range(12):
        core.run_turn("오늘 문제", state, tools)
    assert state["report_draft"]["action_taken"] == "콘덴서 교체했어"


class MeasurementTools(FakeTools):
    def execute(self, name, args):
        if name != "get_measurements":
            return super().execute(name, args)
        self.calls.append((name, copy.deepcopy(args)))
        phase = args["item"].split("_")[-1]
        values = {"R": (191.690002, "06:05:55"), "S": (193.309998, "06:31:30"), "T": (192.669998, "14:19:50")}
        value, time = values[phase]
        return {**args, "item": "voltage" + phase, "data_available": True, "unit": "V", "threshold": 210,
                "coverage": 1, "min": [value], "note": "LONG_TECHNICAL_NOTE " * 100,
                "summary": {"minimum_observed": value, "minimum_observed_timestamp": args["date"] + "T" + time,
                            "minimum_hourly_mean": 221.23, "minimum_hour": args["date"] + "T01:00:00"}}


def test_three_phases_are_one_readable_historical_answer_with_exact_minimum_times(state):
    result = core.run_turn("3호기 전압 어때?", state, MeasurementTools(state))
    text = result["text"]
    assert text.startswith("2025-04-29 3호기에서 설정 하한보다 낮아진 기록")
    assert all(s in text for s in ("R상 전압: 최저 191.69 V · 06:05:55", "S상 전압: 최저 193.31 V · 06:31:30", "T상 전압: 최저 192.67 V · 14:19:50"))
    assert text.count("선택일의 최저 기록") == 1
    assert "실시간 상태나 하루 종일의 상태를 뜻하지 않습니다" in text
    assert len(text) < 750
    assert not any(s in text for s in ("13(3호기)", "voltageR", "phase_voltage", "LONG_TECHNICAL_NOTE", "221.23", "T06", "로컬 도구 모드"))
    assert state["messages"][-1]["mode"] == "local"
    assert len([entry for entry in result["trace"] if entry["name"] == "get_measurements"]) == 3


@pytest.mark.parametrize("followup", ["언제 그랬어?", "얼마나 낮아?", "기준보다 얼마나 낮아?"])
def test_measurement_followups_requery_same_equipment_and_show_times_or_deficits(state, followup):
    tools = MeasurementTools(state)
    core.run_turn("3호기 전압 어때?", state, tools)
    tools.calls.clear()
    result = core.run_turn(followup, state, tools)
    assert len(tools.calls) == 3 and all(name == "get_measurements" for name, _ in tools.calls)
    assert all(args["equipment"] == "13(3호기)" and args["date"] == "2025-04-29" for _, args in tools.calls)
    assert all(t in result["text"] for t in ("06:05:55", "06:31:30", "14:19:50"))
    if "얼마나" in followup:
        assert all(s in result["text"] for s in ("18.31 V 낮음", "16.69 V 낮음", "17.33 V 낮음"))


@pytest.mark.parametrize("key,value", [("view_date", "2025-04-28"), ("selected_equipment", "12(4호기)")])
def test_measurement_followup_does_not_reuse_focus_after_ui_changes(state, key, value):
    tools = MeasurementTools(state)
    core.run_turn("3호기 전압 어때?", state, tools)
    state[key] = value
    tools.calls.clear()
    result = core.run_turn("언제 그랬어?", state, tools)
    assert not tools.calls and "191.69" not in result["text"]
    assert "현재 선택한 설비" in result["text"]


@pytest.mark.parametrize("question", ["4호기는 언제 낮았어?", "4월 28일은 언제 그랬어?", "요금은 언제 내?"])
def test_short_measurement_focus_never_hijacks_new_scope_or_topic(state, question):
    tools = MeasurementTools(state)
    core.run_turn("3호기 전압 어때?", state, tools)
    tools.calls.clear()
    result = core.run_turn(question, state, tools)
    assert not any(name == "get_measurements" for name, _ in tools.calls)
    assert "191.69" not in result["text"] and "06:05:55" not in result["text"]
    if "요금" in question:
        assert "납부일 정보는 현재 자료에 없습니다" in result["text"]


def test_anomaly_answer_hides_ai_score_keeps_trace_and_formats_rule_values(state):
    payload = {"date": "2025-04-29", "data_available": True, "count": 1, "ai_count": 1,
               "items": [{"equipment": "17(6호기)", "item": "powerFactorR", "timestamp": "2025-04-29T01:02:03",
                          "value": 83.456789, "threshold": 85, "unit": "%"}],
               "ai_items": [{"equipment": "13(3호기)", "item": "AI", "timestamp": "2025-04-29T02:03:04",
                             "value": .679675123, "threshold": .61, "unit": "anomaly_score"}],
               "note": "LONG_TECHNICAL_NOTE " * 100}
    class AnomalyTools(FakeTools):
        def execute(self, name, args):
            self.calls.append((name, args))
            return copy.deepcopy(payload)
    result = core.run_turn("오늘 문제 있어?", state, AnomalyTools(state))
    text = result["text"]
    assert "하한 미달 1구간" in text and "AI 탐지 1구간" in text
    assert "6호기 R상 역률: 01:02:03부터 기록된 구간 · 구간 최저 83.46 %" in text
    assert "3호기: AI가 특이한 패턴을 감지한 구간 · 시작 02:03:04" in text
    assert not any(s in text for s in ("powerFactorR", "17(6호기)", "T01", "0.68", ".679", "anomaly_score", "초과", "LONG_TECHNICAL_NOTE"))
    assert result["trace"][0]["result"] == payload
    assert "고장 확정을 뜻하지 않습니다" in text


def test_hourly_minimum_does_not_invent_an_exact_observation_second():
    text = core._format("get_measurements", {"date": "2025-04-29", "equipment": "13(3호기)",
                        "item": "voltageR", "min": [209.111], "hours": ["2025-04-29T01:00:00"],
                        "threshold": 210, "unit": "V", "data_available": True})
    assert "209.11 V" in text and "정확한 시각 정보 없음" in text
    assert "01:00:00" not in text


def test_equal_threshold_and_missing_data_do_not_become_fault_or_healthy_claims():
    equal = core._format("get_measurements", {"min": 210, "threshold": 210, "unit": "V", "item": "voltageR"})
    assert "하한 미달을 찾지 못했습니다" in equal and "낮아진 기록" not in equal
    absent = core._format_measurements([{"equipment": "13(3호기)", "date": "2025-05-31", "item": f"voltage{p}",
                                        "data_available": False, "note": "LONG_TECHNICAL_NOTE"} for p in "RST"])
    assert absent.count("판단할 수 없습니다") == 1 and "정상" not in absent
    assert "LONG_TECHNICAL_NOTE" not in absent


def test_api_failure_after_measurement_calls_formats_one_three_phase_answer(state):
    calls = [call("get_measurements", {"date": "2025-04-29", "equipment": "13(3호기)", "item": f"phase_voltage_{phase}"}, phase) for phase in "RST"]
    client = FakeClient(response(*calls), TimeoutError("private"))
    result = core.run_turn("3호기 전압 어때?", state, MeasurementTools(state), client=client)
    assert result["mode"] == "local" and result["text"].count("선택일의 최저 기록") == 1
    assert "R상 전압: 최저 191.69 V" in result["text"]
    assert "voltageR" not in result["text"] and "로컬 도구 모드" not in result["text"]


def test_provider_adapter_is_the_only_call_boundary_and_can_be_replaced(state, monkeypatch):
    requests = []
    def different_provider(messages, tools, **options):
        requests.append((copy.deepcopy(messages), copy.deepcopy(tools), options["instructions"]))
        if len(requests) == 1:
            return response(call("get_equipment_list", {}, "equipment"))
        return response(text="목록에서 확인했습니다.")
    monkeypatch.setattr(core, "call_model", different_provider)
    result = core.run_turn("설비가 몃 개야?", state, FakeTools(state), api_key="mock-placeholder")
    assert result["mode"] == "openai" and len(requests) == 2
    assert result["trace"][0]["name"] == "get_equipment_list"
    assert "몃 개야?" in requests[0][0][-1]["content"]  # Natural text reaches model.
    assert '"today":"2025-05-31"' in requests[0][2]
    assert '"view_date":"2025-04-29"' in requests[0][2]


def test_adapter_reuses_one_owned_client_for_a_turn_then_closes_it(state, monkeypatch):
    clients = []
    client = FakeClient(response(call()), response(text="조회 완료"))
    client.closed = False
    client.close = lambda: setattr(client, "closed", True)
    def factory(**kwargs):
        clients.append(kwargs)
        return client
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    core.run_turn("선택일 이상", state, FakeTools(state), api_key="unit-test-placeholder")
    assert len(clients) == 1 and client.closed
    assert clients[0]["timeout"] == 20 and clients[0]["max_retries"] == 0
    assert "unit-test-placeholder" not in json.dumps(state)


@pytest.mark.parametrize("tool,payload", [
    ("get_forecast", {"hours": 505, "period_start": "2025-05-10", "period_end": "2025-05-31",
                      "total_kwh": 123.0, "cost": 22140, "series": [{"timestamp": "2025-05-10T00:00:00", "kwh": 9.99}]}),
    ("get_measurements", {"item": "voltageR", "min": 191.69, "max": 230, "min_timestamp": "2025-04-29T06:05:55",
                          "hours": ["06:00"], "hourly_min": [191.69], "hourly_max": [230], "mean": [221],
                          "series": [{"hour": 6, "value": 221}], "summary": {"minimum_observed": 191.69,
                          "minimum_observed_timestamp": "2025-04-29T06:05:55", "maximum_observed": 230}}),
    ("get_power", {"total_kwh": 100, "cost": 18000, "co2": 42.4, "peak_kw": 5,
                   "peak_timestamp": "2025-04-29T07:00:05", "series": [{"t": "07:00", "kwh": 5}]})
])
def test_even_small_payloads_never_send_time_series(tool, payload):
    original = copy.deepcopy(payload)
    encoded = core._tool_output(payload, tool)
    parsed = json.loads(encoded)
    assert len(json.dumps(payload)) < core.MAX_TOOL_CHARS
    assert "series" not in parsed and "hourly_min" not in parsed and "hourly_max" not in parsed
    assert "mean" not in parsed and parsed["summary_only"] is True
    if tool == "get_forecast":
        assert parsed["hours"] == 505 and parsed["total_kwh"] == 123
    if tool == "get_measurements":
        assert parsed["min"] == 191.69 and parsed["max"] == 230
        assert parsed["summary"]["minimum_observed_timestamp"] == "2025-04-29T06:05:55"
    assert payload == original


def test_old_history_and_nested_report_evidence_are_minimized_before_api(state):
    old_result = {"hours": 505, "total_kwh": 100, "series": [{"timestamp": "private-observation", "kwh": 1}]}
    state["_agent_turns"] = [[{"role": "user", "content": "이전 요금"},
                              call("get_forecast", {"month": "2025-05", "equipment": None}, "old"),
                              {"type": "function_call_output", "call_id": "old", "output": json.dumps(old_result)},
                              {"role": "assistant", "content": "이전 결과"}]]
    client = FakeClient(response(text="확인했습니다."))
    core.run_turn("설명해줘", state, FakeTools(state), client=client)
    inputs = json.dumps(client.requests[0]["input"])
    assert "private-observation" not in inputs
    assert 'hours' in inputs and '505' in inputs
    payload = {"rows": [{"name": "전력", "source": "auto", "value": "100kWh"}],
               "evidence": {"series": ["sensitive-detail"]}, "measurements": {"R": {"series": [1, 2], "min": 1}}}
    parsed = json.loads(core._tool_output(payload, "create_report"))
    assert "evidence" not in parsed and "measurements" not in parsed
    assert parsed["rows"][0]["value"] == "100kWh"


def test_anomaly_summary_remains_idempotent_across_ten_turn_history():
    payload = {"count": 200, "ai_count": 30,
               "items": [{"equipment": "13(3호기)", "item": "voltageR", "timestamp": str(n)} for n in range(200)],
               "ai_items": [{"equipment": "12(4호기)", "item": "AI", "timestamp": str(n)} for n in range(30)]}
    first = json.loads(core._tool_output(payload, "get_anomalies"))
    second = json.loads(core._tool_output(first, "get_anomalies"))
    assert second["affected_equipment"] == first["affected_equipment"]
    assert second["count"] == 200 and second["ai_count"] == 30


@pytest.mark.parametrize("class_name,status,expected", [
    ("AuthenticationError", 401, "api_authentication"),
    ("PermissionDeniedError", 403, "api_permission"),
    ("RateLimitError", 429, "api_rate_or_quota"),
    ("APITimeoutError", None, "api_timeout"),
    ("APIConnectionError", None, "api_connection"),
    ("BadRequestError", 400, "api_request"),
    ("InternalServerError", 503, "api_server"),
])
def test_safe_error_classification_never_retains_provider_body(state, class_name, status, expected):
    error_type = type(class_name, (Exception,), {})
    error = error_type("secret-key-and-private-request-body")
    error.status_code = status
    error.body = {"message": "private-data"}
    result = core.run_turn("설비 몇 개야?", state, FakeTools(state), client=FakeClient(error))
    assert result["error"] == expected
    assert result["error_info"] == {"code": expected, "class": class_name, "status": status}
    assert "secret-key" not in json.dumps(result) and "private-data" not in json.dumps(state)
    assert result["mode"] == "local" and result["trace"][0]["name"] == "get_equipment_list"
    if expected == "api_timeout":
        assert "응답이 지연되고 있습니다. 다시 시도할까요?" in result["text"]


def test_equipment_count_and_unregistered_number_have_no_sensor_side_effects(state):
    tools = FakeTools(state)
    count = core.run_turn("설비 몇 대야?", state, tools)
    assert "총 2개" in count["text"] and tools.calls == [("get_equipment_list", {})]
    tools.calls.clear()
    missing = core.run_turn("14호기 전압 어때?", state, tools)
    assert "확인할 수 없습니다" in missing["text"]
    assert tools.calls == [("get_equipment_list", {})]
    assert state["selected_equipment"] == "12(4호기)"


def test_today_is_fixed_but_unspecified_date_uses_dashboard_selection(state):
    assert core._resolve_date("오늘", state) == "2025-05-31"
    assert core._resolve_date("어제", state) == "2025-05-30"
    assert core._resolve_date("3호기 전압", state) == "2025-04-29"
    state["view_date"] = "2025-04-12"
    assert core._resolve_date("오늘", state) == "2025-05-31"
    assert core._resolve_date("3호기 전압", state) == "2025-04-12"


def test_shared_view_normalizes_date_and_canonical_name_not_display_alias(state):
    core._set_view(state, view_date="2025-04-28T13:14:15", selected_equipment="13 (3호기)")
    assert state["view_date"] == "2025-04-28" and state["selected_equipment"] == "13(3호기)"
    core._set_view(state, view_date="bad-date", selected_equipment="10호기")
    assert state["view_date"] == "2025-04-28" and state["selected_equipment"] == "13(3호기)"


def test_threshold_reason_changes_do_not_apply_identical_mutation_twice(state):
    client = FakeClient(response(call("set_thresholds", {"power_factor": 90, "reason": "사용자 요청"}, "one"),
                                 call("set_thresholds", {"power_factor": 90, "reason": "기준 변경 요청"}, "two")), response(text="완료"))
    tools = FakeTools(state)
    core.run_turn("역률 90으로 올려줘", state, tools, client=client)
    assert len(tools.calls) == 1 and len(state["threshold_history"]) == 1


def test_model_can_change_topic_to_factory_forecast_after_three_phase_measurement(state):
    tools = MeasurementTools(state)
    calls = [call("get_measurements", {"date": "2025-04-29", "equipment": "13(3호기)", "item": f"phase_voltage_{p}"}, p) for p in "RST"]
    core.run_turn("3호끼 전압 어때?", state, tools, client=FakeClient(response(*calls), response(text="선택일 3호기 전압 결과입니다.")))
    client = FakeClient(response(call("get_forecast", {"month": "2025-05", "equipment": None}, "factory")), response(text="공장 전체 예측입니다."))
    result = core.run_turn("전체 요금은?", state, tools, client=client)
    assert result["mode"] == "openai"
    assert tools.calls[-1] == ("get_forecast", {"month": "2025-05", "equipment": None})
    assert '"selected_equipment":"13(3호기)"' in client.requests[0]["instructions"]
    tool_results = [json.loads(x["output"]) for x in client.requests[0]["input"] if x.get("type") == "function_call_output"]
    assert all("series" not in value and "hours" not in value for value in tool_results)


def test_successful_threshold_change_without_sensor_data_is_not_reported_as_failed():
    text = core._format("set_thresholds", {"after": {"phase_voltage": 210, "line_voltage": 360, "power_factor": 90},
                        "anomaly_count_before": None, "anomaly_count_after": None, "data_available": False})
    assert "설정 기준을 변경했습니다" in text and "역률 90.00%" in text
    assert "변화는 확인할 수 없습니다" in text and "None" not in text


def test_original_contract_display_name_keeps_canonical_followup_scope(state):
    class ContractTools(MeasurementTools):
        def execute(self, name, args):
            if name == "get_equipment_list":
                self.calls.append((name, args))
                return [{"channel": "13", "name": "13 (3호기)", "equipment": "13(3호기)"}]
            return super().execute(name, args)
    tools = ContractTools(state)
    core.run_turn("3호기 전압 어때?", state, tools)
    assert state["selected_equipment"] == state["_last_measurement_query"]["equipment"] == "13(3호기)"
    tools.calls.clear()
    result = core.run_turn("언제 그랬어?", state, tools)
    assert len(tools.calls) == 3 and "06:05:55" in result["text"]


def test_dashboard_anomaly_view_is_forwarded_without_overwriting_explicit_range(state):
    state["anomaly_view"] = "monthly"
    assert '"view_range":"monthly"' in core._instructions(state)
    state["view_range"] = "daily"
    assert '"view_range":"daily"' in core._instructions(state)


def test_explicit_completed_action_uses_human_verbatim_without_confirmation_loop(state):
    tools = FakeTools(state)
    client = FakeClient(response(call("create_report", {"date": "2025-04-29", "equipment": "12(4호기)",
                      "action_taken": "콘덴서 교체 완료", "extra_fields": None})), response(text="초안을 만들었습니다."))
    result = core.run_turn("콘덴서 교체했어", state, tools, client=client)
    assert result["mode"] == "openai" and len(tools.calls) == 1
    assert tools.calls[0][1]["action_taken"] == "콘덴서 교체했어"
    assert state["report_draft"]["action_taken"] == "콘덴서 교체했어"


def test_model_cannot_rename_user_inspection_time_to_action_time(state):
    text = "3호기 보고서 초안을 만들어줘. 점검자는 김테스트고 점검 일시는 2025-04-29 15:00이야."
    wrong = {"date": "2025-04-29", "equipment": "13(3호기)", "action_taken": "",
             "extra_fields": [{"name": "점검자", "value": "김테스트"}, {"name": "조치 시각", "value": "2025-04-29 15:00"}]}
    right = copy.deepcopy(wrong)
    right["extra_fields"][1]["name"] = "점검 일시"
    tools = FakeTools(state)
    client = FakeClient(response(call("create_report", wrong, "wrong")), response(call("create_report", right, "right")), response(text="입력했습니다."))
    result = core.run_turn(text, state, tools, client=client)
    assert result["mode"] == "openai" and len(tools.calls) == 1
    assert tools.calls[0][1]["extra_fields"][1]["name"] == "점검 일시"
    assert "retry_instruction" in result["trace"][0]["result"]


def test_report_model_payload_preserves_all_rows_and_missing_fields_without_evidence(state):
    rows = [{"name": f"항목{n}", "value": 210.12345 if n % 2 else "사용자 입력", "source": "user" if n % 2 else "auto",
             "required": bool(n % 3), "coordinates": "unneeded-layout"} for n in range(80)]
    rows[-1] = {"name": "점검 일시", "value": None, "source": "user", "required": True}
    payload = {"date": "2025-04-29", "equipment": "13(3호기)", "rows": rows, "missing_user_fields": ["점검 일시"],
               "ready_to_export": False, "persisted": False, "context": {"private": "x"*400_000},
               "events": [{"equipment": "13(3호기)", "item": "voltageR", "timestamp": str(n), "value": 200} for n in range(528)],
               "ai_events": [{"equipment": "13(3호기)", "item": "AI", "timestamp": str(n)} for n in range(88)]}
    encoded = core._tool_output(payload, "create_report")
    result = json.loads(encoded)
    assert len(encoded) < core.MAX_TOOL_CHARS and len(result["rows"]) == 80
    assert result["rows"][-1]["value"] is None and result["missing_user_fields"] == ["점검 일시"]
    assert "context" not in result and "events" not in result and "ai_events" not in result
    assert result["anomaly_summary"]["count"] == 528 and result["anomaly_summary"]["ai_count"] == 88
    assert json.loads(core._tool_output(result, "create_report"))["anomaly_summary"] == result["anomaly_summary"]


def test_equipment_names_remove_channel_prefix_without_inventing_machine_number():
    payload = {"items": [{"equipment": "4(분쇄기(1))", "kwh": 72.48}, {"equipment": "12(4호기)", "kwh": 72.1}]}
    summary = json.loads(core._tool_output(payload, "get_power_ranking"))
    assert summary["items"][0]["equipment_name"] == "분쇄기(1)"
    assert summary["items"][1]["equipment_name"] == "4호기"


def test_report_storage_and_actual_export_blockers_survive_model_summary(state):
    report = {"date": "2025-04-29", "equipment": "13(3호기)", "rows": [],
              "missing_user_fields": ["종합 판정"], "missing_auto_fields": ["전월 대비 전력량"],
              "export_blockers": ["종합 판정"], "storage_status": "session_draft_only",
              "persisted": False, "ready_to_export": False}
    summary = json.loads(core._tool_output(report, "create_report"))
    assert summary["storage_status"] == "session_draft_only"
    assert summary["export_blockers"] == ["종합 판정"]
    state["report_draft"] = report
    instructions = core._instructions(state)
    assert '"storage_status":"session_draft_only"' in instructions
    assert '"export_blockers":["종합 판정"]' in instructions
    status = core._report_status(report)
    assert "파일 저장·전송은 하지 않았습니다" in status and "종합 판정" in status
    assert not any(word in status for word in ("전월", "session_draft_only", "export_blockers", "persisted"))


def test_optional_auto_fields_do_not_block_grounded_export_status():
    report = {"storage_status": "session_draft_only", "missing_user_fields": [],
              "missing_auto_fields": ["계약전력 대비", "전월 비교"], "export_blockers": [], "ready_to_export": True}
    status = core._report_status(report)
    assert "내보낼 수 있습니다" in status and "전월" not in status and "계약" not in status
    assert "저장·전송은 하지 않았습니다" in status


def test_grounded_report_status_is_separate_from_model_prose_and_has_no_json_keys(state):
    tools = FakeTools(state)
    client = FakeClient(response(call("create_report", {"date": "2025-04-29", "equipment": "12(4호기)",
                        "action_taken": "", "extra_fields": None})), response(text="초안에 반영했습니다."))
    result = core.run_turn("보고서 초안 만들어줘", state, tools, client=client)
    assert result["text"] == "초안에 반영했습니다."
    assert "조치자, 종합 판정" in result["report_status"]
    assert "report_status" not in json.dumps(state["messages"], ensure_ascii=False)
    source_status = core._report_status({"export_blockers": ["source_identity_mismatch"], "ready_to_export": False})
    assert "자료 출처 일치 확인" in source_status and "source_identity" not in source_status


def grouped_payload(day='2025-04-29', available=True):
    modules = ['1(PM-3)', '2(L-1전등)', '3(분쇄기(2))', '4(분쇄기(1))', '5(좌측분전반)',
               '11(우측분전반1)', '12(4호기)', '13(3호기)', '14(2호기)', '15(예비건조기)',
               '16(호이스트)', '17(6호기)', '18(우측분전반2)']
    items = []
    if available:
        for number, module in enumerate(modules, 1):
            items.append({'equipment': module, 'measurements': [
                {'item': f'phase_voltage_{phase}', 'unit': 'V', 'threshold': 210,
                 'minimum_observed': 200 + number + offset,
                 'minimum_observed_timestamp': day + f'T06:{number:02d}:{offset:02d}',
                 'maximum_observed': 240 + number + offset, 'data_available': True,
                 'record_count': 17280, 'coverage': 1.0}
                for offset, phase in enumerate('RST')]})
    return {'date': day, 'equipment': None, 'item': 'phase_voltage', 'grouped': True,
            'data_available': available, 'items': items, 'equipment_count': 13,
            'latest_available_date': '2025-04-29', 'note': '설비별 관측 요약'}


class BulkTools(FakeTools):
    def execute(self, name, args):
        if name != 'get_measurements' or args.get('equipment') != '전체':
            return super().execute(name, args)
        self.calls.append((name, copy.deepcopy(args)))
        return grouped_payload(args['date'], args['date'] == '2025-04-29')


@pytest.mark.parametrize('query', ['전체 설비 전압 보여줘', '모든 설비 전압 어때?', '설비별 전압 다 알려줘'])
def test_local_bulk_is_one_call_and_keeps_all_thirteen_equipment(state, query):
    tools = BulkTools(state)
    result = core.run_turn(query, state, tools)
    assert tools.calls == [('get_measurements', {'date': '2025-04-29', 'equipment': '전체', 'item': 'phase_voltage'})]
    assert '13개 설비 중 9개' in result['text']
    assert '| PM-3 | 201.00 V · 06:01:00' in result['text']
    assert '| 우측분전반2 | 213.00 V · 06:13:00' in result['text']
    assert '| R상 전압 | S상 전압 | T상 전압 |' in result['text']
    assert '실시간 상태나 하루 종일' in result['text']
    assert len(result['trace'][0]['result']['items']) == 13
    assert state['selected_equipment'] == '12(4호기)'


def test_bulk_missing_date_does_not_switch_to_latest_or_claim_normal(state):
    tools = BulkTools(state)
    result = core.run_turn('오늘 전체 설비 전압 알려줘', state, tools)
    assert len(tools.calls) == 1 and tools.calls[0][1]['date'] == '2025-05-31'
    assert state['view_date'] == '2025-04-29'
    assert '2025-05-31' in result['text'] and '2025-04-29' in result['text']
    assert '조회 날짜는 바꾸지 않았습니다' in result['text'] and '정상' not in result['text']


def test_group_payload_always_retains_every_equipment_and_partial_coverage():
    payload = grouped_payload()
    for row in payload['items']:
        for reading in row['measurements']:
            reading['note'] = 'unneeded verbose source information ' * 2000
            reading['under_threshold_hours'] = list(range(24))
            reading['series'] = [{'timestamp': str(n), 'mean': n} for n in range(24)]
    for reading in payload['items'][-1]['measurements']:
        reading.update(minimum_observed=None, minimum_observed_timestamp=None, maximum_observed=None,
                       data_available=False, record_count=0, coverage=0)
    payload['items'][0]['measurements'][0]['coverage'] = .5
    encoded = core._tool_output(payload, 'get_measurements')
    summary = json.loads(encoded)
    assert len(encoded) < core.MAX_TOOL_CHARS and 'truncated' not in summary
    assert len(summary['items']) == summary['equipment_count'] == 13
    assert summary['available_equipment_count'] == 12
    assert summary['requested_sensor_count'] == 39 and summary['observed_sensor_count'] == 36
    assert summary['items'][0]['readings']['R']['coverage'] == .5
    assert summary['items'][-1]['readings']['R']['available'] is False
    assert summary['items'][-1]['readings']['R']['below'] is None
    assert summary['items'][0]['readings']['R']['min'] == 201
    assert summary['items'][1]['readings']['R']['min'] == 202  # Never combined across equipment.
    assert 'unneeded verbose' not in encoded and 'series' not in encoded and 'under_threshold_hours' not in encoded
    assert json.loads(core._tool_output(summary, 'get_measurements')) == summary
    text = core._format('get_measurements', payload)
    assert '| 우측분전반2 | 자료 없음 | 자료 없음 | 자료 없음 |' in text
    assert '일부 항목·시간대의 기록이 없습니다' in text


def test_bulk_failure_after_api_tool_preserves_table_and_never_requeries(state):
    tools = BulkTools(state)
    client = FakeClient(response(call('get_measurements', {'date': '2025-04-29', 'equipment': '전체', 'item': 'phase_voltage'}, 'all')),
                        TimeoutError('private'))
    result = core.run_turn('전체 설비 전압 어때?', state, tools, client=client)
    assert len(tools.calls) == 1 and result['mode'] == 'local'
    assert '| PM-3 |' in result['text'] and '| 우측분전반2 |' in result['text']
    transmitted = [json.loads(item['output']) for item in client.requests[1]['input'] if item.get('type') == 'function_call_output'][0]
    assert len(transmitted['items']) == 13 and transmitted['model_summary'] is True


def test_no_data_group_summary_keeps_requested_scope_but_zero_observations():
    summary = json.loads(core._tool_output(grouped_payload('2025-06-03', False), 'get_measurements'))
    assert summary['equipment_count'] == 13 and summary['requested_sensor_count'] == 39
    assert summary['observed_sensor_count'] == summary['available_equipment_count'] == 0
    assert summary['latest_available_date'] == '2025-04-29' and summary['date'] == '2025-06-03'
