"""Streamlit integration checks with deterministic data and a mocked model call."""
import copy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP.parent))
from agent_runtime import agent_ui as ui  # noqa: E402
from agent_runtime.agent_reports import ReportService  # noqa: E402


class FakeData:
    modules = ["13(3호기)", "12(4호기)"]

    def normalize_equipment(self, value):
        return str(value).replace(" ", "") if value else None

    def get_anomalies(self, date, equipment=None, thresholds=None):
        return {"date": date, "equipment": equipment, "data_available": date.startswith("2025-04"), "count": 2 if date.startswith("2025-04") else None, "ai_count": 1 if date.startswith("2025-04") else None, "items": [], "ai_items": [], "source_id": "test-source", "note": ""}

    def get_measurements(self, date, equipment, item, thresholds=None):
        return {"date": date, "equipment": equipment, "item": item, "data_available": date.startswith("2025-04"), "representative_value": 220 if date.startswith("2025-04") else None, "representative_timestamp": date + "T12:00:05", "threshold": 210, "source_id": "test-source"}

    def get_power(self, date, equipment=None, granularity="hour"):
        return {"date": date, "equipment": equipment, "data_available": date.startswith("2025-04"), "total_kwh": 10 if date.startswith("2025-04") else None, "source_id": "test-source"}

    def get_equipment_list(self, date, thresholds=None):
        return [{"equipment": value, "name": value} for value in self.modules]


@pytest.fixture
def environment(monkeypatch):
    base = APP / "runtime" / "test_ui"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as temp:
        reports = ReportService(Path(temp))
        data = FakeData()
        monkeypatch.setattr(ui, "get_services", lambda: (data, reports))
        monkeypatch.setattr(ui, "load_settings", lambda: SimpleNamespace(api_key="", model="gpt-5-mini", note=""))
        yield data, reports


def app():
    path = ui.get_services()[1].app_dir / "ui_smoke.py"
    path.write_text("from agent_runtime.agent_ui import render_agent_tab\nrender_agent_tab()", encoding="utf-8")
    return AppTest.from_file(str(path), default_timeout=15)


def test_first_render_has_real_summary_no_api_call_and_disabled_export(environment, monkeypatch):
    monkeypatch.setattr(ui, "run_turn", lambda *args, **kwargs: pytest.fail("No model call before a user question"))
    at = app().run()
    assert not at.exception
    assert at.session_state["view_date"] == "2025-05-31"
    assert len(at.session_state["messages"]) == 1
    assert "실측 기록이 없어" in at.session_state["messages"][0]["content"]
    assert at.button(key="agent_prepare_export").disabled
    assert any("일일 점검 체크리스트" in item.label for item in at.expander)
    at.run()
    assert len(at.session_state["messages"]) == 1


def test_configured_model_is_used_without_manual_switch(environment, monkeypatch):
    captured = []
    monkeypatch.setattr(ui, "load_settings", lambda: SimpleNamespace(api_key="mock-test-only", model="gpt-5-mini", note=""))
    def model(prompt, state, tools, **kwargs):
        captured.append(kwargs)
        state["messages"].extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": "조회 결과입니다."}])
        return {"text": "조회 결과입니다.", "mode": "openai", "trace": []}
    monkeypatch.setattr(ui, "run_turn", model)
    at = app().run()
    at.chat_input(key="agent_chat_input").set_value("4호기 전압 어때?").run()
    assert not at.exception
    assert captured == [{"api_key": "mock-test-only", "model": "gpt-5-mini"}]
    assert at.session_state["last_mode"] == "openai"
    assert len(at.session_state["messages"]) == 3
    assert not at.radio


def test_failed_model_response_shows_demo_label(environment, monkeypatch):
    monkeypatch.setattr(ui, "load_settings", lambda: SimpleNamespace(api_key="mock-test-only", model="gpt-5-mini", note=""))
    def failed(prompt, state, tools, **kwargs):
        state["messages"].extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": "실제 자료의 로컬 조회 결과입니다."}])
        return {"text": "실제 자료의 로컬 조회 결과입니다.", "mode": "local", "trace": [], "error": "api_unavailable"}
    monkeypatch.setattr(ui, "run_turn", failed)
    at = app().run()
    at.chat_input(key="agent_chat_input").set_value("오늘 어때?").run()
    assert not at.exception
    assert any("데모 모드" in c.value and "마지막 API" in c.value for c in at.caption)


def test_report_confirmation_and_required_inputs_gate_download(environment):
    at = app().run()
    at.button(key="agent_create_report").click().run()
    assert not at.exception
    assert at.session_state["report_draft"]
    assert at.button(key="agent_prepare_export").disabled
    assert not at.session_state["template_asked"]
    at.button(key="agent_accept_default_template").click().run()
    assert at.session_state["template_asked"]
    for widget in at.text_input:
        if widget.label == "점검 일시 *":
            widget.set_value("2025-05-31 14:00")
        elif widget.label == "점검자 *":
            widget.set_value("검증 담당")
        elif widget.label == "종합 판정 *":
            widget.set_value("실측 자료 없음")
    next(button for button in at.button if button.label == "입력 내용을 보고서에 반영").click().run()
    assert not at.exception
    assert at.session_state["report_draft"]["ready_to_export"]
    assert not at.button(key="agent_prepare_export").disabled
    at.button(key="agent_prepare_export").click().run()
    assert not at.exception
    assert at.session_state["export_bytes"].startswith(b"PK")
    assert at.session_state["_agent_recent_fields"]


def test_template_request_opens_upload_and_does_not_apply_without_confirmation(environment, monkeypatch):
    def reply(prompt, state, tools, **kwargs):
        state["messages"].extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": "양식을 올려 주세요."}])
        return {"text": "양식을 올려 주세요.", "mode": "local", "trace": []}
    monkeypatch.setattr(ui, "run_turn", reply)
    at = app().run()
    at.chat_input(key="agent_chat_input").set_value("양식 바꿀게").run()
    assert not at.exception
    assert at.session_state["agent_template_panel_open"]
    assert environment[1].get_report_template()["is_default"]


def test_order_progress_highlighting_and_html_are_safe():
    rows = [{"name": "조치 내용", "source": "user", "cell": "C45", "value": "완료"}, {"name": "점검자", "source": "user", "cell": "C7", "value": "<script>alert(1)</script>"}, {"name": "설비명", "source": "auto", "cell": "C10", "value": "13(3호기)"}, {"name": "비고", "source": "user", "cell": "G17", "value": None}]
    assert [r["name"] for r in ui.ordered_rows(rows)] == ["점검자", "설비명", "비고", "조치 내용"]
    manual, done = ui.checklist(rows)
    assert len(manual) == 3 and done == 2
    markup = ui._preview_html(rows, {"점검자": 1})
    assert "user recent" in markup
    assert "&lt;script&gt;" in markup and "<script>" not in markup
    assert "(미입력)" in markup and "auto" in markup


def test_shared_date_and_equipment_changes_invalidate_old_draft(environment):
    data, reports = environment
    state = {"view_date": "2025-04-29", "selected_equipment": "13 (3호기)", "selected_year": 2025, "selected_month": 4, "selected_day": 29}
    ui._init_state(state, data, reports.app_dir, legacy_calendar=True)
    assert state["selected_equipment"] == "13(3호기)"
    state["report_draft"] = {"date": "2025-04-29", "equipment": "13(3호기)"}
    state["export_bytes"] = b"old"
    state["selected_month"], state["selected_day"] = 5, 31
    ui._init_state(state, data, reports.app_dir, legacy_calendar=True)
    assert state["view_date"] == "2025-05-31"
    assert state["report_draft"] is None and "export_bytes" not in state
    state["view_date"] = "2025-04-20"  # A chat query moves the shared calendar too.
    ui._init_state(state, data, reports.app_dir, legacy_calendar=True)
    assert (state["selected_month"], state["selected_day"]) == (4, 20)


def test_chat_date_reaches_calendar_before_rerun(environment, monkeypatch):
    state = {"view_date": "2025-05-31", "selected_year": 2025, "selected_month": 5, "selected_day": 31, "messages": [], "_agent_legacy_calendar": True}
    def model(prompt, state, tools, **kwargs):
        state["view_date"] = "2025-04-20"
        state["messages"].append({"role": "assistant", "content": "4월20일을 조회했습니다."})
        return {"mode": "openai", "trace": []}
    monkeypatch.setattr(ui, "run_turn", model)
    ui._respond("4월20일 기록", state, object(), SimpleNamespace(api_key="mock-test-only", model="gpt-5-mini"))
    assert (state["selected_year"], state["selected_month"], state["selected_day"]) == (2025, 4, 20)


def test_manual_template_requires_apply_and_restores_default(environment):
    at = app().run()
    at.text_input(key="agent_manual_template").set_value("점검자, 현장 메모").run()
    at.button(key="agent_parse_manual").click().run()
    assert not at.exception
    assert at.session_state["pending_template"]
    assert environment[1].get_report_template()["is_default"]
    at.button(key="agent_apply_template").click().run()
    assert not at.exception
    assert not environment[1].get_report_template()["is_default"]
    assert at.session_state["template_asked"]
    at.button(key="agent_reset_template").click().run()
    assert not at.exception
    assert environment[1].get_report_template()["is_default"]


def test_threshold_card_keeps_missing_counts_distinct_from_zero():
    unavailable = ui.threshold_change_summary({"before": {"power_factor": 85}, "after": {"power_factor": 90}, "anomaly_count_before": None, "anomaly_count_after": None, "data_available": False, "newly_abnormal": None})
    assert unavailable["changes"] == ["역률 85% → 90%"]
    assert "확인할 수 없습니다" in unavailable["counts"]
    assert "0 → 0" not in unavailable["counts"]
    available = ui.threshold_change_summary({"before": {"power_factor": 85}, "after": {"power_factor": 90}, "anomaly_count_before": 0, "anomaly_count_after": 3, "data_available": True, "newly_abnormal": ["13(3호기)"]})
    assert available["counts"] == "센서 기준 위반 구간 0 → 3개"
    assert available["newly"] == ["13(3호기)"]


def test_threshold_history_and_reset_use_real_tool_then_rerun(environment):
    at = app().run()
    at.session_state["thresholds"] = {**ui.DEFAULT_THRESHOLDS, "power_factor": 90}
    at.session_state["threshold_history"] = [{"at": "2025-05-31 10:00", "before": dict(ui.DEFAULT_THRESHOLDS), "after": {**ui.DEFAULT_THRESHOLDS, "power_factor": 90}, "reason": "사용자 요청"}]
    at.session_state["export_bytes"] = b"stale"
    at.run()
    assert not at.button(key="agent_reset_thresholds").disabled
    assert any("센서 기준·변경 이력" == item.label for item in at.expander)
    at.button(key="agent_reset_thresholds").click().run()
    assert not at.exception
    assert at.session_state["thresholds"] == ui.DEFAULT_THRESHOLDS
    assert len(at.session_state["threshold_history"]) == 2
    assert at.session_state["trace"][-1]["name"] == "set_thresholds"
    assert at.session_state["trace"][-1]["result"]["anomaly_count_after"] is None
    assert "export_bytes" not in at.session_state
    assert at.button(key="agent_reset_thresholds").disabled
    assert any("센서 기준 변경 결과" in item.value for item in at.markdown)


def test_optional_legacy_calendar_sync_does_not_oscillate(environment):
    data, reports = environment
    state = {"view_date": "2025-05-31", "selected_year": 2025, "selected_month": 5, "selected_day": 31}
    ui._init_state(state, data, reports.app_dir, legacy_calendar=True)
    state["view_date"] = "2025-04-29"  # Chat-driven date change.
    for _ in range(2):
        ui._init_state(state, data, reports.app_dir, legacy_calendar=True)
        assert state["view_date"] == "2025-04-29"
        assert (state["selected_month"], state["selected_day"]) == (4, 29)
    state.update(selected_month=5, selected_day=31)  # Sidebar-driven date change.
    for _ in range(2):
        ui._init_state(state, data, reports.app_dir, legacy_calendar=True)
        assert state["view_date"] == "2025-05-31"


def test_default_state_does_not_require_or_modify_host_calendar(environment, monkeypatch):
    data, reports = environment
    state = {"view_date": "2025-04-29", "selected_year": 2000, "selected_month": 1, "selected_day": 1}
    ui._init_state(state, data, reports.app_dir)
    assert state["view_date"] == "2025-04-29"
    assert (state["selected_year"], state["selected_month"], state["selected_day"]) == (2000, 1, 1)
    monkeypatch.setattr(ui, "run_turn", lambda *args, **kwargs: {"mode": "local", "trace": []})
    ui._respond("전압 조회", state, object(), SimpleNamespace(api_key="", model="gpt-5-mini"))
    assert state["selected_year"] == 2000
    empty = {}
    ui._init_state(empty, data, reports.app_dir)
    assert empty["view_date"] == "2025-05-31" and "selected_year" not in empty


def test_tab_accepts_injected_services_and_persistent_state(environment, monkeypatch):
    data, reports = environment
    state = {"view_date": "2025-04-29", "selected_equipment": "12(4호기)"}
    monkeypatch.setattr(ui, "get_services", lambda: pytest.fail("Explicit dependencies must be used"))
    monkeypatch.setattr(ui, "_injected_test_objects", (data, reports, state), raising=False)
    at = AppTest.from_string("from agent_runtime import agent_ui as ui\ndata, reports, state = ui._injected_test_objects\nui.render_agent_tab(data=data, reports=reports, state=state)", default_timeout=15).run()
    assert not at.exception
    assert state["view_date"] == "2025-04-29" and len(state["messages"]) == 1
    assert "selected_year" not in state


def test_ui_module_has_no_host_dashboard_import_or_page_entrypoint():
    import ast
    tree = ast.parse(Path(ui.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {"dashboard_app", "dashboard_data"}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "set_page_config"


def test_function_only_example_reuses_supplied_state_and_services(environment, monkeypatch):
    from agent_runtime.examples.integrate_agent import ask_agent, render_existing_agent_tab
    data, reports = environment
    state = {"view_date": "2025-04-29", "selected_equipment": "12(4호기)"}
    reply = ask_agent("현재 센서 기준 알려줘", data, reports, state)
    assert reply["mode"] == "local" and reply["text"]
    assert state["messages"][-1]["role"] == "assistant"
    captured = []
    monkeypatch.setattr(ui, "render_agent_tab", lambda **kwargs: captured.append(kwargs))
    render_existing_agent_tab(data, reports, state)
    assert captured == [{"data": data, "reports": reports, "state": state, "legacy_calendar": False}]
