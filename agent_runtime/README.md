# 에이전트·점검 보고서 연결 모듈

기존 팀 화면에 가져다 쓰는 패키지입니다. `dashboard_app.py`나 `dashboard_data.py`를 가져오지 않으며 별도 웹앱·서버를 실행하지 않습니다. 기존 `agent/` 요구 문서와 팀 양식 원본은 그대로 두고, 이 폴더에 실행 코드와 기본 양식 사본을 둡니다.

## 기존 화면에 연결

`data`는 아래의 `DataService` 또는 같은 메서드를 제공하는 팀 데이터 서비스, `reports`는 `ReportService`, `api_key`는 외부 설정으로 받은 키입니다. 상태는 사용자별로 유지하세요. `run_turn()`은 Streamlit 없이 호출할 수 있습니다.

```python
from agent_runtime import AgentTools, run_turn
state = {"view_date": "2025-05-31", "selected_equipment": "12(4호기)"}
tools = AgentTools(data, reports, state)
reply = run_turn("4월 29일 4호기 전압 어때?", state, tools, api_key=api_key)
print(reply["text"])
```

선택 제공되는 Streamlit 탭은 기존 탭 본문에서만 호출합니다. 페이지 설정·사이드바·탭 이동은 팀 화면이 담당합니다.

```python
from agent_runtime.agent_ui import render_agent_tab
render_agent_tab(data=data, reports=reports, state=st.session_state)
```

일반 `dict`도 전달할 수 있지만 화면 재실행 때 같은 객체를 유지해야 합니다. `state`를 생략하면 `st.session_state`를 사용합니다. 기존 화면에 `selected_year/month/day`가 있다면 `legacy_calendar=True`를 명시해 달력 연동만 켤 수 있습니다. 기본값에서는 이 키들을 읽거나 변경하지 않습니다. 함수만 포함한 연결 예제는 [examples/integrate_agent.py](examples/integrate_agent.py)를 참고하세요.

| 공유 상태 키 | 의미 |
| --- | --- |
| `view_date` | `YYYY-MM-DD` 조회일. 기본값·‘오늘’은 `2025-05-31` |
| `selected_equipment` | 공백 없는 설비 ID, 예: `13(3호기)`, `12(4호기)` |
| `thresholds` | `phase_voltage: 210`, `line_voltage: 360`, `power_factor: 85`; 사용자 규칙만 변경 |
| `messages`, `trace` | 대화와 도구 조회 근거. `run_turn()`이 대화를 추가하므로 중복 추가하지 않음 |
| `report_draft`, `report_context` | 현재 보고서 초안과 날짜·설비 |
| `pending_template`, `template_asked` | 양식 적용 전 확인 및 기본 양식 확인 상태 |

`reply`에는 `text`, `mode`(`openai`/`local`), `trace`가 있습니다. API 실패 정보는 민감한 원문 대신 정리된 코드로 반환합니다.

## 데이터와 API 설정

저장된 ML 결과가 있는 루트를 `MARGARET_AI_ROOT` 환경변수로 지정하는 방식을 권장합니다. 명시적으로 연결할 수도 있습니다.

```python
from pathlib import Path
from agent_runtime.agent_data import DataService
from agent_runtime import ReportService
data = DataService(Path("ml"))
reports = ReportService(Path("agent_runtime"))
```

`ReportService`는 전달한 폴더 아래 `runtime/`에 양식 설정을 보관합니다. 배포 환경에서는 쓰기 가능한 사용자별 폴더를 전달하세요. 해당 폴더에 기본 Excel 양식이 없으면 패키지 사본을 사용합니다.

- 대화 모델 기본값: **`gpt-5-mini`**. 코어에는 `api_key`와 선택 인자 `model`을 전달합니다.
- UI는 환경변수 `OPENAI_API_KEY`, `OPENAI_MODEL`을 우선합니다. 파일 설정은 `MARGARET_AI_ROOT/.streamlit/secrets.toml` → 저장소 루트 → 패키지 폴더 순서로 읽으며 뒤 파일이 앞 파일을 덮어씁니다. `load_settings(project=...)`는 지정 폴더만 읽습니다. `.env.example`은 참고용이며 `.env`를 자동으로 읽지는 않습니다.
- 키가 없거나 API 응답을 완료하지 못하면 **데모 모드**로 표시하고 같은 실제 데이터 도구로 로컬 조회합니다. 임의 데이터는 만들지 않습니다.
- 기본 조회일인 5월 31일에는 센서 실측이 없습니다. 센서는 **4월 1–29일**, 전력 예측은 **5월 10일 00시–31일 00시 포함 505시간**입니다. 자료 없음은 정상·0건과 구분합니다.
- AI 이상 판정은 저장된 모델 기준으로 고정됩니다. 센서 규칙 변경은 AI 점수나 판정을 바꾸지 않습니다. AI 판정은 고장 확정이나 원인 진단이 아닙니다.
- 5월 MAE는 공장 전체에만 해당합니다. 제공된 정답 단위의 물리 kWh 해석은 확인이 필요하며, 전월 대비율과 설비별 MAE를 만들어 내지 않습니다.

학습·추론 자산은 별도 `ml/` 패키지에서 준비합니다. 이 패키지는 모델을 다시 학습하지 않습니다. `MARGARET_AI_ROOT`가 없으면 `저장소/ml` → 저장소 → 저장소의 부모 중 `outputs/agent_data/manifest.json`이 있는 위치를 찾고, 없으면 `저장소/ml`을 사용합니다.

필요한 파일은 다음과 같습니다. 원본 CSV는 Git 밖에 보관해도 됩니다.

- 실측 조회: `outputs/agent_data/manifest.json`과 이 manifest가 가리키는 날짜별 parquet, 원본 CSV, `models/anomaly_v2/selected_model.joblib`. 데이터와 모델의 동일성을 검사합니다.
- 5월 예측: `models/forecast_v2/selected_model.joblib`, `outputs/forecast_v2/{summary.json,model_freeze.json,selected/may_equipment_physical.csv}`.
- 최초 캐시 준비만 추가: ML 패키지의 `src/manufacturing_ai`와 해당 requirements. 저장 모델과 호환되는 `scikit-learn==1.4.2`, `joblib` 등이 필요합니다.

4월 캐시가 없다면 원본 CSV와 위 자산을 준비한 뒤 저장소 루트에서 실행하세요. 전체 원본을 한 번 읽고 4월의 고정 AI 점수를 저장하므로 수분 이상 걸릴 수 있습니다.

```text
python -m agent_runtime.prepare_data --project-root /path/to/ml --source /path/to/rtu_data_full.csv
```

새 PC에서는 `--source`로 원본 위치를 명시하세요. 원본 경로는 manifest에 기록됩니다. 복사 과정에서 수정 시각만 바뀐 경우 동일한 바이트인지 SHA 검증하며, 원본·모델의 내용이 바뀌면 확인 후 `--rebuild`로 다시 준비합니다. 키, 원본 데이터, 생성 캐시, 보고서와 런타임 상태는 Git에 포함하지 않습니다.

## 도구 12개와 보고서

| 용도 | 도구 |
| --- | --- |
| 데이터 조회 | `get_equipment_list`, `get_anomalies`, `get_measurements`, `get_power`, `get_power_ranking`, `get_forecast` |
| 사용자 규칙 | `get_thresholds`, `set_thresholds` |
| 보고서·양식 | `get_report_template`, `save_report_template`, `reset_report_template`, `create_report` |

`get_measurements`는 개별 9개 센서뿐 아니라 `equipment="전체"`, `item="phase_voltage"`/`"line_voltage"`/`"power_factor"`로 전체 설비의 해당 3상을 한 번에 조회할 수 있습니다. 조회일에 자료가 없으면 안내만 하며 다른 날짜로 자동 대체하지 않습니다.

팀 자체 데이터 서비스를 주입한다면 `DataService`의 메서드·반환 형식을 맞춰 주세요. 전체·묶음 조회에는 `get_measurements_group(date, equipment, item, thresholds=None)`도 필요합니다.

보고서는 실제 조회값과 사용자가 입력한 점검자·조치 내용 등을 합쳐 **초안**으로 만듭니다. 필수 입력과 양식 확인 후 UI에서 XLSX를 다운로드하거나 `reports.export_xlsx(state["report_draft"])`로 바이트를 받습니다. 자동 전송하지 않습니다. 기본 Excel 배치와 공란 규칙을 유지하고, 새 `.xlsx`/`.docx` 양식은 항목을 확인한 뒤 적용합니다. 내보내기는 XLSX입니다.

## 개발 확인

저장소 루트에서 실행합니다. 코어만 쓰면 `requirements.txt`, Streamlit 탭까지 쓰면 `requirements-ui.txt`를 설치하세요. 테스트는 가짜 키와 도구 응답을 사용하며 실제 API 호출을 하지 않습니다.

```text
python -m pip install -r agent_runtime/requirements-dev.txt
python -m pytest agent_runtime/tests -q --basetemp=agent_runtime/runtime/pytest_tmp
```
