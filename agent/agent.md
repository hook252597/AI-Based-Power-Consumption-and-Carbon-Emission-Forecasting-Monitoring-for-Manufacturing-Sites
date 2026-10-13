# 역할
너는 Python으로 LLM 에이전트를 구현하는 개발자다. 아래 명세대로 에이전트를 구현하라.
이 에이전트는 이미 만들어진 Streamlit 대시보드의 "에이전트" 탭 안에서 동작한다.
사용자는 중소 제조업체 관리팀 담당자(설비·비용 관리, 데이터 분석 비전공)다.

# 전제
- 대시보드는 Streamlit + Plotly로 이미 구현되어 있다. 에이전트는 그 안의 한 탭이고, 별도 웹앱을 새로 만들지 않는다.
- 모델은 상용 API(OpenAI / Anthropic / Google 중 하나)를 쓰고 function calling(tool use)을 사용한다.
- 모델 호출 부분은 `call_model(messages, tools)` 함수 하나로 분리해, 다른 제공사로 바꿀 때 그 함수만 고치면 되게 한다.
- **도구를 위한 별도 서버(FastAPI 등)를 만들지 않는다.** 도구는 같은 파이썬 프로세스 안의 일반 함수로 두고, 모델이 도구 호출을 반환하면 그 함수를 직접 호출해 결과를 돌려준다.
- API 키는 코드에 적지 않는다. `st.secrets` 또는 환경변수에서 읽고, 없으면 데모 모드로 떨어진다(아래 8번).
- 기본 보고서 양식 파일 `BasicSafetyReportForm.xlsx` 가 프로젝트에 함께 들어 있다. 이 파일이 처음부터 등록된 기본 양식이다. 양식이 하나도 없는 상태는 존재하지 않는다.

# 데이터
- 학습 데이터: training-datasets/rtu_data_full.csv (2024-12 ~ 2025-04)
- 검증 데이터: may-datasets/rtu_ground_truth_may.csv (2025-05)
- 설비 13개 (앞의 숫자는 RTU 채널 번호 — 임의로 다시 매기지 않는다)
    1 (PM-3) / 2 (L-1전등) / 3 (분쇄기(2)) / 4 (분쇄기(1)) / 5 (좌측분전반) /
    11 (우측분전반1) / 12 (4호기) / 13 (3호기) / 14 (2호기) / 15 (예비건조기) /
    16 (호이스트) / 17 (6호기) / 18 (우측분전반2)
- 전력량: activePower(kW) × (5초 ÷ 3600초)를 5초 단위로 누적 합산 → kWh
- 전기요금: 전력량(kWh) × 180원
- 탄소 배출량: 전력량(kWh) × 0.424 kgCO2
- 이상 판정 기본 임계치: 상전압 R/S/T 210V 미만, 선간전압 RS/ST/TR 360V 미만, 역률 R/S/T 85% 미만
- **"오늘"은 2025-05-31로 고정한다.** 대시보드와 같은 기준이어야 한다. 실제 오늘 날짜를 쓰면 데이터가 없어 에이전트가 빈 답을 한다.

---

# 1. 도구 함수 12개

아래 시그니처와 반환 구조를 그대로 구현한다. 모든 반환값은 JSON 직렬화 가능해야 하고,
데이터가 없으면 예외를 던지지 말고 빈 결과와 `"note"` 설명을 담아 돌려준다.

```python
get_equipment_list() -> list[dict]
# [{"channel": "12", "name": "12 (4호기)", "status": "abnormal",
#   "abnormal_items": ["상전압"]}, ...]   # 13개 전부

get_anomalies(date: str, equipment: str | None = None) -> dict
# {"date": "2025-05-31", "count": 3,
#  "items": [{"equipment": "12 (4호기)", "item": "상전압 R",
#             "timestamp": "2025-05-31 14:00", "value": 204.1,
#             "threshold": 210, "deviation_pct": -2.8, "duration_min": 180}],
#  "note": ""}

get_measurements(date: str, equipment: str, item: str) -> dict
# item 은 다음 중 하나:
#   "phase_voltage_R" | "phase_voltage_S" | "phase_voltage_T"
#   "line_voltage_RS" | "line_voltage_ST" | "line_voltage_TR"
#   "power_factor_R"  | "power_factor_S"  | "power_factor_T"
# {"equipment": "...", "item": "...", "unit": "V",
#  "threshold": 210, "series": [{"hour": 0, "value": 221.3}, ...],
#  "min": 204.1, "min_hour": 14, "under_threshold_hours": [13,14,15]}

get_power(date: str, equipment: str | None = None,
          granularity: str = "hour") -> dict      # "hour" | "day"
# {"date": "...", "equipment": "전체" 또는 설비명,
#  "series": [{"t": "14", "kwh": 212.4}, ...],
#  "total_kwh": 1636.0, "total_cost_krw": 294480, "total_co2_kg": 693.7,
#  "cost_rate_krw_per_kwh": 180, "co2_factor_kg_per_kwh": 0.424}

get_power_ranking(date: str) -> dict
# {"date": "...", "total_kwh": 1636.0,
#  "items": [{"rank": 1, "equipment": "3 (분쇄기(2))", "kwh": 290, "share_pct": 17.7}, ...]}
#  # 소비량 내림차순, 13개 전부

get_forecast(month: str, equipment: str | None = None) -> dict   # month: "2025-05"
# {"month": "2025-05",
#  "expected_kwh": 64830, "expected_cost_krw": 11669400, "expected_co2_kg": 27488,
#  "prev_month_diff_pct": 8.2,
#  "mae_kwh_per_hour": 4.3, "model": "LSTM"}

get_thresholds() -> dict
# {"phase_voltage": 210, "line_voltage": 360, "power_factor": 85,
#  "is_default": True, "changed_log": []}

set_thresholds(phase_voltage: float | None = None,
               line_voltage: float | None = None,
               power_factor: float | None = None,
               reason: str = "") -> dict
# 바꾼 값으로 다시 분석하고 결과 요약을 돌려준다
# {"before": {...}, "after": {...},
#  "anomaly_count_before": 3, "anomaly_count_after": 6,
#  "newly_abnormal": ["13 (3호기)", "14 (2호기)", "17 (6호기)"],
#  "no_longer_abnormal": [],
#  "log_entry": {"at": "2025-05-31 14:20", "by": "사용자",
#                "change": "역률 85 → 90", "reason": "..."}}

get_report_template() -> dict
# 지금 쓰고 있는 양식. 기본 양식이 늘 등록되어 있으므로 None 을 돌려주지 않는다.
# {"name": "제조현장 안전 점검 보고서", "is_default": True,
#  "registered_at": "...", "source_file": "BasicSafetyReportForm.xlsx",
#  "layout": "table",
#  "fields": [{"name": "발생일시", "source": "auto", "maps_to": "anomaly.timestamp"},
#             {"name": "조치내용", "source": "user"},
#             {"name": "결재란",   "source": "blank"}]}

save_report_template(fields: list[dict], layout: str = "table",
                     source_file: str | None = None) -> dict
# 새 양식으로 교체한다. is_default 는 False 가 된다.

reset_report_template() -> dict
# 기본 양식으로 되돌린다. 기본 양식은 지우지 말고 늘 따로 보관해 둔다.

create_report(date: str, equipment: str,
              action_taken: str = "", extra_fields: dict | None = None) -> dict
# 등록된 양식(없으면 기본 양식)에 맞춰 보고서 초안을 만든다
# {"template_name": "...", "rows": [{"name": "...", "value": "...", "source": "auto"}],
#  "missing_user_fields": ["조치자", "종합 판정"], "ready_to_export": False}
```

### 도구 구현 규칙
- `source="auto"` 항목은 위 도구들의 반환값에서 채운다. 데이터가 없으면 빈 문자열로 두고 `missing_user_fields`에 넣는다. 지어내지 않는다.
- `set_thresholds`는 값을 바꾸기만 하는 게 아니라 **반드시 재분석 결과까지 담아 돌려준다.** 건수 변화와 새로 포함/제외된 설비를 모델이 알아야 사용자에게 설명할 수 있다.
- 모든 도구는 실패해도 예외를 밖으로 던지지 않는다. `{"error": "사유"}` 형태로 돌려주고, 모델이 그대로 사용자에게 전달하게 한다.

---

# 2. function calling 루프

```
사용자 메시지 →
  messages 에 추가 →
  call_model(messages, tools) →
    모델이 도구 호출을 반환하면:
      해당 함수 실행 → 결과를 messages 에 tool 결과로 추가 → 다시 call_model
    모델이 텍스트를 반환하면:
      화면에 출력하고 종료
```

- 한 번의 사용자 메시지에 도구를 여러 번 부를 수 있게 한다. 예: "어디야?"에 `get_anomalies` → `get_measurements` 두 번.
- **최대 반복 횟수를 5회로 제한한다.** 넘으면 루프를 끊고 "확인에 실패했습니다"로 답한다. 무한 루프는 요금이 계속 나간다.
- 도구 호출과 결과를 화면 어딘가에 접어서 볼 수 있게 남긴다(디버깅용). 발표 때는 접어둔다.
- 스트리밍 응답을 쓰면 체감 속도가 크게 좋아진다. 가능하면 적용한다.

---

# 3. 시스템 프롬프트 주입

- 에이전트의 성격·말투·규칙은 별도 파일 `agentSystem.md`에 들어 있다. 코드에 하드코딩하지 말고 그 파일을 읽어 system 메시지로 넣는다.
- 그 아래에 **매 턴 현재 상태를 덧붙인다.** 다음 값들이다:

```
[현재 상태]
오늘 날짜: 2025-05-31
조회 중인 날짜: 2025-05-31
선택된 설비: 12 (4호기)
이상현황 보기 기준: 일일
현재 임계치: 상전압 210V / 선간전압 360V / 역률 85%
보고서 양식: 등록됨 (제조현장 안전 점검 보고서)
```

- 이게 없으면 "어디야?", "얘는 어때?" 같은 말을 알아듣지 못한다. 화면 상태와 에이전트가 아는 상태는 항상 같아야 한다.
- 상태 블록은 매 턴 새로 만들어 넣고, 지난 턴의 상태 블록은 messages에서 지운다. 쌓이면 모델이 옛날 상태를 보고 답한다.

---

# 4. 대화 상태 관리

- 모든 상태는 `st.session_state`에 보관한다: `messages`, `selected_equipment`, `view_date`, `view_range`, `thresholds`, `report_draft`, `checklist`.
- 최근 **10턴**만 유지하고 그 앞은 버린다. 단 보고서 작성 중에는 조치 내용이 사라지면 안 되니, 보고서 초안은 messages가 아니라 `report_draft`에 따로 둔다.
- 선택된 설비는 에이전트 전용 값을 따로 두지 않는다. 대시보드 전체가 공유하는 하나의 값을 읽고 쓴다.

---

# 5. 보고서 양식과 체크리스트

## 5-1. 기본 양식 등록
- 프로젝트에 함께 들어 있는 `BasicSafetyReportForm.xlsx` 를 앱 시작 시 파싱해 `report_template.json` 으로 변환하고, 이것을 기본 양식으로 등록한다.
- **양식 파일의 칸 배경색이 항목 분류를 나타낸다.** 파싱할 때 이 규칙을 쓴다.
    · 연한 청록 배경 → `source="auto"` (데이터에서 자동으로 채움)
    · 연한 노랑 배경 → `source="user"` (사람이 직접 적음)
    · 배경색 없음   → `source="blank"` (문서번호·결재란 등, 비워 둠)
  색을 읽지 못하면 항목 이름으로 판단하고, 애매하면 `"user"` 로 둔다.
- 파싱 결과를 `report_template_default.json` 에 따로 복사해 둔다. 사용자가 다른 양식으로 바꿔도 이 파일은 지우지 않는다. `reset_report_template()` 이 여기서 복구한다.

## 5-2. 최초 1회 질문
- 보고서를 처음 만들려 할 때 **딱 한 번만** 사용자에게 다른 양식이 있는지 묻는다.
- 이 "물어봤는지" 여부는 `agent_state.json` 의 `template_asked` 로 관리한다. **세션 상태가 아니라 파일에 저장한다.** 앱을 다시 켜도 다시 묻지 않아야 한다.
- 사용자가 없다고 하면 기본 양식을 그대로 쓰고 `template_asked = true` 로 기록한다.
- 사용자가 파일을 올리면 5-4의 절차로 교체하고 `template_asked = true` 로 기록한다.
- 한 번 기록된 뒤에는 에이전트가 양식 이야기를 먼저 꺼내지 않는다.

## 5-3. 예외 — 사용자가 먼저 바꾸자고 할 때
- 사용자가 "다른 양식 쓸게", "양식 바꿀게", "이 양식으로 해줘" 같이 **먼저 말하면** `template_asked` 와 상관없이 **몇 번이든** 새 양식을 받는다.
- 이때는 되묻지 말고 바로 업로드를 요청한다. 업로드 위젯은 대화창 안에 띄운다.
- "원래 양식으로 돌려줘", "기본으로 돌아가자" 는 `reset_report_template()` 으로 처리한다.

## 5-4. 새 양식을 받았을 때
- `.xlsx` 는 openpyxl, `.docx` 는 python-docx 로 파싱한다.
- `.hwp` 는 파싱하지 않는다. "항목을 말로 불러주세요"로 받아 `save_report_template()` 에 넘기는 경로를 반드시 남긴다.
- 파싱한 항목을 auto / user / blank 로 분류해 사용자에게 보여주고 확인받은 뒤에 저장한다. 확인 없이 저장하지 않는다.
- 저장 후 체크리스트와 오른쪽 보고서 미리보기를 새 양식 기준으로 다시 그린다.

## 5-5. 체크리스트
- **하드코딩하지 않는다.** 양식의 `source == "user"` 항목만 뽑아 만든다. 양식이 바뀌면 체크리스트도 자동으로 바뀐다.
- 양식 전문은 매 턴 프롬프트에 넣지 않는다. 토큰 낭비다. 보고서를 만드는 시점에 `get_report_template()` 도구로 가져오게 한다.

---

# 6. 임계치 변경

- 사용자가 대화로 임계치를 바꾸면 `set_thresholds`를 호출한다.
- 변경 결과는 대화 안에 짧은 카드로 보여준다: 바뀐 값, 이상 건수 변화, 새로 포함된 설비.
- **변경 이력을 남긴다.** 시각·바뀐 값·사유 한 줄. 화면에서 볼 수 있게 한다.
- **기본값으로 되돌리는 버튼을 둔다.**
- 바뀐 임계치는 이상현황 탭의 그래프 가이드 선에 반영된다. 에이전트 탭에는 그래프를 그리지 않는다.
- 임계치는 안전 기준값이다. 기준을 낮추면 이상이 사라진 게 아니라 안 보이게 되는 것이므로, 기록과 되돌리기는 끄지 않는다.

---

# 7. 화면 (에이전트 탭)

4번 영역을 좌우로만 나눈다. 좌 45% / 우 55%.

**왼쪽**
- 맨 위: 체크리스트 진행 띠 한 줄 ("일일 점검 체크리스트  7 / 12 완료"). 누르면 펼쳐지고 평소엔 접혀 있다.
- 가운데: 대화 영역. 사용자 메시지는 오른쪽, 에이전트는 왼쪽.
- 맨 아래: 입력창 고정.
- 화면을 처음 열면 에이전트가 오늘 이상 요약을 먼저 말한다.

**오른쪽**
- 맨 위: 양식 이름 + 내보내기 버튼. 필수 항목이 다 차기 전에는 비활성.
- 본문: 양식 항목을 순서대로. 빈 항목은 "(미입력)" 흐리게, 방금 채워진 항목은 잠깐 강조.
- 자동으로 채워진 항목과 사람이 적은 항목을 색으로 구분한다.

대화로 값이 채워지면 왼쪽 진행률과 오른쪽 보고서가 **같은 순간에** 갱신되어야 한다. 이게 이 화면의 핵심이다.

---

# 8. 오류 처리와 데모 모드

- API 키가 없거나 호출이 실패하면 앱이 죽지 않고 **데모 모드**로 떨어진다. 미리 준비한 대화 흐름을 재생해 발표가 멈추지 않게 한다. 화면 어딘가에 "데모 모드" 표시를 작게 남긴다.
- API 호출에 타임아웃(20초)을 건다. 넘으면 "응답이 지연되고 있습니다. 다시 시도할까요?"로 답한다.
- 도구가 `{"error": ...}`를 돌려주면 모델이 그대로 사용자에게 전달하게 하고, 숫자를 추측해 메우지 않게 한다.
- 발표 전에 아래 질문으로 한 번씩 확인한다. 특히 마지막 두 개가 중요하다.
    1. "오늘 문제 있는 곳 있어?"
    2. "어디야?"
    3. "4호기 전압 어때?"
    4. "이번 달 전기요금 얼마나 나와?"
    5. "콘덴서 교체했어"
    6. "역률 기준 90%로 올려줘"
    7. "되돌려줘"
    8. "6월 3일은 어때?"            → 데이터가 없으므로 "없습니다"라고 답해야 한다
    9. "작년 대비 어때?"            → 모르면 모른다고 해야 한다
    10. "지금 제일 전기 많이 쓰는 설비 3개"

---

# 지킬 것
- 수치는 반드시 도구 반환값에서만 가져온다. 모델이 기억이나 추정으로 숫자를 만들지 않게 한다.
- 도구를 늘리지 않는다. 위 12개로 충분하고, 많아질수록 모델이 엉뚱한 도구를 고른다.
- 토큰을 아낀다. 시계열 전체를 프롬프트에 넣지 말고 요약(최소/최대/임계 미달 시각)만 넘긴다.
- 한 파일에 다 넣지 말고 나눈다: `agent_tools.py`(도구 함수), `agent_core.py`(모델 호출과 루프), `agent_ui.py`(화면).
- 모델 제공사별 분기는 `call_model` 안에만 둔다.

# 출력
실행 가능한 전체 코드를 파일별로 나누어 작성하라.
코드 맨 위 주석에 필요한 패키지와 환경변수 이름, 실행 명령을 적는다.
설명은 코드 뒤에 짧게 덧붙인다.
