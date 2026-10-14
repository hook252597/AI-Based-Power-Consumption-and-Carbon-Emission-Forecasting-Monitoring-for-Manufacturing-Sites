# AI 기반 제조 현장 전력량, 탄소 배출량 예측 및 모니터링
본 프로젝트는 과거 제조 현장 전력 데이터를 AI로 학습하여 미래 전력 소비량·전기요금·탄소 배출량을 예측하고,  관리팀이 대시보드 하나로 비용 산정과 추이 파악을 바로 처리할 수 있게 함을 목적으로 둠.

- 기준 시각: **2025-04-29 09:00 고정** (00~08시까지 실측, 이후는 AI 예측)
- 설비: 13개 (RTU 채널 1~5, 11~18)
- 화면: 현황·예측 / 이상탐지 / 에이전트

## 데이터 흐름

```
data/rtu_data_full.csv (원본 5초 RTU 기록, Git 제외)
   │  python tools/prepare_dashboard_cache.py
   │    ├─ ml/models/anomaly_v2   이상 판정 (Isolation Forest)
   │    └─ ml/models/forecast_v2  4/30~5/31 전력 예측 (Extra Trees)
   ▼
ml/outputs/agent_data/ (캐시, Git 제외)
   ├─ dashboard_hourly.parquet / .json     2024-12-01~2025-04-29 설비·시간별 실측 + AI 판정 + 경고 기준
   └─ dashboard_forecast.parquet / .json   2025-04-30~2025-05-31 설비·시간별 예측
   ▼
dashboard_app.py ──(에이전트 탭)──▶ agent/
```

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `dashboard_app.py` | Streamlit 대시보드 진입점 |
| `agent/` | 대화형 에이전트 (`agent_tools.py` 도구, `agent_core.py` 모델 호출 루프, `agent_ui.py` 화면), 명세 `agent.md`, 시스템 프롬프트 `agentSystem.md`, 기본 보고서 양식 `BasicSafetyReportForm.xlsx` |
| `tools/prepare_dashboard_cache.py` | 원본 CSV와 저장된 모델로 대시보드 캐시 생성 |
| `ml/` | 모델 학습·검증 코드와 결과물 (자세한 내용은 `ml/README.md`) |
| `data/` | 원본 데이터. `rtu_ground_truth_may.csv`는 5월 공장 전체 실측 |
| `dashboard/` | 대시보드 명세 프롬프트(`dashBoardPrompt.md`)와 화면 이미지 |
| `analyze/` | EDA 분석 보고서 |
| `.streamlit/` | 테마(`config.toml`), API 키(`secrets.toml`, Git 제외) |

## 실행 방법 (저장소 루트에서)

```powershell
pip install -r requirements.txt

# 1) 캐시 생성 — 최초 1회, 약 4분. data/rtu_data_full.csv 필요
python tools/prepare_dashboard_cache.py
#    python tools/prepare_dashboard_cache.py --forecast-only    # 예측만 다시 생성
#    python tools/prepare_dashboard_cache.py --thresholds-only  # AI 경고 기준만 갱신

# 2) 대시보드 실행
streamlit run dashboard_app.py
```

## 에이전트 API 키

`.streamlit/secrets.toml` 또는 환경변수에 넣습니다. 키가 없으면 데모 모드로 동작합니다.

```toml
OPENAI_API_KEY = "..."
OPENAI_MODEL = "gpt-5-mini"   # 생략 시 기본값
```

`secrets.toml`은 커밋하지 마세요.

## Git에 올리지 않는 파일

- `data/rtu_data_full.csv`, `data/rtu_data_full_flagged.csv` (GitHub 100MB 제한 초과)
- `ml/outputs/agent_data/` (캐시 생성 스크립트로 다시 만듦)
- `.streamlit/secrets.toml`
- `agent/report_template.json`, `agent/report_template_default.json`, `agent/agent_state.json`, `agent/templates/` (에이전트가 실행 중 생성)
