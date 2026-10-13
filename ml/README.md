# 마가렛트 모델·평가 결과

팀 공동 저장소에 추가할 `ml/` 구성요소입니다. 대시보드 담당자는 저장된 CSV를 읽거나 추론 함수를 호출할 수 있습니다. 기존 저장소의 `dashboard_app.py`, `dashboard/`, `analyze/`, 루트 README 및 설정은 이 묶음에 포함하지 않습니다.

## 무엇이 들어 있나요?

| 경로 | 용도 |
|---|---|
| `models/forecast/` | 선택된 평균 예측 모델, 비교용 XGBoost 모델 |
| `models/anomaly_v2/selected_model.joblib` | 13개 설비의 전압·역률 Isolation Forest와 고정 경고 기준 |
| `src/manufacturing_ai/` | 학습·평가 함수와 저장 모델 추론 함수 |
| `scripts/` | 전력 예측 및 새 센서 기록 판정 실행 |
| `outputs/forecast/` | 5월 전력·요금·탄소 예측, 평가 결과, 선택 근거 |
| `outputs/anomaly_v2/` | 이상탐지 비교 결과, 시간별 경고 집계, 실험 절차 |
| `docs/` | 성능 해석, 입력·출력 규격, 대시보드 연결, 재현 범위 |
| `examples/sensor_demo.csv` | 실행 확인용 인공 센서 예시. 실제 측정 기록이 아님 |
| `tests/` | 원본 데이터 없이 실행하는 단위 테스트 |
| `MANIFEST.json` | 공유 대상 파일 목록·크기·SHA256 |

## 먼저 알아둘 결과

- **전력:** XGBoost를 비교했지만 평균 예측을 최종 선택했습니다. 5월 정규화 MAE 약 **0.228%**이며, 분류 정확도와 다릅니다. AI가 단순 평균보다 좋아졌다고 주장하지 않습니다.
- **이상탐지:** 개선 모델의 4월 정밀도 **97.77%**, 재현율 **98.28%**입니다. 이미 확인한 4월 데이터에 대한 재평가이며 새로운 데이터의 독립 검증 성능은 아닙니다.
- 이상 표시는 교육자료의 전압·역률 기준과 일치합니다. 실제 설비 고장 정답을 확보한 것은 아닙니다.
- 전력의 물리적 단위와 제공 정답의 수치 규칙이 달라, 두 종류의 결과를 구분해 제공합니다.

자세한 수치와 한계는 [모델 결과](docs/MODEL_RESULTS.md), 컬럼·단위는 [데이터 규격](docs/DATA_CONTRACT.md)을 확인하세요.

## 실행 방법

검증 환경은 **Python 3.10**, 라이브러리 버전은 `requirements.txt`에 고정했습니다. 아래 명령은 저장소의 `ml` 폴더에서 실행합니다. API 키나 OpenAI API는 필요하지 않습니다.

```powershell
# Windows PowerShell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/predict_power.py --start "2025-05-10" --end "2025-05-31"
.\.venv\Scripts\python.exe scripts/predict_anomaly_v2.py --input examples/sensor_demo.csv
.\.venv\Scripts\python.exe -m pytest -q --basetemp=tmp/pytest
```

```bash
# macOS/Linux (Python 3.10 설치 필요)
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/predict_power.py --start "2025-05-10" --end "2025-05-31"
.venv/bin/python scripts/predict_anomaly_v2.py --input examples/sensor_demo.csv
.venv/bin/python -m pytest -q --basetemp=tmp/pytest
```

운영체제별 새 환경 설치는 팀원 환경에서 확인해야 합니다. 현재 묶음은 Windows/Python 3.10 환경에서 독립 경로 실행을 검사했습니다.

- 전력 출력: `outputs/forecast/custom_prediction.csv` — 설비별·시간별 결과.
- 이상탐지 출력: `outputs/anomaly_v2/new_observations.csv` — 원본 행 순서 유지.
- 이상탐지에는 실제 새 데이터 CSV 경로를 `--input`에 지정하면 됩니다.
- 추론 결과 경로는 `--output`으로 지정하되 `ml/` 안에 저장합니다.
- 전력 출력은 저장 모델 경로에 덮어쓸 수 없도록 제한했습니다.
- 저장 모델은 이 저장소에서 제공한 파일을 사용합니다. 모델 형식과 라이브러리 버전을 함께 유지하세요.

## 팀원별 연결 지점

- **대시보드:** [연결 안내](docs/INTEGRATION.md)의 결과 CSV와 컬럼을 사용합니다. 이 묶음을 추가하는 것만으로 기존 대시보드가 자동 연결되지는 않습니다.
- **모델 담당:** 선택 기준과 후보 성능은 `outputs/anomaly_v2/protocol.json`, `selection.json`, `summary.json`에 있습니다.
- **발표·보고서:** [모델 결과](docs/MODEL_RESULTS.md)의 검증 범위와 단위 조건을 수치와 함께 표시합니다.

이 묶음은 **저장 모델 사용과 결과 공유**를 우선합니다. 대용량 원본·학습 캐시·개인 실행 환경과 로컬 전체 재학습 스크립트는 포함하지 않았습니다. 재학습에 필요한 자료와 절차는 [재현 범위](docs/REPRODUCIBILITY.md)를 확인하세요.

[공유 묶음 검증 결과](VALIDATION.json) · [파일 목록과 해시](MANIFEST.json)
