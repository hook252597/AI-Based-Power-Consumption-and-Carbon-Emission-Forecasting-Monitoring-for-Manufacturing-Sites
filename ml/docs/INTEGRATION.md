# 대시보드 연결 안내

이 구성요소는 기존 팀 대시보드와 별개로 준비했습니다. 첫 연결은 저장된 결과 CSV를 읽는 방식이 가장 간단합니다.

## 준비된 결과 읽기

```python
from pathlib import Path
import pandas as pd

# 팀 저장소 루트의 dashboard_app.py에서 사용하는 경우
ML_ROOT = Path(__file__).resolve().parent / "ml"
forecast = pd.read_csv(
    ML_ROOT / "outputs/forecast/selected/may_physical.csv",
    parse_dates=["datetime"],
)
anomalies = pd.read_csv(
    ML_ROOT / "outputs/anomaly_v2/april_hourly_anomalies.csv",
    parse_dates=["localtime"],
)
```

전력 화면에는 물리 단위 파일을 쓰고, 과제 정답 수치 규칙이 필요하면 명확한 별도 표기로 제공합니다. 이상탐지의 4월 재평가 성능과 5월 전력 예측 결과는 기간과 평가 방법이 다릅니다.

## 새 센서 기록 판정

```python
from pathlib import Path
import sys
import pandas as pd

ML_ROOT = Path(__file__).resolve().parent / "ml"
sys.path.insert(0, str(ML_ROOT / "src"))
from manufacturing_ai.anomaly_v2 import load_selected, predict_observations

# 앱 시작 또는 모델 캐시 초기화 시 한 번 읽습니다.
model = load_selected(ML_ROOT / "models/anomaly_v2/selected_model.joblib")
sensor_rows = pd.read_csv(ML_ROOT / "examples/sensor_demo.csv")
predicted = predict_observations(sensor_rows, model)
```

실제 연결 시 인공 예시 대신 데이터 규격을 만족하는 새 센서 행을 전달합니다. 팀 대시보드의 기존 캐시 방식에 맞춰 모델을 재사용하고 화면 갱신마다 파일을 다시 읽지 않습니다.

## 화면에 표시할 문구와 주의점

- 전력: `예측 전력량(kWh)`, `고정 단가 기준 예상 요금(원)`, `전력량 기반 예상 탄소(kgCO₂)`.
- 이상탐지: `모델 이상 판정`과 `전압·역률 기준 위반`을 구분합니다.
- 성능: `4월 기존 표시 대비 재평가: 정밀도 97.77%, 재현율 98.28%`.
- 전력 모델은 평균 예측을 선택했습니다. 이를 XGBoost 예측 결과로 표시하지 않습니다.
- 현재 예시는 저장 모델을 호출하는 연결 지침입니다. 기존 `dashboard_app.py` 수정·연결·배포는 수행하지 않았습니다.
