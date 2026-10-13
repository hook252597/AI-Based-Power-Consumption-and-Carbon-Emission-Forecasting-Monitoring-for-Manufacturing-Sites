# 입력·출력 규격

## 전력 예측

입력은 시작·종료 시각입니다. 시간 간격은 1시간이며 끝 시각을 포함합니다. `2025-05-10 00:00`부터 `2025-05-31 00:00`은 505개 시각입니다. 현재 저장 모델은 2025-04-29 23:00까지 학습했으므로 그 이후 시각만 예측합니다.

CLI 출력은 설비당·시간당 한 행입니다.

| 컬럼 | 의미 |
|---|---|
| `datetime` | 한국 현지 시각, 시간 단위 |
| `module(equipment)` | 학습에 포함된 설비 식별 문자열 |
| `energy_kWh` | 해당 1시간의 예측 전력량, kWh |
| `cost_KRW_estimate` | `energy_kWh × 180`, 원 |
| `carbon_kgCO2_estimate` | `energy_kWh × 0.424`, kgCO₂ |

공장 전체는 같은 시각의 13개 설비 결과를 합산합니다. 위 요금은 과제의 고정 단가 추정이며 실제 청구 요금표를 구현한 것은 아닙니다.

원본 `activePower`는 강의자료와 누적 계량값의 관계상 **W**로 해석했습니다. 5초 동안의 평균값이므로 물리적 전력량은 `Σ(activePower) ÷ 720,000` kWh입니다. 제공된 5월 정답은 이 값의 1,000배 수치인 `hourly_pow`에 180과 0.424를 곱합니다.

- `outputs/forecast/selected/may_physical.csv`: `datetime`, `energy_kWh`, `cost_KRW_estimate`, `carbon_kgCO2_estimate`. 공장 전체 물리 단위 환산.
- `outputs/forecast/selected/may_equipment_physical.csv`: `datetime`, `module(equipment)`, `prediction_kWh`. 설비별 물리 단위 결과. CLI의 `energy_kWh`와 같은 단위입니다.
- `outputs/forecast/selected/may_reference_convention.csv`: 과제 정답의 원본 수치 규칙. `hourly_pow`를 kWh로 그대로 표시하거나 그 요금·탄소 수치를 물리 단위로 표시하지 않습니다.

공식 단위 설명의 충돌은 아직 확인이 필요합니다. 1,000배 관계는 단위 변환이며 5월 오차를 줄이기 위해 학습한 보정계수가 아닙니다.

## 이상탐지

5초 기록 한 행마다 아래 컬럼이 필요합니다. `examples/sensor_demo.csv`는 형식 확인용 인공 예시입니다.

| 컬럼 | 단위·형식 |
|---|---|
| `module(equipment)` | 저장 모델에 등록된 설비 이름과 정확히 일치 |
| `localtime` | `YYYYMMDDHHMMSS` 형식의 한국 현지 시각 |
| `activePower` | W |
| `voltageR`, `voltageS`, `voltageT` | V |
| `currentR`, `currentS`, `currentT` | A |
| `powerFactorR`, `powerFactorS`, `powerFactorT` | 0~100 척도의 % 값. 0~1 비율 아님 |

현재 API의 입력 규격은 전력·전류까지 유지하지만 선택 모델이 점수 계산에 사용하는 항목은 전압 3개와 역률 3개입니다. 미등록 설비 및 센서의 빈 값·무한대는 오류로 처리합니다. `is_outlier`, `outlier_reason`은 추론에 필요하지 않습니다.

| 출력 컬럼 | 의미 |
|---|---|
| `module(equipment)`, `localtime` | 입력 기록 식별값 |
| `anomaly_score` | 클수록 평소 패턴에서 벗어난 모델 점수. 확률 아님 |
| `threshold` | 설비별로 고정한 경고 기준 |
| `is_anomaly` | `anomaly_score >= threshold`인 모델 판정 |
| `documented_rule_anomaly` | 전압 어느 상이든 `<210` 또는 역률 어느 상이든 `<85`인 별도 규칙 판정 |
| `candidate` | 고정한 실험 후보 이름 |

두 종류의 판정은 서로 합치지 않습니다. 센서 한 개만 변하는 사례를 모델이 놓치는 한계가 있어 규칙 판정도 함께 확인합니다.

`outputs/anomaly_v2/april_hourly_anomalies.csv`는 이미 집계한 결과입니다. `selected_flags`는 시간 안에 탐지된 **5초 기록 수**이며 고장 사건 수가 아닙니다. 5월 전력 정답 파일은 이런 센서 컬럼이 없으므로 이상탐지 입력으로 사용할 수 없습니다.
