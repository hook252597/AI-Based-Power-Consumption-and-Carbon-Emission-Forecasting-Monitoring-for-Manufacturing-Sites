# 설비의 평소 패턴을 이용한 이상 탐지 비교 실험

2026-10-16에 완료한 실험의 코드·모델·평가 결과입니다. **기존 대시보드의 운영 모델에는 연결하지 않았습니다.** 최종 검증이나 고장 탐지 성능 입증 자료로 해석하지 마세요.

## 먼저 볼 파일

- [작업 인계와 다음 순서](../../docs/HANDOFF_2026-10-16.md)
- [비교 보고서](outputs/context_anomaly_report.html): 파일을 내려받아 브라우저에서 열면 그래프까지 인터넷 없이 확인할 수 있습니다. GitHub 파일 화면에서는 HTML이 실행되지 않습니다.
- [실험 결론](outputs/decision.json)
- [본 실험 실측 경보 수](outputs/real_alert_metrics.csv), [합성 변화 시험](outputs/synthetic_results.csv)
- [별도 센서 관계 실험](outputs/relation_experiment/summary.json)

## 확인된 결과와 한계

12~2월 학습 → 3월 1~15일 경보량 기준 조정 → 3월 16~31일 확인 → 4월 개발 재평가 순서입니다. 기존 이상치 라벨과 5월 정답은 사용하지 않았습니다. 4월은 이미 확인한 개발 자료라 최종 독립 검증이 아닙니다.

합성 변화 51개 중 시간 정보를 추가한 Isolation Forest 27개, 새로 학습한 시점별 정보 IF 42개, 단순 통계 42개를 탐지했습니다. 후속 센서 관계 실험은 41개를 탐지했고, 원래 경보가 없던 대조군에서는 40/50개를 추가 탐지했습니다. 기존 운영 IF와 이번 비교용 IF는 입력·집계 단위가 다릅니다.

실제 관계 경보 예시 52개의 최대 잔차는 9.08W, 관측 전력 대비 최대 0.321%였습니다. 전체 경보를 대표하는 표본이 아니며 실제 점검 가치도 확인되지 않았습니다. 합성 탐지를 실제 고장 정밀도·재현율로 부르지 않습니다.

3분 연속 조건과 1구간/설비·관측일 목표는 비교용 실험 설정입니다. 실제 안전 기준이 아닙니다. 고정 규칙 안의 변화를 주입했으므로 규칙 탐지 0은 설계에 따른 결과입니다. ±10% 및 지속 시간별 알림 정책은 **아직 미구현**입니다.

## 포함 / 제외

- `src/`, `tests/`: 집계, 과거 정보만 사용하는 특징, 모델 학습·평가·보고서와 검증 코드
- `models/*.joblib`, `models/relation/*.joblib`: 이번 실험 모델 26개(약 34MB); 운영용 자동 선택이나 배포 금지
- `outputs/`: 고정 실험 설정, 학습·평가 내역, 비교표, 검토 예시, HTML, 화면 검증 자료
- `outputs/source_data_manifest.json`: 원본 출처·해시·단위 가정·집계 품질 기록(집계 캐시 자체는 아님)
- 원본 CSV, 1분 집계 캐시, 전체 점수·주입 시계열 Parquet는 대용량 재생성 자료라 제외했습니다. 로컬 원본은 그대로 보존되어 있습니다.

실험 당시 고정한 `protocol.json`과 코드의 `deployment_gate`에는 당시의 Git 업로드 금지 조건이 남아 있습니다. 재현 해시를 보존하기 위해 수정하지 않았습니다. 이후 사용자가 **이 별도 브랜치에 작업 보관을 승인**했으며, 운영 적용·main 병합 승인은 아닙니다. 당시 실행 기록의 업로드/배포 여부도 실행 시점의 사실입니다.

## 재현

Python 3.10 환경에서 저장소 루트 기준입니다. 실험용 환경을 따로 만드는 것을 권장합니다.

```powershell
python -m pip install -r experiments/anomaly_context_v3/requirements.txt
python -m pytest experiments/anomaly_context_v3/tests -q
python experiments/anomaly_context_v3/src/run_relation_probe.py --self-test
python experiments/anomaly_context_v3/src/build_report.py
```

위 테스트와 저장된 평가 결과로 HTML 다시 만들기는 원본 CSV 없이 가능합니다. 전체 재학습에는 제공된 12~4월 `rtu_data_full.csv`가 필요합니다. 원본 위치를 명시하세요.

```powershell
python experiments/anomaly_context_v3/src/build_minute_data.py --source "D:/datasets/rtu_data_full.csv"
python experiments/anomaly_context_v3/src/run_experiment.py
python experiments/anomaly_context_v3/src/run_relation_probe.py
python experiments/anomaly_context_v3/src/audit_sensor_relation.py
python experiments/anomaly_context_v3/src/build_report.py
```

원본 파일은 약 4.9GB이며 집계 때 여러 분과 추가 디스크 공간이 필요합니다. `data/manifest.json`이 `complete: true`인 경우에만 학습이 진행됩니다. 재학습은 이 실험 폴더의 모델·평가 결과를 갱신하므로 저장 결과와 비교하려면 먼저 복사본을 보관하세요. 전체 실험 과정에서 OpenAI API를 호출하지 않습니다.

전력은 누적 에너지와의 일치를 근거로 W로 해석한 작업 가정이며 제공자 단위 확인이 남아 있습니다. 원본 Unix timestamp와 localtime의 불일치도 기록되어 있어 localtime 기준을 사용했습니다. 누락·부분 분은 정상으로 채우지 않습니다.
