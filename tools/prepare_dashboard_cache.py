"""대시보드용 12~4월 설비·시간별 집계 캐시 생성 (최초 1회, 수 분 소요).

원본 5초 RTU 기록(data/rtu_data_full.csv, 읽기 전용)을 한 번 읽어
설비×시간 단위로 요약하고, 저장된 Isolation Forest(ml/models/anomaly_v2)의
고정 판정을 함께 기록한다. 모델을 다시 학습하지 않는다.

실행 (저장소 루트):
    python tools/prepare_dashboard_cache.py
    python tools/prepare_dashboard_cache.py --source D:/data/rtu_data_full.csv

    python tools/prepare_dashboard_cache.py --forecast-only   # AI 예측만 다시 생성
    python tools/prepare_dashboard_cache.py --thresholds-only # AI 경고 기준만 기록(기존 캐시 갱신용)

출력 (Git 제외 폴더 ml/outputs/agent_data/):
    dashboard_hourly.parquet  — 설비·시간별 집계
    dashboard_hourly.json     — 원본·모델 식별 정보, 기간, 행 수, 설비별 AI 경고 기준(ai_thresholds)
    dashboard_forecast.parquet — 4/30~5/31 설비·시간별 AI 전력 예측(Extra Trees, kWh)
    dashboard_forecast.json    — 예측 모델 식별 정보와 저장 예측 대조 결과

집계 컬럼:
    module(equipment), datetime(시간 시작), n(5초 기록 수, 완전한 시간=720),
    kwh(Σ activePower[W] × 5초 / 3,600,000), 전압·역률 9개 항목의 시간 내 최저값,
    ai_flags(Isolation Forest 이상 5초 기록 수), ai_score_max(시간 내 최대 이상점수)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ML = REPO / "ml"
MODEL = ML / "models/anomaly_v2/selected_model.joblib"
OUT_DIR = ML / "outputs/agent_data"
OUT = OUT_DIR / "dashboard_hourly.parquet"
META = OUT_DIR / "dashboard_hourly.json"

MODULE = "module(equipment)"
SENSORS = ["voltageR", "voltageS", "voltageT", "voltageRS", "voltageST", "voltageTR",
           "powerFactorR", "powerFactorS", "powerFactorT"]
MODEL_INPUTS = ["activePower", "voltageR", "voltageS", "voltageT", "currentR", "currentS", "currentT",
                "powerFactorR", "powerFactorS", "powerFactorT"]
# 4/30은 00:00:00 한 시점만 있어 불완전한 시간이므로 제외(12/1 00:00 ~ 4/29 23:59:55)
END_EXCLUSIVE = 20250430000000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    agg = {"n": ("activePower", "size"), "power_sum": ("activePower", "sum"),
           "ai_flags": ("ai_flag", "sum"), "ai_score_max": ("ai_score", "max")}
    agg.update({c: (c, "min") for c in SENSORS})
    return frame.groupby([MODULE, "hour_key"], sort=False).agg(**agg).reset_index()


def _ai_thresholds(bundle: dict) -> dict:
    """설비별 AI 이상점수 경고 기준(모델에 고정된 값). 키: module(equipment) 표기."""
    return {str(module): float(item["threshold"]) for module, item in bundle["equipment"].items()}


def write_ai_thresholds() -> None:
    """원본 CSV를 다시 읽지 않고, 모델 파일의 AI 경고 기준만 dashboard_hourly.json에 기록한다."""
    sys.path.insert(0, str(ML / "src"))
    from manufacturing_ai.anomaly_v2 import load_selected

    if not META.is_file():
        raise SystemExit("dashboard_hourly.json이 없습니다. 먼저 전체 캐시를 만드세요: python tools/prepare_dashboard_cache.py")
    meta = json.loads(META.read_text(encoding="utf-8"))
    digest = _sha256(MODEL)
    if meta.get("model_sha256") and meta["model_sha256"] != digest:
        raise SystemExit("캐시를 만든 이상탐지 모델과 현재 모델 파일이 다릅니다. 전체 캐시를 다시 만드세요.")
    meta["ai_thresholds"] = _ai_thresholds(load_selected(MODEL))
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"AI THRESHOLDS READY: {len(meta['ai_thresholds'])} equipment → {META.name}", flush=True)


def build(source: Path, chunksize: int) -> None:
    sys.path.insert(0, str(ML / "src"))
    from manufacturing_ai.anomaly_v2 import load_selected, predict_observations

    bundle = load_selected(MODEL)
    columns = list(dict.fromkeys([MODULE, "localtime", *MODEL_INPUTS, *SENSORS]))
    dtypes = {c: "float32" for c in columns if c not in (MODULE, "localtime", "activePower")}
    dtypes.update({MODULE: "string", "localtime": "int64", "activePower": "float64"})
    parts, seen, start = [], 0, time.monotonic()
    for index, chunk in enumerate(pd.read_csv(source, usecols=columns, dtype=dtypes, chunksize=chunksize)):
        seen += len(chunk)
        chunk = chunk.loc[chunk["localtime"] < END_EXCLUSIVE].reset_index(drop=True)
        if chunk.empty:
            continue
        inference = predict_observations(chunk, bundle)
        chunk["ai_flag"] = inference["is_anomaly"].to_numpy(dtype=bool).astype("int32")
        chunk["ai_score"] = inference["anomaly_score"].to_numpy(dtype=float)
        chunk["hour_key"] = chunk["localtime"] // 10_000  # YYYYMMDDHH
        parts.append(_aggregate(chunk))
        if index % 5 == 0:
            print(f"scanned {seen:,} rows ({time.monotonic() - start:.0f}s)", flush=True)

    # 청크 경계에서 나뉜 시간을 다시 합친다
    merged = pd.concat(parts, ignore_index=True)
    agg = {"n": "sum", "power_sum": "sum", "ai_flags": "sum", "ai_score_max": "max"}
    agg.update({c: "min" for c in SENSORS})
    hourly = merged.groupby([MODULE, "hour_key"], sort=False).agg(agg).reset_index()
    hourly["datetime"] = pd.to_datetime(hourly.pop("hour_key").astype(str), format="%Y%m%d%H")
    hourly["kwh"] = hourly.pop("power_sum") / 720_000.0  # W × 5s / 3.6e6
    hourly = hourly.sort_values(["datetime", MODULE]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".parquet.partial")
    hourly.to_parquet(temporary, index=False)
    temporary.replace(OUT)
    stat = source.stat()
    meta = {
        "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source.name, "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "model_file": "models/anomaly_v2/selected_model.joblib", "model_sha256": _sha256(MODEL),
        "model_candidate": bundle["candidate"],
        "period_start": hourly["datetime"].min().isoformat(), "period_end": hourly["datetime"].max().isoformat(),
        "rows": len(hourly), "source_rows_scanned": seen,
        "incomplete_hours": int((hourly["n"] != 720).sum()),
        "energy_formula": "activePower(W) * 5 / 3,600,000 = kWh",
        "ai_thresholds": _ai_thresholds(bundle),
        "note": "AI 판정은 고정 모델 결과다. 12~2월은 이상탐지 모델 학습 기간이라 판정이 학습 데이터에 대한 결과다.",
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"READY: {len(hourly):,} equipment-hours, {meta['period_start']} ~ {meta['period_end']} "
          f"({time.monotonic() - start:.0f}s)", flush=True)


FORECAST_MODEL = ML / "models/forecast_v2/selected_model.joblib"
FORECAST_SAVED = ML / "outputs/forecast_v2/selected/may_equipment_physical.csv"
FORECAST_OUT = OUT_DIR / "dashboard_forecast.parquet"
FORECAST_META = OUT_DIR / "dashboard_forecast.json"
# 학습 마지막 시각(2025-04-29 23:00) 이후만 예측한다 → 학습에 쓰지 않은 미래 구간
FORECAST_START, FORECAST_END = "2025-04-30 00:00", "2025-05-31 23:00"


def build_forecast() -> None:
    """저장된 Extra Trees로 4/30~5/31 설비·시간별 전력량을 예측한다(재학습 없음).
    저장된 5월 예측(5/10~31 00시)과 값이 같은지 확인해 라이브러리 버전 차이를 검증한다."""
    import joblib
    import numpy as np
    sys.path.insert(0, str(ML / "src"))
    from manufacturing_ai.forecast import grid_for_dates

    model = joblib.load(FORECAST_MODEL)
    if pd.Timestamp(FORECAST_START) <= pd.Timestamp(model.last_training_time):
        raise SystemExit("예측 시작 시각은 학습 마지막 시각 이후여야 합니다.")
    grid = grid_for_dates(FORECAST_START, FORECAST_END, model.modules)
    grid["prediction_kWh"] = model.predict(grid)
    saved = pd.read_csv(FORECAST_SAVED, parse_dates=["datetime"])
    joined = grid.merge(saved, on=["datetime", MODULE], suffixes=("", "_saved"))
    diff = float(np.abs(joined["prediction_kWh"] - joined["prediction_kWh_saved"]).max())
    if len(joined) != len(saved) or diff > 1e-9:
        raise SystemExit(f"저장된 5월 예측과 일치하지 않습니다(최대 차이 {diff}). 라이브러리 버전을 확인하세요.")
    temporary = FORECAST_OUT.with_suffix(".parquet.partial")
    grid.to_parquet(temporary, index=False)
    temporary.replace(FORECAST_OUT)
    meta = {"complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_file": "models/forecast_v2/selected_model.joblib", "model_sha256": _sha256(FORECAST_MODEL),
            "model_kind": model.kind, "last_training_time": str(model.last_training_time),
            "period_start": FORECAST_START, "period_end": FORECAST_END, "rows": len(grid),
            "saved_may_rows_checked": len(joined), "saved_may_max_abs_diff": diff,
            "unit": "kWh per equipment-hour"}
    FORECAST_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FORECAST READY: {len(grid):,} equipment-hours, {FORECAST_START} ~ {FORECAST_END}, "
          f"saved-May max diff {diff:.2e}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(REPO / "data/rtu_data_full.csv"))
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--forecast-only", action="store_true", help="AI 전력 예측 캐시만 다시 만든다(원본 CSV 불필요)")
    parser.add_argument("--thresholds-only", action="store_true",
                        help="AI 이상탐지 경고 기준만 dashboard_hourly.json에 기록한다(원본 CSV 불필요, 몇 초)")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.thresholds_only:
        write_ai_thresholds()
        return
    if not args.forecast_only:
        source = Path(args.source).expanduser().resolve()
        if not source.is_file():
            raise SystemExit(f"원본 CSV를 찾지 못했습니다: {source}")
        build(source, args.chunksize)
    build_forecast()


if __name__ == "__main__":
    main()
