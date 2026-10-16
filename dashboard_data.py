"""Shared read-only dashboard data helpers. UI lives in dashboard_app.py."""
from __future__ import annotations

import calendar
import hashlib
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------
TEAM_NAME = "마가렛트"
PROJECT_TITLE = "AI 기반 제조 현장 전력량·탄소 배출량 예측 및 모니터링"
PROJECT_SUBTITLE = "과거 전력 데이터를 학습해 전력 소비량·전기요금·탄소 배출량을 예측하고 설비 이상을 모니터링합니다."
MAKERS = "박정열 · 김민찬 · 최경철"

NOW = datetime(2025, 4, 30, 0, 0)  # 과거 데이터 기준 시점 — 4월 29일 전체 실측 사용
TODAY = (NOW - timedelta(seconds=1)).date()                  # 마지막 실측일

# 요금/탄소 환산 계수 (명세)
WON_PER_KWH = 180.0
KGCO2_PER_KWH = 0.424

# 5월 정답(rtu_ground_truth_may.csv)의 hourly_pow와 모델 예측(hourly_pow_predicted)은 Wh 단위.
# 근거: 4월 센서 실측(activePower W × 5초 누적)이 공장 전체 시간당 약 39 kWh인데
#       hourly_pow는 시간당 약 39,000 → Wh로 봐야 규모가 일치. 에이전트도 같은 기준을 쓴다.
GT_WH_PER_KWH = 1000.0

# 의미 색상 (라이트/다크 공통) — 색은 의미에 맞게만 사용
SEMANTIC_COLORS = {
    "ok": "#16a34a",        # 정상=초록
    "bad": "#dc2626",       # 이상=빨강
    "actual": "#16a34a",    # 실제값=녹색
    "forecast": "#2563eb",  # 예측값=파랑
    "threshold": "#dc2626", # 임계치 가이드 선=빨강 점선
    "up": "#dc2626",        # 전월대비 증가=빨강
    "down": "#16a34a",      # 전월대비 감소=초록
}

# 라이트/다크 테마 팔레트 (구조색만. 의미색은 SEMANTIC_COLORS 공통 사용)
THEMES = {
    "light": {
        "bg": "#ffffff",
        "panel": "#ffffff",
        "border": "#e5e7eb",
        "row_border": "#f1f5f9",
        "text": "#111827",
        "text_muted": "#6b7280",
        "text_faint": "#9ca3af",
        "tab_active": "#047857",
        "plot_bg": "rgba(0,0,0,0)",
        "grid": "#e5e7eb",
        "axis": "#6b7280",
        "btn_bg": "#ffffff",
        "btn_text": "#111827",
        "btn_border": "#d1d5db",
        "btn_sel_bg": "#dcfce7",
        "btn_sel_text": "#065f46",
        "input_bg": "#ffffff",
    },
    "dark": {
        "bg": "#0e1117",
        "panel": "#161b22",
        "border": "#30363d",
        "row_border": "#21262d",
        "text": "#e6edf3",
        "text_muted": "#9da7b3",
        "text_faint": "#6e7681",
        "tab_active": "#2dd4bf",
        "plot_bg": "rgba(0,0,0,0)",
        "grid": "#30363d",
        "axis": "#9da7b3",
        "btn_bg": "#21262d",
        "btn_text": "#e6edf3",
        "btn_border": "#30363d",
        "btn_sel_bg": "#14532d",
        "btn_sel_text": "#bbf7d0",
        "input_bg": "#161b22",
    },
}

# KS 상 색상 (한달 그래프). 다크 모드에서 검정 선은 밝은 흰색 계열로 교체.
def ks_phase_colors(theme: str) -> dict:
    black = "#f5f5f5" if theme == "dark" else "#000000"
    return {
        "R": "#8B4513",   # 갈색
        "S": black,       # 검정 (다크: 흰색 계열)
        "T": "#9ca3af",   # 회색
        "RS": "#dc2626",  # 적색
        "ST": "#eab308",  # 황색
        "TR": "#2563eb",  # 청색
    }


# 임계치 기본값 (명세)
DEFAULT_THRESHOLDS = {
    "phase_voltage": 210.0,   # 상전압 R/S/T  (V)
    "line_voltage": 360.0,    # 선간전압 RS/ST/TR (V)
    "power_factor": 85.0,     # 역률 R/S/T (%)
}

# 설비 13개 — 이름·순서·채널번호 고정(명세). (채널번호, 표시이름)
EQUIPMENT_LIST = [
    ("1", "1 (PM-3)"),
    ("2", "2 (L-1전등)"),
    ("3", "3 (분쇄기(2))"),
    ("4", "4 (분쇄기(1))"),
    ("5", "5 (좌측분전반)"),
    ("11", "11 (우측분전반1)"),
    ("12", "12 (4호기)"),
    ("13", "13 (3호기)"),
    ("14", "14 (2호기)"),
    ("15", "15 (예비건조기)"),
    ("16", "16 (호이스트)"),
    ("17", "17 (6호기)"),
    ("18", "18 (우측분전반2)"),
]
EQUIPMENT_NAMES = [name for _, name in EQUIPMENT_LIST]

TABS = ["현황·예측", "이상탐지", "에이전트"]

# 점검 보고서 양식은 에이전트(agent/BasicSafetyReportForm.xlsx)가 관리한다.

# 원본 데이터 경로 (읽기 전용 — 절대 수정하지 않음)
WORKSPACE_ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = {
    # 명세 경로 우선, 없으면 레포 내 실제 파일로 폴백
    "ground_truth_may": [
        WORKSPACE_ROOT / "may-datasets" / "rtu_ground_truth_may.csv",
        WORKSPACE_ROOT / "data" / "rtu_ground_truth_may.csv",
    ],
    "rtu_full": [
        WORKSPACE_ROOT / "training-datasets" / "rtu_data_full.csv",
        WORKSPACE_ROOT / "rtu_data_full_flagged.csv",
    ],
}


# -----------------------------------------------------------------------------
# 실측 데이터 로딩 (자료가 없으면 빈 결과)
# -----------------------------------------------------------------------------
def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_ground_truth_may() -> tuple[pd.DataFrame, str]:
    """2025-05 실측(검증) 시간당 집계 데이터. 없으면 빈 자료 반환.
    반환: (DataFrame, source) — source는 'file:<path>' 또는 'missing'.
    컬럼: id(datetime), hourly_pow(원본, Wh), may_bill, may_carbon, agg_pow, kwh(환산)
    """
    path = _first_existing(DATA_CANDIDATES["ground_truth_may"])
    if path is not None:
        df = pd.read_csv(path)  # 읽기 전용
        df["id"] = pd.to_datetime(df["id"])
        source = f"file:{path.name}"
    else:
        df, source = pd.DataFrame(columns=["id", "hourly_pow"]), "missing"
        df["id"] = pd.to_datetime(df["id"])
    df["kwh"] = df["hourly_pow"] / GT_WH_PER_KWH  # 원본 Wh → kWh (화면·계산은 kwh만 사용)
    return df, source


@st.cache_data(show_spinner=False)
def load_rtu_raw_source() -> str:
    """원시 측정 데이터의 출처만 확인(대용량이라 전체 메모리 적재는 지양)."""
    path = _first_existing(DATA_CANDIDATES["rtu_full"])
    return f"file:{path.name}" if path is not None else "missing"


# -----------------------------------------------------------------------------
# AI 모델 연동 계층 (ml/ 산출물 · tools/prepare_dashboard_cache.py 캐시)
# -----------------------------------------------------------------------------
ML_ROOT = WORKSPACE_ROOT / "ml"
FORECAST_DIR = ML_ROOT / "outputs" / "forecast_v2"
FORECAST_MODEL_PATH = ML_ROOT / "models" / "forecast_v2" / "selected_model.joblib"
MODEL_LABELS = {
    "extra_trees": "Extra Trees", "random_forest": "Random Forest",
    "xgboost": "XGBoost", "hist_gradient_boosting": "HistGradientBoosting",
}
SENSOR_MODULE_COL = "module(equipment)"
SENSOR_ITEMS = ["voltageR", "voltageS", "voltageT", "voltageRS", "voltageST", "voltageTR",
                "powerFactorR", "powerFactorS", "powerFactorT"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


FORECAST_CACHE_PATH = ML_ROOT / "outputs" / "agent_data" / "dashboard_forecast.parquet"


@st.cache_data(show_spinner=False)
def load_equipment_forecast() -> tuple[pd.DataFrame, dict]:
    """선택된 예측 모델(forecast_v2, Extra Trees)의 설비·시간별 전력량 예측(kWh)과 메타정보.
    - 우선: tools/prepare_dashboard_cache.py가 만든 4/30~5/31 예측(학습 이후 미래 구간)
    - 없으면: ML 담당자가 저장한 5/10~5/31 00시 예측
    반환 df 컬럼: channel, equipment(대시보드 표기), datetime, date, hour, kwh
    모델 파일이 고정 기록과 다르면 예측을 쓰지 않는다(빈 df, meta['available']=False).
    """
    meta = {"available": False, "model": None, "mae": None, "note": ""}
    try:
        summary = json.loads((FORECAST_DIR / "summary.json").read_text(encoding="utf-8"))
        freeze = json.loads((FORECAST_DIR / "model_freeze.json").read_text(encoding="utf-8"))
        if _sha256(FORECAST_MODEL_PATH) != freeze["model_sha256"]["selected"]:
            raise ValueError("예측 모델 파일이 고정 기록과 일치하지 않습니다.")
        if FORECAST_CACHE_PATH.is_file():
            df = pd.read_parquet(FORECAST_CACHE_PATH)
        else:
            df = pd.read_csv(FORECAST_DIR / "selected" / "may_equipment_physical.csv", parse_dates=["datetime"])
        channel = df[SENSOR_MODULE_COL].str.split("(", n=1).str[0]
        df = pd.DataFrame({"channel": channel, "equipment": channel.map(dict(EQUIPMENT_LIST)),
                           "datetime": df["datetime"], "kwh": df["prediction_kWh"].astype(float)})
        df = df.dropna(subset=["equipment"])
        df = df[df["datetime"] >= pd.Timestamp(NOW)]  # 예측은 기준 시점 이후만 표시
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        family = summary.get("selected_family", "")
        meta.update(
            available=len(df) > 0,
            model=MODEL_LABELS.get(family, family),
            candidate=summary.get("selected_candidate"),
            mae=float(summary["selected_may"]["mae_kWh_if_reference_is_Wh"]),
            period=f"{pd.Timestamp('2025-05-10'):%m/%d %H시} ~ {pd.Timestamp('2025-05-31'):%m/%d %H시}",
            rows=len(df),
        )
        return df.sort_values(["datetime", "channel"]).reset_index(drop=True), meta
    except (OSError, ValueError, KeyError, TypeError) as exc:
        meta["note"] = f"AI 예측을 불러오지 못했습니다: {exc}"
        return pd.DataFrame(columns=["channel", "equipment", "datetime", "date", "hour", "kwh"]), meta


@st.cache_data(show_spinner=False)
def load_model_forecast() -> tuple[pd.DataFrame, dict]:
    """공장 전체 시간당 AI 예측(kWh). 반환: (df[id, pred], meta)"""
    eq, meta = load_equipment_forecast()
    g = eq.groupby("datetime")["kwh"].sum() if len(eq) else pd.Series(dtype=float)
    return pd.DataFrame({"id": g.index, "pred": g.to_numpy()}), meta


def model_pred_for(day: date) -> pd.Series:
    """선택일의 시간(0~23)별 모델 예측. 예측이 없는 시간은 Series에 없음."""
    fc, _ = load_model_forecast()
    d = fc[fc["id"].dt.date == day]
    return d.groupby(d["id"].dt.hour)["pred"].mean()


EQUIPMENT_HOURLY_PATH = ML_ROOT / "outputs" / "agent_data" / "dashboard_hourly.parquet"
FULL_HOUR_RECORDS = 720  # 5초 × 720 = 1시간


@st.cache_data(show_spinner=False)
def load_equipment_hourly() -> pd.DataFrame:
    """12~4월 설비·시간별 실측 집계(tools/prepare_dashboard_cache.py 산출물).
    컬럼: channel, equipment(대시보드 표기), datetime, date, hour, n, kwh,
          전압·역률 9개(시간 내 최저값), ai_flag(Isolation Forest 이상 여부), ai_score(최대)
    캐시가 없으면 빈 DataFrame."""
    if not EQUIPMENT_HOURLY_PATH.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(EQUIPMENT_HOURLY_PATH)
    channel = df[SENSOR_MODULE_COL].str.split("(", n=1).str[0]
    df["channel"] = channel
    df["equipment"] = channel.map(dict(EQUIPMENT_LIST))
    df = df.dropna(subset=["equipment"]).drop(columns=[SENSOR_MODULE_COL])
    # 기준 시점까지 끝난 시간만 실측으로 쓴다(예: 09:00 기준이면 08시 구간까지)
    df = df[df["datetime"] + pd.Timedelta(hours=1) <= pd.Timestamp(NOW)]
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["ai_flag"] = df["ai_flags"] > 0
    df = df.rename(columns={"ai_score_max": "ai_score"})
    return df.sort_values(["datetime", "channel"]).reset_index(drop=True)


def sensor_dates() -> list[str]:
    """설비별 센서 실측이 있는 날짜 목록(YYYY-MM-DD)."""
    eh = load_equipment_hourly()
    return sorted({d.isoformat() for d in eh["date"]}) if len(eh) else []


@st.cache_data(show_spinner=False)
def factory_hourly_actual() -> pd.Series:
    """공장 전체 시간당 실측 전력량(kWh), 기준 시점(NOW)까지 끝난 시간만.
    - 12/1 ~ 4/29: 설비별 5초 실측 합(13개 설비 모두 완전한 시간만)
    - 5/10 ~ 5/31: 5월 정답 파일(공장 전체, Wh → kWh) — 기준 시점 이후면 제외"""
    parts = []
    eh = load_equipment_hourly()
    if len(eh):
        g = eh.groupby("datetime").agg(kwh=("kwh", "sum"), n=("n", "sum"), m=("channel", "nunique"))
        full = g[(g["m"] == len(EQUIPMENT_LIST)) & (g["n"] == FULL_HOUR_RECORDS * len(EQUIPMENT_LIST))]
        parts.append(full["kwh"])
    gt, _ = load_ground_truth_may()
    parts.append(gt.set_index("id")["kwh"])
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s[s.index + pd.Timedelta(hours=1) <= pd.Timestamp(NOW)]


@st.cache_data(show_spinner=False)
def sensor_equipment_daily(day_iso: str) -> pd.DataFrame | None:
    """선택일의 설비·시간별 실측(이상탐지 판단용 전압·역률은 시간별 최저값). 없으면 None."""
    eh = load_equipment_hourly()
    if not len(eh):
        return None
    d = eh[eh["date"] == date.fromisoformat(day_iso)]
    if d.empty:
        return None
    cols = ["channel", "equipment", "hour", *SENSOR_ITEMS, "kwh", "n", "ai_flag", "ai_score"]
    return d[cols].sort_values(["channel", "hour"]).reset_index(drop=True)


EQUIPMENT_DAY_COLUMNS = ["channel", "equipment", "hour", *SENSOR_ITEMS, "kwh", "ai_flag", "ai_score"]


def equipment_daily(day: date) -> tuple[pd.DataFrame, str]:
    """설비별 하루치 데이터(더미 없음).
    반환: (df, source)
      - 'sensor'  : 센서 실측(전력·전압·역률·AI 판정)
      - 'forecast': AI 설비별 전력 예측만(전압·역률은 예측 대상이 아니라 비어 있음)
      - 'none'    : 해당 일 데이터 없음(빈 df)"""
    real = sensor_equipment_daily(day.isoformat())
    if real is not None and len(real):
        return real, "sensor"
    eq, _ = load_equipment_forecast()
    f = eq[eq["date"] == day] if len(eq) else eq
    if len(f):
        out = f[["channel", "equipment", "hour", "kwh"]].copy()
        for c in SENSOR_ITEMS:
            out[c] = np.nan
        out["ai_flag"], out["ai_score"] = False, np.nan
        return out[EQUIPMENT_DAY_COLUMNS].reset_index(drop=True), "forecast"
    return pd.DataFrame(columns=EQUIPMENT_DAY_COLUMNS), "none"


NO_DATA_TEXT = "해당 일은 데이터가 없습니다."


def is_forecast_month(year: int, month: int) -> bool:
    """기준 시점보다 뒤의 달(예: 5월)은 AI 예측으로 보여주는 '예상' 화면."""
    return (year, month) > (TODAY.year, TODAY.month)


def to_display_equipment(value) -> str:
    """'12(4호기)'(agent/ML 표기) 등 어떤 표기든 대시보드 표기 '12 (4호기)'로 변환."""
    token = re.sub(r"[\s()]", "", str(value or ""))
    for name in EQUIPMENT_NAMES:
        if re.sub(r"[\s()]", "", name) == token:
            return name
    m = re.match(r"\s*(\d+)", str(value or ""))
    if m:
        for ch, name in EQUIPMENT_LIST:
            if ch == m.group(1):
                return name
    return EQUIPMENT_NAMES[0]


# -----------------------------------------------------------------------------
# 파생 계산 (Task 2.3)
# -----------------------------------------------------------------------------
def kwh_from_active_power_5s(active_power_kw: np.ndarray | pd.Series) -> float:
    """activePower(kW) × (5초/3600초)를 5초 단위로 누적 합산 → kWh."""
    return float(np.sum(np.asarray(active_power_kw, dtype=float) * (5.0 / 3600.0)))


def bill_from_kwh(kwh: float) -> float:
    """전기요금(원) = 전력량(kWh) × 180원."""
    return kwh * WON_PER_KWH


def carbon_from_kwh(kwh: float) -> float:
    """탄소 배출량(kgCO2) = 전력량(kWh) × 0.424."""
    return kwh * KGCO2_PER_KWH


def won(x: float) -> str:
    return f"{x:,.0f} 원"


def kwh(x: float) -> str:
    return f"{x:,.0f} kWh"


def kgco2(x: float) -> str:
    return f"{x:,.0f} kgCO2"


# -----------------------------------------------------------------------------
# 집계 / 예측 (Task 8~9)
# -----------------------------------------------------------------------------
def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    n = min(len(actual), len(pred))
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(np.asarray(actual[:n]) - np.asarray(pred[:n]))))


def actual_until_now() -> pd.DataFrame:
    """기준 시점(NOW)까지의 공장 전체 시간당 실측(kWh). 컬럼: id, kwh
    12~4월은 설비별 센서 합, 5월은 정답 파일. 이후 시각은 아직 오지 않은 시간."""
    s = factory_hourly_actual()
    return pd.DataFrame({"id": s.index, "kwh": s.to_numpy()})


def month_hourly_values(year: int, month: int) -> pd.DataFrame:
    """해당 월의 시간별 전력량: 실측(NOW까지) 우선, 없으면 AI 예측.
    컬럼: id, actual, pred, value(실측 우선 결합값)"""
    start = pd.Timestamp(year, month, 1)
    idx = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="h", inclusive="left")
    actual = actual_until_now().set_index("id")["kwh"].groupby(level=0).mean()
    fc, _ = load_model_forecast()
    pred = fc.set_index("id")["pred"].groupby(level=0).mean()
    out = pd.DataFrame({"id": idx, "actual": actual.reindex(idx).to_numpy(),
                        "pred": pred.reindex(idx).to_numpy()})
    out["value"] = out["actual"].where(out["actual"].notna(), out["pred"])
    return out


@st.cache_data(show_spinner=False)
def monthly_power_series(year: int, month: int) -> pd.DataFrame:
    """월별(1~말일) 일 단위 전력량/탄소 실제·예측 추이.
    - 실제: 기준 시점까지 24시간이 모두 측정된 날만 일 합계(진행 중인 오늘은 제외)
    - 예측: AI 모델(forecast_v2 선택 모델)의 시간별 예측이 24시간 모두 있는 날의 일 합계
    데이터가 없는 날은 NaN(선 끊김).
    """
    hv = month_hourly_values(year, month)
    hv["day"] = hv["id"].dt.day
    grouped = hv.groupby("day").agg(a_sum=("actual", "sum"), a_n=("actual", "count"),
                                    p_sum=("pred", "sum"), p_n=("pred", "count"))
    days = list(range(1, calendar.monthrange(year, month)[1] + 1))
    grouped = grouped.reindex(days)
    actual = np.where(grouped["a_n"] == 24, grouped["a_sum"], np.nan).astype(float)
    pred = np.where(grouped["p_n"] == 24, grouped["p_sum"], np.nan).astype(float)
    return pd.DataFrame({
        "day": days,
        "power_actual": actual,
        "power_pred": pred,
        "carbon_actual": actual * KGCO2_PER_KWH,
        "carbon_pred": pred * KGCO2_PER_KWH,
    })


@st.cache_data(show_spinner=False)
def hourly_power_series(day: date) -> pd.DataFrame:
    """선택일의 시간별(0~23시) 전력량 실제·예측.
    실측 ground truth에서 해당 날짜의 시간당 전력량(기준 시점까지)을 사용하고,
    예측은 AI 모델(forecast_v2 선택 모델)의 시간별 예측. 데이터 없으면 NaN(선 끊김)."""
    gt = actual_until_now()
    d = gt[gt["id"].dt.date == day].copy()
    hours = list(range(0, 24))
    if len(d):
        d["hour"] = d["id"].dt.hour
        s = d.groupby("hour")["kwh"].mean()
        actual = np.array([s.get(h, np.nan) for h in hours], dtype=float)
    else:
        actual = np.full(len(hours), np.nan)
    p = model_pred_for(day)
    pred = np.array([p.get(h, np.nan) for h in hours], dtype=float)
    return pd.DataFrame({"hour": hours, "actual": actual, "pred": pred})


def equipment_monthly(equipment: str, year: int, month: int) -> pd.DataFrame:
    """특정 설비의 월별(1~말일) 상별 '그 날 최저값'. 센서 실측이 있는 날만. (한달 기준 그래프용)
    상전압 R/S/T, 선간전압 RS/ST/TR, 역률 R/S/T.
    """
    rows = []
    for d in month_days(year, month):
        day = date(year, month, d)
        g, source = equipment_daily(day)
        if source != "sensor":
            continue
        gg = g[g["equipment"] == equipment]
        if gg.empty:
            continue
        rows.append({
            "day": d,
            "R": float(gg["voltageR"].min()), "S": float(gg["voltageS"].min()),
            "T": float(gg["voltageT"].min()),
            "RS": float(gg["voltageRS"].min()), "ST": float(gg["voltageST"].min()),
            "TR": float(gg["voltageTR"].min()),
            "pfR": float(gg["powerFactorR"].min()), "pfS": float(gg["powerFactorS"].min()),
            "pfT": float(gg["powerFactorT"].min()),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def equipment_power_ranking(day: date) -> pd.DataFrame:
    """설비별 하루 전력 소비량(kWh) 내림차순 + 비율(%).
    센서 실측일은 5초 유효전력 누적(오늘은 기준 시점까지), 예측일은 AI 설비별 예측 합.
    데이터가 없으면 빈 DataFrame."""
    df, _ = equipment_daily(day)
    if df.empty:
        return pd.DataFrame(columns=["equipment", "kwh", "ratio", "rank"])
    out = (df.groupby("equipment", sort=False)["kwh"].sum().rename("kwh").reset_index()
           .sort_values("kwh", ascending=False).reset_index(drop=True))
    total = out["kwh"].sum() or 1.0
    out["ratio"] = out["kwh"] / total * 100.0
    out["rank"] = out.index + 1
    return out


def month_power_estimate(year: int, month: int) -> float | None:
    """월 전체 전력량(kWh) 추정: 시간별 실측(없으면 AI 예측) 합 + 둘 다 없는 시간은
    시간 평균으로 보정. 해당 월에 자료가 전혀 없으면 None."""
    hv = month_hourly_values(year, month)
    valid = hv["value"].dropna()
    if not len(valid):
        return None
    return float(valid.sum() + valid.mean() * (len(hv) - len(valid)))


def monthly_kpis(year: int, month: int, day: date, thresholds: dict) -> dict:
    """현황·예측 KPI: 금월 예상(요금/전력/탄소), 전월 대비(%), 일일 누적, MAE.
    - 금월 예상: 시간별로 기준 시점까지는 실측, 이후는 AI 예측을 합산하고
      둘 다 없는 시간은 해당 월의 관측·예측 시간 평균으로 보정 (화면에 구분 표시)
    - 전월 대비: 같은 방식으로 계산한 전월 값과 비교(전월 자료가 없으면 None)
    - 일일 누적: 선택일의 기준 시점까지 실측 합(기준일이면 기준 시점 이전까지)
    - MAE: AI 모델 평가 구간 전체(summary.json) + 선택일 실측 대비 오차
    값을 만들 근거가 없으면 None.
    """
    month_power_est = month_power_estimate(year, month)
    prev = pd.Timestamp(year, month, 1) - pd.offsets.MonthBegin(1)
    prev_power = month_power_estimate(prev.year, prev.month)
    delta_pct = ((month_power_est - prev_power) / prev_power * 100.0
                 if month_power_est is not None and prev_power else None)

    # 일일 누적(선택일): 실측 우선, 실측이 없으면 None
    hdf = hourly_power_series(day)
    day_power = float(np.nansum(hdf["actual"])) if hdf["actual"].notna().any() else None

    # 선택일 MAE(시간당) — 실측과 예측이 모두 있는 시간만
    pair = hdf.dropna()
    day_mae = mae(pair["actual"].to_numpy(), pair["pred"].to_numpy()) if len(pair) else None
    _, meta = load_model_forecast()

    def _opt(fn, v):
        return fn(v) if v is not None else None

    return {
        "month_bill": _opt(bill_from_kwh, month_power_est),
        "month_power": month_power_est,
        "month_carbon": _opt(carbon_from_kwh, month_power_est),
        "delta_pct": delta_pct,
        "day_bill": _opt(bill_from_kwh, day_power),
        "day_power": day_power,
        "day_carbon": _opt(carbon_from_kwh, day_power),
        "mae": meta.get("mae"),
        "day_mae": day_mae,
        "model": meta.get("model"),
        "model_period": meta.get("period"),
    }


# -----------------------------------------------------------------------------
# 이상 판정 (Task 6 상태 원 / Task 10 그래프 공용)
# -----------------------------------------------------------------------------
def analyze_equipment_anomalies(day: date, thresholds: dict) -> dict[str, dict]:
    """선택 날짜의 설비별 이상 여부 판정.
    반환: {설비명: {"phase", "line", "pf", "ai", "any": bool}}
    - phase: 상전압 R/S/T 중 하나라도 임계 미만 (규칙)
    - line : 선간전압 RS/ST/TR 중 하나라도 임계 미만 (규칙)
    - pf   : 역률 R/S/T 중 하나라도 임계 미만 (규칙)
    - ai   : Isolation Forest 모델 이상 판정 (센서 실측일만; 규칙 임계치와 무관하게 고정)
    센서 실측이 없는 날(예측일 포함)은 판정할 수 없으므로 빈 dict.
    """
    df, source = equipment_daily(day)
    if source != "sensor":
        return {}
    ph_th = thresholds["phase_voltage"]
    ln_th = thresholds["line_voltage"]
    pf_th = thresholds["power_factor"]
    result: dict[str, dict] = {}
    for name, g in df.groupby("equipment", sort=False):
        phase = bool((g[["voltageR", "voltageS", "voltageT"]] < ph_th).any().any())
        line = bool((g[["voltageRS", "voltageST", "voltageTR"]] < ln_th).any().any())
        pf = bool((g[["powerFactorR", "powerFactorS", "powerFactorT"]] < pf_th).any().any())
        ai = bool(g["ai_flag"].any())
        result[name] = {"phase": phase, "line": line, "pf": pf, "ai": ai,
                        "any": phase or line or pf or ai}
    # EQUIPMENT_LIST 순서 보장
    return {name: result[name] for name in EQUIPMENT_NAMES if name in result}


def anomaly_tooltip(flags: dict) -> str:
    """상태 원 hover 툴팁용 이상 항목 문자열."""
    items = []
    if flags.get("phase"):
        items.append("상전압")
    if flags.get("line"):
        items.append("선간전압")
    if flags.get("pf"):
        items.append("역률")
    if flags.get("ai"):
        items.append("AI 이상탐지")
    return " / ".join(items) if items else "이상 없음"


def month_days(year: int, month: int) -> list[int]:
    """사이드바 일자 선택지: 데이터 유무와 관계없이 1일~말일 전체."""
    return list(range(1, calendar.monthrange(year, month)[1] + 1))


def available_months() -> list[tuple[int, int]]:
    """사이드바에 노출할 (연, 월) 목록: 실측이 있는 월 + AI 예측이 있는 월(5월)."""
    months = {(d.year, d.month) for d in actual_until_now()["id"]}
    months |= {(int(s[:4]), int(s[5:7])) for s in sensor_dates()}
    fc, _ = load_model_forecast()
    months |= {(t.year, t.month) for t in fc["id"]}
    months.add((TODAY.year, TODAY.month))
    return sorted(months)


# -----------------------------------------------------------------------------
# 전역 상태 초기화 (Task 3)
# -----------------------------------------------------------------------------
