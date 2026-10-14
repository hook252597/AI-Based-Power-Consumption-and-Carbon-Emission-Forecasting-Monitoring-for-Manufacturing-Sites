# -*- coding: utf-8 -*-
# =============================================================================
# 마가렛트 — AI 기반 제조 현장 전력량·탄소 배출량 예측 및 모니터링 대시보드
# -----------------------------------------------------------------------------
# 실행 방법:
#   1) 의존성 설치 (버전 고정)
#        pip install streamlit==1.39.0 plotly==5.24.1 pandas==2.2.3
#        pip install openai openpyxl pyarrow python-docx==1.1.2 xgboost==3.1.3   # 에이전트·예측 캐시
#   2) (최초 1회) 12~4월 설비별 실측 + AI 판정, 4/30~5월 AI 예측 캐시 준비 — 수 분 소요
#        python tools/prepare_dashboard_cache.py
#   3) 앱 실행
#        streamlit run dashboard_app.py
#
# AI 연동:
#   - 전력 예측: ml/models/forecast_v2 선택 모델(Extra Trees)의 4/30~5/31 설비별 예측.
#     모델 파일 SHA256이 고정 기록(model_freeze.json)과 일치할 때만 사용.
#   - 이상탐지: ml/models/anomaly_v2 Isolation Forest의 고정 판정(12~4월 설비·시간별 캐시).
#   - 에이전트 탭: agent/ 패키지(agent/agent.md 명세, agent/agentSystem.md 프롬프트).
#     OPENAI_API_KEY가 없으면 데모 모드(같은 도구를 규칙으로 호출)로 동작합니다.
#
# 주의:
#   - 원본 데이터 파일(data/rtu_ground_truth_may.csv, rtu_data_full_flagged.csv 등)은
#     읽기 전용으로만 접근하며 절대 수정/덮어쓰기하지 않습니다.
#
# 기준 해상도: 1366 x 768 (노트북). 세로 스크롤 없이 한 화면.
# 고정 기준 시점("오늘·지금"): 2025-04-29 09:00 (설비별 센서 실측의 마지막 날)
#   - 이 시각 이후의 실측은 '아직 오지 않은 시간'으로 보고 화면에 쓰지 않는다.
#   - 12~4월: 설비별 실측(전력·전압·역률·AI 판정) / 5월: AI 예측('예상' 화면)
#   - 데이터가 없는 날은 더미로 채우지 않고 '해당 일은 데이터가 없습니다'로 표시한다.
#   - 에이전트 기준 시점(agent/agent_tools.py NOW)과 같은 값이어야 한다.
# =============================================================================

from __future__ import annotations

import calendar
import hashlib
import json
import re
from datetime import date, datetime
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

NOW = datetime(2025, 4, 29, 9, 0)  # 고정 기준 시점(현재 시각) — 설비별 실측의 마지막 날 09시
TODAY = NOW.date()                  # 고정 기준일(오늘)

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
# 데이터 로딩 / 더미 생성 계층 (Task 2)
# -----------------------------------------------------------------------------
def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_ground_truth_may() -> tuple[pd.DataFrame, str]:
    """2025-05 실측(검증) 시간당 집계 데이터. 없으면 더미 생성.
    반환: (DataFrame, source) — source는 'file:<path>' 또는 'dummy'.
    컬럼: id(datetime), hourly_pow(원본, Wh), may_bill, may_carbon, agg_pow, kwh(환산)
    """
    path = _first_existing(DATA_CANDIDATES["ground_truth_may"])
    if path is not None:
        df = pd.read_csv(path)  # 읽기 전용
        df["id"] = pd.to_datetime(df["id"])
        source = f"file:{path.name}"
    else:
        df, source = _dummy_ground_truth_may(), "dummy"
    df["kwh"] = df["hourly_pow"] / GT_WH_PER_KWH  # 원본 Wh → kWh (화면·계산은 kwh만 사용)
    return df, source


def _dummy_ground_truth_may() -> pd.DataFrame:
    """명세와 같은 구조의 시간당 더미 실측 데이터(2025-05-10 ~ 2025-05-31)."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-05-10 00:00:00", "2025-05-31 00:00:00", freq="h")
    hour = idx.hour.values
    # 주간(08~18시) 부하가 높은 일중 패턴
    base = 30000 + 12000 * np.sin((hour - 6) / 24 * 2 * np.pi).clip(min=-0.6)
    hourly_pow = base + rng.normal(0, 800, len(idx))
    hourly_pow = np.clip(hourly_pow, 15000, None)
    bill = hourly_pow * WON_PER_KWH
    carbon = hourly_pow * KGCO2_PER_KWH
    agg = np.cumsum(hourly_pow) + 19_700_000
    return pd.DataFrame(
        {
            "id": idx,
            "hourly_pow": hourly_pow,
            "may_bill": bill,
            "may_carbon": carbon,
            "agg_pow": agg,
        }
    )


@st.cache_data(show_spinner=False)
def load_rtu_raw_source() -> str:
    """원시 측정 데이터의 출처만 확인(대용량이라 전체 메모리 적재는 지양)."""
    path = _first_existing(DATA_CANDIDATES["rtu_full"])
    return f"file:{path.name}" if path is not None else "dummy"


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
      둘 다 없는 시간(5/1~9 등)은 시간 평균으로 보정
    - 전월 대비: 같은 방식으로 계산한 전월 값과 비교(전월 자료가 없으면 None)
    - 일일 누적: 선택일의 기준 시점까지 실측 합(오늘이면 09시까지)
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
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("selected_month", TODAY.month)      # 1~12
    ss.setdefault("selected_year", TODAY.year)
    ss.setdefault("selected_day", TODAY.day)          # 1~말일
    ss.setdefault("active_tab", TABS[0])              # 현황·예측
    ss.setdefault("theme", "light")                   # light/dark (초기값; OS 연동은 Task 4)
    ss.setdefault("selected_equipment", EQUIPMENT_NAMES[0])  # 전역 공유 선택 설비
    ss.setdefault("anomaly_view", "daily")            # daily/monthly
    ss.setdefault("thresholds", dict(DEFAULT_THRESHOLDS))  # 에이전트와 공유
    ss.setdefault("threshold_history", [])                 # 에이전트가 기록(dict)
    # 어떤 표기('12(4호기)' 등)로 들어와도 대시보드 표기로 통일
    ss["selected_equipment"] = to_display_equipment(ss["selected_equipment"])


# -----------------------------------------------------------------------------
# 페이지 설정 & 전역 CSS (4영역 골격)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{TEAM_NAME} · 전력·탄소 모니터링 대시보드",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def resolve_initial_theme() -> str:
    """최초 1회 OS 설정(prefers-color-scheme)을 쿼리파라미터로 받아 테마 결정.
    JS가 ?mg_theme=dark 를 붙여 1회 리런시키고, 그 뒤에는 사용자의 선택을 기억한다.
    """
    ss = st.session_state
    if ss.get("_theme_from_os_applied"):
        return ss["theme"]
    qp = st.query_params.get("mg_theme")
    if qp in ("light", "dark"):
        ss["theme"] = qp
        ss["_theme_from_os_applied"] = True
    else:
        # OS 설정을 읽어 쿼리파라미터로 재진입 (한 번만)
        st.markdown(
            """
            <script>
            (function() {
              const params = new URLSearchParams(window.location.search);
              if (!params.has('mg_theme')) {
                const dark = window.matchMedia &&
                  window.matchMedia('(prefers-color-scheme: dark)').matches;
                params.set('mg_theme', dark ? 'dark' : 'light');
                window.location.search = params.toString();
              }
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )
    return ss["theme"]


def theme_css(theme: str) -> None:
    t = THEMES[theme]
    st.markdown(
        f"""
        <style>
          :root {{
            --mg-bg: {t['bg']}; --mg-panel: {t['panel']}; --mg-border: {t['border']};
            --mg-text: {t['text']}; --mg-text-muted: {t['text_muted']};
            --mg-text-faint: {t['text_faint']};
          }}

          /* 1366x768 기준: 한 화면에 담기 위해 기본 패딩 축소 */
          .block-container {{ padding-top: 0.6rem; padding-bottom: 0.4rem;
                             max-width: 100% !important; }}
          header[data-testid="stHeader"] {{ height: 0; }}
          #MainMenu, footer {{ visibility: hidden; }}

          /* 테마 배경/텍스트 (앱 전역 + 기본 텍스트 요소) */
          .stApp {{ background: {t['bg']}; color: {t['text']}; }}
          .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
          .stMarkdown, [data-testid="stMarkdownContainer"],
          [data-testid="stCaptionContainer"], .stCaption {{
            color: {t['text']};
          }}
          /* Streamlit 캡션/보조 텍스트는 muted */
          [data-testid="stCaptionContainer"] * {{ color: {t['text_muted']} !important; }}

          /* ---- 사이드바(② 영역): 테마에 맞춰 배경/글자 전환 ---- */
          [data-testid="stSidebar"] {{
            background: {t['panel']} !important;
            border-right: 1px solid {t['border']};
          }}
          /* 사이드바 일반 텍스트(라벨/마크다운). 버튼 내부는 제외하여 버튼 규칙과 충돌 방지 */
          [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
          [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {{
            color: {t['text']} !important;
          }}
          [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
          [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
            color: {t['text_muted']} !important;
          }}

          .mg-header-title {{ font-size: 24px; font-weight: 800; color: {t['text']}; }}
          .mg-header-proj  {{ font-size: 14px; font-weight: 600; color: {t['text_muted']}; }}
          .mg-header-sub   {{ font-size: 12px; color: {t['text_muted']}; margin-top: 2px; }}
          .mg-maker        {{ text-align: right; font-size: 11px; color: {t['text_faint']}; }}
          .mg-maker b      {{ color: {t['text_muted']}; }}

          .mg-side-box {{ border: 1px solid {t['border']}; border-radius: 8px;
                         padding: 8px; height: 600px; overflow: hidden;
                         background: {t['panel']}; }}
          .mg-equip-row {{ display:flex; justify-content:space-between;
                          align-items:center; height: 40px;
                          border-bottom: 1px solid {t['row_border']};
                          font-size: 13px; color: {t['text']}; }}
          .mg-dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
          .mg-dot.ok {{ background:{SEMANTIC_COLORS['ok']}; }}   /* 정상=초록 */
          .mg-dot.bad {{ background:{SEMANTIC_COLORS['bad']}; }}  /* 이상=빨강 */
          .mg-dot.none {{ background:#9ca3af; }}                 /* 데이터 없음=회색 */

          .mg-main-box {{ border: 1px solid {t['border']}; border-radius: 8px;
                         padding: 10px; min-height: 600px; background: {t['panel']}; }}

          /* ---- Streamlit 버튼 색상(테마 연동) ----
             Streamlit 1.39의 실제 DOM: <button data-testid="stBaseButton-secondary|primary">
             (이전에 쓰던 kind="..."/baseButton-... 선택자는 실제 DOM과 불일치해 미적용이었음) */

          /* 선택 안 된(secondary) 버튼: 테마 배경 + 테마 글자색 */
          [data-testid="stBaseButton-secondary"] {{
            background: {t['btn_bg']} !important;
            border: 1px solid {t['btn_border']} !important;
            color: {t['btn_text']} !important;
          }}
          [data-testid="stBaseButton-secondary"] p,
          [data-testid="stBaseButton-secondary"] span,
          [data-testid="stBaseButton-secondary"] div {{
            color: {t['btn_text']} !important;
          }}
          [data-testid="stBaseButton-secondary"]:hover {{
            border-color: {SEMANTIC_COLORS['ok']} !important;
            color: {t['btn_sel_text']} !important;
          }}

          /* 선택(primary) 버튼: 연녹색 배경 + 가독성 높은 글자 */
          [data-testid="stBaseButton-primary"] {{
            background: {t['btn_sel_bg']} !important;
            border: 1px solid {SEMANTIC_COLORS['ok']} !important;
            color: {t['btn_sel_text']} !important;
            font-weight: 700 !important;
          }}
          [data-testid="stBaseButton-primary"] p,
          [data-testid="stBaseButton-primary"] span,
          [data-testid="stBaseButton-primary"] div {{
            color: {t['btn_sel_text']} !important;
          }}

          /* 사이드바 스코프를 포함해 특이도를 높인 재지정
             (사이드바 일반 텍스트 규칙이 버튼 라벨 색을 덮지 못하도록) */
          [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
          [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p,
          [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] span,
          [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] div {{
            color: {t['btn_text']} !important;
          }}
          [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
            background: {t['btn_bg']} !important;
            border: 1px solid {t['btn_border']} !important;
          }}
          [data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
          [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p,
          [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] span,
          [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] div {{
            color: {t['btn_sel_text']} !important;
          }}
          [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
            background: {t['btn_sel_bg']} !important;
            border: 1px solid {SEMANTIC_COLORS['ok']} !important;
          }}

          /* 라디오(탭/테마/기준) 글자색 */
          [data-testid="stRadio"] label, [data-testid="stRadio"] p,
          [role="radiogroup"] label, [role="radiogroup"] p {{
            color: {t['text']} !important;
          }}

          /* 셀렉트박스(월/일) */
          [data-baseweb="select"] > div {{
            background: {t['input_bg']} !important;
            border-color: {t['btn_border']} !important;
          }}
          [data-baseweb="select"] div, [data-baseweb="select"] span,
          [data-baseweb="select"] input {{ color: {t['text']} !important; }}
          /* 셀렉트박스 펼침 목록(드롭다운). Streamlit 1.39 실제 DOM:
             data-testid="stSelectboxVirtualDropdown".
             이 목록은 <body> 직속에 떠서 baseweb 자체 색을 쓰므로,
             컨테이너와 "모든 하위 요소"에 테마 배경/글자를 강제해 라이트/다크를 분리한다. */
          [data-testid="stSelectboxVirtualDropdown"],
          [data-testid="stSelectboxVirtualDropdown"] > div,
          [data-testid="stSelectboxVirtualDropdown"] ul,
          [data-testid="stSelectboxVirtualDropdown"] li,
          [data-testid="stSelectboxVirtualDropdown"] [role="listbox"],
          [data-testid="stSelectboxVirtualDropdown"] [role="option"] {{
            background: {t['panel']} !important;
          }}
          [data-testid="stSelectboxVirtualDropdown"] li,
          [data-testid="stSelectboxVirtualDropdown"] li *,
          [data-testid="stSelectboxVirtualDropdown"] [role="option"],
          [data-testid="stSelectboxVirtualDropdown"] [role="option"] * {{
            color: {t['text']} !important;
          }}
          /* 마우스오버/선택된 옵션: 연녹색 배경 + 가독성 글자 */
          [data-testid="stSelectboxVirtualDropdown"] li:hover,
          [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
          [data-testid="stSelectboxVirtualDropdown"] [aria-selected="true"] {{
            background: {t['btn_sel_bg']} !important;
          }}
          [data-testid="stSelectboxVirtualDropdown"] li:hover *,
          [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover *,
          [data-testid="stSelectboxVirtualDropdown"] [aria-selected="true"] * {{
            color: {t['btn_sel_text']} !important;
          }}

          /* expander(체크리스트/이력) */
          [data-testid="stExpander"] summary, [data-testid="stExpander"] p {{
            color: {t['text']} !important;
          }}
          [data-testid="stExpander"] details {{
            background: {t['panel']} !important;
            border-color: {t['border']} !important;
          }}

          /* chat 입력창 */
          [data-testid="stChatInput"] textarea {{
            color: {t['text']} !important; background: {t['input_bg']} !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(theme: str) -> dict:
    """테마에 맞춘 Plotly 공통 레이아웃. 그래프 생성 시 fig.update_layout(**plotly_layout(theme))."""
    t = THEMES[theme]
    return {
        "paper_bgcolor": t["plot_bg"],
        "plot_bgcolor": t["plot_bg"],
        "font": {"color": t["text"], "size": 11},
        "xaxis": {"gridcolor": t["grid"], "linecolor": t["axis"],
                  "tickcolor": t["axis"], "zerolinecolor": t["grid"]},
        "yaxis": {"gridcolor": t["grid"], "linecolor": t["axis"],
                  "tickcolor": t["axis"], "zerolinecolor": t["grid"]},
        "margin": {"l": 40, "r": 10, "t": 30, "b": 30},
        "legend": {"font": {"color": t["text"], "size": 10}},
    }


# -----------------------------------------------------------------------------
# ① 헤더
# -----------------------------------------------------------------------------
def render_header() -> None:
    left, right = st.columns([5, 2], vertical_alignment="center")
    with left:
        st.markdown(
            f"""
            <div style="display:flex; align-items:baseline; gap:12px; flex-wrap:wrap;">
                <span class="mg-header-title">🟩 {TEAM_NAME}</span>
                <span class="mg-header-proj">{PROJECT_TITLE}</span>
            </div>
            <div class="mg-header-sub">{PROJECT_SUBTITLE}</div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="mg-maker">기준 시점(현재) <b>{NOW:%Y-%m-%d %H:%M}</b><br>'
            f'제작 <b>{MAKERS}</b></div>',
            unsafe_allow_html=True,
        )
    st.divider()


# -----------------------------------------------------------------------------
# ② 필터 · 설비 목록 (좌측)
# -----------------------------------------------------------------------------
def render_sidebar(gt_source: str) -> None:
    ss = st.session_state
    st.markdown("**조회 기간**")

    # 월/일 선택 — 위젯 key를 전역 상태와 매 런 동기화(위젯 생성 전에만 세팅 가능).
    # index=로 초기값을 주면 선택이 바뀔 때마다 위젯이 새로 만들어져
    # 다음 클릭이 무시되는 문제가 있어 key + on_change 방식으로 고정한다.
    # 연·월은 실측이 있는 기간(2024-12 ~ 기준 시점의 월)만 노출 — 연도가 바뀌는 12월 포함
    months = [f"{y:04d}-{m:02d}" for y, m in available_months()]
    current = f"{ss['selected_year']:04d}-{ss['selected_month']:02d}"
    if current not in months:
        current = f"{TODAY.year:04d}-{TODAY.month:02d}"
        ss["selected_year"], ss["selected_month"] = TODAY.year, TODAY.month
    ss["w_month"] = current
    st.selectbox("월 선택", months, key="w_month",
                 format_func=lambda v: f"{int(v[:4])}년 {int(v[5:])}월",
                 label_visibility="collapsed", on_change=_on_month_change)

    # 일 선택 — 1일~말일 전체 노출(데이터가 없는 날은 화면에서 '데이터 없음'으로 표시)
    days = month_days(ss["selected_year"], ss["selected_month"])
    if ss["selected_day"] not in days:
        ss["selected_day"] = days[-1]
    ss["w_day"] = ss["selected_day"]
    st.selectbox("일 선택", days, key="w_day", format_func=lambda d: f"{d}일",
                 label_visibility="collapsed", on_change=_on_day_change)

    # 선택 날짜 기준 설비 이상 판정
    sel_date = date(ss["selected_year"], ss["selected_month"], ss["selected_day"])
    anomalies = analyze_equipment_anomalies(sel_date, ss["thresholds"])
    has_status = bool(anomalies)  # 센서 실측이 있는 날만 상태 판정 가능
    if not has_status:
        st.caption(NO_DATA_TEXT)

    # 설비 목록 — 상태 원(실제 이상 반영) + 클릭 시 전역 선택 + 이상탐지 탭 전환
    for name in EQUIPMENT_NAMES:
        flags = anomalies.get(name, {"any": False})
        is_bad = flags.get("any", False)
        tooltip = anomaly_tooltip(flags) if has_status else NO_DATA_TEXT
        selected = ss["selected_equipment"] == name
        c_btn, c_dot = st.columns([5, 1], vertical_alignment="center")
        with c_btn:
            if st.button(name, key=f"equip_{name}",
                         use_container_width=True,
                         type="primary" if selected else "secondary",
                         help=tooltip):
                ss["selected_equipment"] = name   # 전역 공유 상태
                ss["_pending_tab"] = "이상탐지"     # 다음 런에서 탭 라디오에 반영
                # 위젯 key가 남아있으면 index와 충돌하므로 제거 후 재생성되게 함
                ss.pop("w_active_tab", None)
                st.rerun()
        with c_dot:
            dot = ("bad" if is_bad else "ok") if has_status else "none"
            st.markdown(
                f'<div style="padding-top:8px;" title="{tooltip}">'
                f'<span class="mg-dot {dot}"></span></div>',
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# ③ 탭 · 테마 토글  +  ④ 본문
# -----------------------------------------------------------------------------
def _delta_html(pct: float | None) -> str:
    """전월 대비: 증가=빨강↑, 감소=초록↓. 비교 자료가 없으면 회색 안내."""
    if pct is None:
        return ('<span style="color:#9ca3af; font-size:12px;">'
                '전월 대비 · 비교 자료 없음</span>')
    up = pct >= 0
    color = SEMANTIC_COLORS["up"] if up else SEMANTIC_COLORS["down"]
    arrow = "▲" if up else "▼"
    return (f'<span style="color:{color}; font-size:12px; font-weight:700;">'
            f'{arrow} {abs(pct):.1f}% 전월 대비</span>')


def _kpi_card(title: str, big: str, delta_html: str, sub: str) -> str:
    return (
        f'<div style="border:1px solid var(--mg-border,#e5e7eb); border-radius:8px;'
        f' padding:10px 12px; height:100%; background:var(--mg-panel,#fff);">'
        f'<div style="font-size:12px; color:var(--mg-text-muted,#6b7280);">{title}</div>'
        f'<div style="font-size:22px; font-weight:800; margin-top:2px;'
        f' color:var(--mg-text,#111);">{big}</div>'
        f'<div style="margin-top:2px;">{delta_html}</div>'
        f'<div style="font-size:12px; color:var(--mg-text-muted,#6b7280); margin-top:4px;">'
        f'{sub}</div>'
        f'</div>'
    )


def render_status_forecast_tab() -> None:
    import plotly.graph_objects as go

    ss = st.session_state
    theme = ss["theme"]
    year, month = ss["selected_year"], ss["selected_month"]
    sel_date = date(year, month, ss["selected_day"])

    k = monthly_kpis(year, month, sel_date, ss["thresholds"])

    def fmt(fn, v):
        return fn(v) if v is not None else "자료 없음"

    day_label = f"오늘({NOW.hour:02d}시 기준)" if sel_date == TODAY else "선택일"
    # 지난 달은 실측 합계, 기준 시점의 달·이후 달은 실측 + AI 예측 기반 '예상'
    is_past_month = (year, month) < (TODAY.year, TODAY.month)
    month_title = f"{month}월" if is_past_month else "금월 예상"
    # 예측 화면의 제목 접두어: 월 그래프는 5월(기준 시점 이후 달), 일 단위 화면은 기준일 이후 날짜
    pre_month = "예상 " if is_forecast_month(year, month) else ""
    pre = "예상 " if pre_month or sel_date > TODAY else ""

    # (1) KPI 카드 4개 (가로 25%씩)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_kpi_card(
            f"{month_title} 전기 요금", fmt(won, k["month_bill"]),
            _delta_html(k["delta_pct"]), f"{day_label} {fmt(won, k['day_bill'])}"),
            unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi_card(
            f"{month_title} 전력 소비량", fmt(kwh, k["month_power"]),
            _delta_html(k["delta_pct"]), f"{day_label} {fmt(kwh, k['day_power'])}"),
            unsafe_allow_html=True)
    with c3:
        st.markdown(_kpi_card(
            f"{month_title} 탄소 배출량", fmt(kgco2, k["month_carbon"]),
            _delta_html(k["delta_pct"]), f"{day_label} {fmt(kgco2, k['day_carbon'])}"),
            unsafe_allow_html=True)
    with c4:
        model_name = k["model"] or "AI 모델"
        day_mae = (f"선택일 오차 {k['day_mae']:,.3f} kWh/시간" if k["day_mae"] is not None
                   else "선택일 실측·예측 비교 자료 없음")
        st.markdown(_kpi_card(
            f"AI 예측 오차 (MAE · {model_name})",
            f"{k['mae']:,.3f} kWh/시간" if k["mae"] is not None else "자료 없음",
            f'<span style="font-size:12px;color:#6b7280;">평가 구간 {k["model_period"] or "-"} 전체</span>',
            day_mae),
            unsafe_allow_html=True)

    _, fc_meta = load_model_forecast()
    if not fc_meta["available"]:
        st.warning(fc_meta["note"])


    st.write("")

    # (2) 꺾은선 그래프 3개 (실제=녹색 실선, 예측=파랑 점선, marker)
    hdf = hourly_power_series(sel_date)
    mdf = monthly_power_series(year, month)

    is_today = sel_date == TODAY

    def line_fig(title, x, actual, pred, unit, now_x=None):
        fig = go.Figure()
        if now_x is not None:  # 현재 시각 표시선
            fig.add_vline(x=now_x, line={"color": "#9ca3af", "dash": "dash", "width": 1},
                          annotation_text="현재", annotation_position="top left",
                          annotation_font_size=10)
        fig.add_trace(go.Scatter(x=x, y=actual, mode="lines+markers", name="실제",
                                 line={"color": SEMANTIC_COLORS["actual"]},
                                 connectgaps=False))
        fig.add_trace(go.Scatter(x=x, y=pred, mode="lines+markers",
                                 name=f"AI 예측({k['model']})" if k["model"] else "예측",
                                 line={"color": SEMANTIC_COLORS["forecast"], "dash": "dot"},
                                 connectgaps=False))
        lay = plotly_layout(theme)
        lay["margin"] = {"l": 44, "r": 8, "t": 34, "b": 26}
        lay["legend"] = {"orientation": "h", "x": 1, "xanchor": "right",
                         "y": 1.18, "yanchor": "top"}
        # 세로축 자동 확대 방지: 값 변동이 작으면(0.2% 수준) 자동 범위가 그 차이를
        # 화면 전체로 키워 실측·예측이 크게 벌어진 것처럼 보인다 → 평균의 ±5% 이상 범위로 고정
        vals = np.concatenate([np.asarray(actual, float), np.asarray(pred, float)])
        vals = vals[~np.isnan(vals)]
        if len(vals):
            center = float(vals.mean())
            lo = min(float(vals.min()), center * 0.95)
            hi = max(float(vals.max()), center * 1.05)
            lay["yaxis"] = {**lay.get("yaxis", {}), "range": [lo, hi]}
        else:
            fig.add_annotation(text=NO_DATA_TEXT, showarrow=False, xref="paper", yref="paper",
                               x=0.5, y=0.5, font={"size": 12, "color": "#9ca3af"})
        fig.update_layout(height=240, title=f"{title} ({unit})", **lay)
        return fig

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(line_fig(f"{pre}시간별 전력량", hdf["hour"],
                                 hdf["actual"], hdf["pred"], "kWh",
                                 now_x=NOW.hour if is_today else None),
                        use_container_width=True)
    with g2:
        st.plotly_chart(line_fig(f"{pre_month}금월 전력 추이", mdf["day"],
                                 mdf["power_actual"], mdf["power_pred"], "kWh"),
                        use_container_width=True)
    with g3:
        st.plotly_chart(line_fig(f"{pre_month}금월 탄소 배출량 추이", mdf["day"],
                                 mdf["carbon_actual"], mdf["carbon_pred"], "kgCO2"),
                        use_container_width=True)

    # (3) 하단: 좌 55% 설비 막대 / 우 45% 도넛 3개
    anomalies = analyze_equipment_anomalies(sel_date, ss["thresholds"])
    left, right = st.columns([55, 45])

    with left:
        st.markdown(f"**{pre}설비별 전력 사용 현황**")
        rank = equipment_power_ranking(sel_date)
        if rank.empty:
            st.info(NO_DATA_TEXT)
        else:
            _equipment_bar(theme, rank, anomalies)

    with right:
        st.markdown(f"**{pre}일일 이상 현황**")
        if not anomalies:
            st.info(NO_DATA_TEXT)
        else:
            n_phase = sum(1 for f in anomalies.values() if f["phase"])
            n_line = sum(1 for f in anomalies.values() if f["line"])
            n_pf = sum(1 for f in anomalies.values() if f["pf"])
            d1, d2, d3 = st.columns(3)
            for col, title, bad in [
                (d1, "상전압 이상(일일 기준)", n_phase),
                (d2, "선간전압 이상(일일 기준)", n_line),
                (d3, "역률 이상(일일 기준)", n_pf),
            ]:
                with col:
                    st.plotly_chart(_donut(theme, title, bad),
                                    use_container_width=True)


def _equipment_bar(theme: str, rank: pd.DataFrame, anomalies: dict) -> None:
    """설비별 전력 사용량 가로 막대(이상 설비는 빨강)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    colors = [SEMANTIC_COLORS["bad"] if anomalies.get(n, {}).get("any")
              else "#64748b" for n in rank["equipment"]]
    fig.add_trace(go.Bar(
        x=rank["kwh"], y=rank["equipment"], orientation="h",
        marker={"color": colors},
        text=[f"{v:,.1f} kWh ({r:.1f}%)" for v, r in zip(rank["kwh"], rank["ratio"])],
        textposition="auto",
    ))
    lay = plotly_layout(theme)
    lay["margin"] = {"l": 8, "r": 8, "t": 6, "b": 20}
    fig.update_layout(height=270, yaxis={"autorange": "reversed",
                      **lay["yaxis"]}, **{k2: v for k2, v in lay.items()
                                          if k2 != "yaxis"})
    st.plotly_chart(fig, use_container_width=True)


def _donut(theme: str, title: str, bad: int | None):
    """정상/이상 도넛. 합=13, 중앙에 이상 건수. 이상 0건이면 전체 초록 + '이상 없음'.
    bad=None이면 판정 자료 없음(회색)."""
    import plotly.graph_objects as go
    total = len(EQUIPMENT_NAMES)
    normal = total - (bad or 0)
    if bad is None:
        values, colors, center = [total], ["#9ca3af"], "자료 없음"
    elif bad == 0:
        values, colors, center = [total], [SEMANTIC_COLORS["ok"]], "이상 없음"
    else:
        values = [normal, bad]
        colors = [SEMANTIC_COLORS["ok"], SEMANTIC_COLORS["bad"]]
        center = f"{bad}<br>건 이상"
    fig = go.Figure(go.Pie(values=values, hole=0.6, marker={"colors": colors},
                           textinfo="none", sort=False, showlegend=False))
    lay = plotly_layout(theme)
    lay["margin"] = {"l": 4, "r": 4, "t": 26, "b": 4}
    fig.update_layout(height=150, title={"text": title, "font": {"size": 11}},
                      annotations=[{"text": center, "showarrow": False,
                                    "font": {"size": 12}}], **lay)
    return fig


def _daily_line(theme, title, hours, values, threshold, unit, below_is_bad=True):
    """일일 측정 꺾은선(녹색) + 임계 빨간 점선 + 임계 미만 붉은 띠 + 벗어난 점 강조."""
    import plotly.graph_objects as go
    vals = np.asarray(values, dtype=float)
    all_nan = np.all(np.isnan(vals))
    fig = go.Figure()
    if all_nan:
        fig.add_annotation(text="데이터 없음", showarrow=False,
                           font={"size": 12, "color": "#9ca3af"})
    else:
        # 측정선(녹색 통일)
        fig.add_trace(go.Scatter(x=hours, y=vals, mode="lines+markers", name="측정",
                                 line={"color": SEMANTIC_COLORS["actual"]},
                                 showlegend=False, connectgaps=False))
        # 임계 벗어난 점 강조(빨강)
        bad_mask = vals < threshold if below_is_bad else vals > threshold
        if np.any(bad_mask):
            fig.add_trace(go.Scatter(
                x=np.asarray(hours)[bad_mask], y=vals[bad_mask], mode="markers",
                marker={"color": SEMANTIC_COLORS["bad"], "size": 9},
                showlegend=False))
        # 임계 가이드선(빨간 점선)
        fig.add_hline(y=threshold, line={"color": SEMANTIC_COLORS["threshold"],
                      "dash": "dash", "width": 1.5})
        # 임계 미만 붉은 배경 띠
        ymin = np.nanmin(vals); ymax = np.nanmax(vals)
        lo = min(ymin, threshold) - (ymax - ymin) * 0.1 - 1
        fig.add_hrect(y0=lo, y1=threshold, fillcolor="rgba(220,38,38,0.08)",
                      line_width=0)
    lay = plotly_layout(theme)
    lay["margin"] = {"l": 36, "r": 6, "t": 26, "b": 20}
    fig.update_layout(height=165,
                      title={"text": f"{title}  <span style='font-size:9px;color:#dc2626'>"
                             f"{threshold:g}{unit} 미만 이상</span>", "font": {"size": 11}},
                      **lay)
    return fig


def _monthly_multi(theme, title, days, series_map, threshold, unit):
    """한달 멀티라인(KS 색상) + 임계 가이드선/붉은 띠 + 임계 밑도는 점 강조."""
    import plotly.graph_objects as go
    ks = ks_phase_colors(theme)
    fig = go.Figure()
    for name, vals in series_map.items():
        vals = np.asarray(vals, dtype=float)
        fig.add_trace(go.Scatter(x=days, y=vals, mode="lines+markers", name=name,
                                 line={"color": ks[name]}, connectgaps=False))
        bad = vals < threshold
        if np.any(bad):
            fig.add_trace(go.Scatter(x=np.asarray(days)[bad], y=vals[bad],
                          mode="markers", name=f"{name} 이상",
                          marker={"color": SEMANTIC_COLORS["bad"], "size": 10,
                                  "line": {"color": SEMANTIC_COLORS["bad"], "width": 2}},
                          showlegend=False))
    fig.add_hline(y=threshold, line={"color": SEMANTIC_COLORS["threshold"],
                  "dash": "longdash", "width": 2.5})
    all_vals = np.concatenate([np.asarray(v, float) for v in series_map.values()])
    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
    fig.add_hrect(y0=min(ymin, threshold) - 1, y1=threshold,
                  fillcolor="rgba(220,38,38,0.08)", line_width=0)
    lay = plotly_layout(theme)
    lay["margin"] = {"l": 40, "r": 8, "t": 30, "b": 24}
    lay["legend"] = {"orientation": "h", "y": 1.25, "x": 0}
    fig.update_layout(height=180,
                      title={"text": f"{title}  <span style='font-size:9px;color:#dc2626'>"
                             f"{threshold:g}{unit} 미만 이상 · 그 날의 최저값</span>",
                             "font": {"size": 11}},
                      **lay)
    return fig


def render_anomaly_tab() -> None:
    ss = st.session_state
    theme = ss["theme"]
    year, month = ss["selected_year"], ss["selected_month"]
    sel_date = date(year, month, ss["selected_day"])
    th = ss["thresholds"]
    anomalies = analyze_equipment_anomalies(sel_date, th)

    # 상단: 설비 선택 버튼 13개 + 일일/한달 토글
    top_l, top_r = st.columns([6, 1.4])
    with top_l:
        cols = st.columns(7)
        for i, name in enumerate(EQUIPMENT_NAMES):
            with cols[i % 7]:
                bad = anomalies.get(name, {}).get("any", False)
                sel = ss["selected_equipment"] == name
                dot = " 🔴" if bad else ""
                if st.button(f"{name}{dot}", key=f"anom_{name}",
                             use_container_width=True,
                             type="primary" if sel else "secondary"):
                    # 전역 공유(페이지 유지). 버튼 클릭이 자연히 다음 런을 유발하므로
                    # 별도 st.rerun() 없이도 즉시 반영된다.
                    ss["selected_equipment"] = name
    with top_r:
        ss["w_anomaly_view"] = "일일 기준" if ss["anomaly_view"] == "daily" else "한달 기준"
        st.radio("기준", ["일일 기준", "한달 기준"], key="w_anomaly_view",
                 horizontal=True, label_visibility="collapsed",
                 on_change=_on_anomaly_view_change)

    equip = ss["selected_equipment"]
    daily, source = equipment_daily(sel_date)
    g = daily[daily["equipment"] == equip]
    hours = g["hour"].tolist() if len(g) else list(range(24))

    if ss["anomaly_view"] == "daily" and source != "sensor":
        # 전압·역률은 센서 실측으로만 확인 가능(예측일·자료 없는 날은 표시하지 않음)
        st.info(NO_DATA_TEXT)
    elif ss["anomaly_view"] == "daily":
        # 3행 3열: 상전압 R/S/T, 선간전압 RS/ST/TR, 역률 R/S/T
        ph = th["phase_voltage"]; ln = th["line_voltage"]; pf = th["power_factor"]
        rows = [
            [("일일 R선 상전압", "voltageR", ph, "V"),
             ("일일 S선 상전압", "voltageS", ph, "V"),
             ("일일 T선 상전압", "voltageT", ph, "V")],
            [("일일 RS 선간전압", "voltageRS", ln, "V"),
             ("일일 ST 선간전압", "voltageST", ln, "V"),
             ("일일 TR 선간전압", "voltageTR", ln, "V")],
            [("일일 R 역률", "powerFactorR", pf, "%"),
             ("일일 S 역률", "powerFactorS", pf, "%"),
             ("일일 T 역률", "powerFactorT", pf, "%")],
        ]
        for row in rows:
            cols = st.columns(3)
            for col, (title, field, thr, unit) in zip(cols, row):
                vals = g[field].tolist() if len(g) else [np.nan] * len(hours)
                with col:
                    st.plotly_chart(_daily_line(theme, title, hours, vals, thr, unit),
                                    use_container_width=True)
    else:
        # 한달: 상전압/선간전압/역률 세로 3개 (KS 색상)
        mdf = equipment_monthly(equip, year, month)
        if mdf.empty:
            st.info("해당 월은 데이터가 없습니다.")
        else:
            d = mdf["day"]
            st.plotly_chart(_monthly_multi(theme, "한달 상전압", d,
                            {"R": mdf["R"], "S": mdf["S"], "T": mdf["T"]},
                            th["phase_voltage"], "V"), use_container_width=True)
            st.plotly_chart(_monthly_multi(theme, "한달 선간전압", d,
                            {"RS": mdf["RS"], "ST": mdf["ST"], "TR": mdf["TR"]},
                            th["line_voltage"], "V"), use_container_width=True)
            st.plotly_chart(_monthly_multi(theme, "한달 역률", d,
                            {"R": mdf["pfR"], "S": mdf["pfS"], "T": mdf["pfT"]},
                            th["power_factor"], "%"), use_container_width=True)


def render_agent_tab() -> None:
    """에이전트 탭: agent/ 패키지의 LLM 에이전트(agent/agent.md 명세, agent/agentSystem.md 프롬프트).
    - 데이터: 대시보드와 같은 캐시(12~4월 설비별 실측 + AI 판정, 4/30~5월 AI 예측)
    - 상태 공유: 사이드바 조회 날짜(view_date), 선택 설비, 임계치(이상탐지 가이드선), 이상현황 보기 기준
    - OPENAI_API_KEY(.streamlit/secrets.toml 또는 환경변수)가 없으면 데모 모드
    """
    ss = st.session_state
    # 화면 상태와 에이전트가 아는 상태를 항상 같게 맞춘다(사이드바 선택이 기준)
    ss["view_date"] = date(ss["selected_year"], ss["selected_month"], ss["selected_day"]).isoformat()
    ss["view_range"] = ss.get("anomaly_view", "daily")
    if not EQUIPMENT_HOURLY_PATH.is_file():
        st.info("설비 실측 캐시가 없습니다. `python tools/prepare_dashboard_cache.py` 실행 후 앱을 다시 시작하세요.")
    try:
        from agent.agent_ui import render_agent_tab as render_new_agent_tab
    except Exception as exc:  # 의존성 누락 등 — 대시보드는 계속 동작
        st.error("에이전트 모듈을 불러오지 못했습니다. `pip install openai openpyxl python-docx==1.1.2` 후 다시 실행하세요.")
        st.caption(f"{type(exc).__name__}: {exc}")
        return
    render_new_agent_tab(ss)
    # 에이전트가 고른 설비는 대시보드 표기로 유지(이상탐지 탭과 공유)
    ss["selected_equipment"] = to_display_equipment(ss.get("selected_equipment"))


TAB_RENDERERS = {
    "현황·예측": render_status_forecast_tab,
    "이상탐지": render_anomaly_tab,
    "에이전트": render_agent_tab,
}


def _on_month_change() -> None:
    value = st.session_state["w_month"]  # "YYYY-MM"
    st.session_state["selected_year"] = int(value[:4])
    st.session_state["selected_month"] = int(value[5:])


def _on_day_change() -> None:
    st.session_state["selected_day"] = st.session_state["w_day"]


def _on_tab_change() -> None:
    st.session_state["active_tab"] = st.session_state["w_active_tab"]


def _on_theme_change() -> None:
    st.session_state["theme"] = st.session_state["w_theme"]
    st.session_state["_theme_from_os_applied"] = True


def _on_anomaly_view_change() -> None:
    v = st.session_state["w_anomaly_view"]
    st.session_state["anomaly_view"] = "daily" if v == "일일 기준" else "monthly"


def render_tabs_and_body() -> None:
    ss = st.session_state
    # 사이드바 등에서 예약한 탭 전환을 위젯 생성 전에 반영
    if ss.get("_pending_tab"):
        ss["active_tab"] = ss.pop("_pending_tab")

    # 위젯 key를 현재 상태와 동기화(위젯 생성 전에만 세팅 가능)
    ss["w_active_tab"] = ss["active_tab"]
    ss["w_theme"] = ss["theme"]

    top_l, top_r = st.columns([5, 1], vertical_alignment="center")
    with top_l:
        # key로 세션에 자동 바인딩 → st.rerun 없이 즉시 전환
        st.radio("페이지", TABS, key="w_active_tab",
                 horizontal=True, label_visibility="collapsed",
                 on_change=_on_tab_change)
    with top_r:
        st.radio("테마", ["light", "dark"], key="w_theme",
                 horizontal=True, label_visibility="collapsed",
                 on_change=_on_theme_change)

    TAB_RENDERERS[ss["active_tab"]]()


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------
def main() -> None:
    init_state()
    theme = resolve_initial_theme()
    theme_css(theme)

    _, gt_source = load_ground_truth_may()
    _ = load_rtu_raw_source()

    render_header()

    # 사이드바(②영역)는 st.sidebar로 분리 → 본문에서 컬럼 중첩 제한(1단계)을 회피
    with st.sidebar:
        render_sidebar(gt_source)

    # 본문(③④영역)은 최상위에 두어 탭 내부에서 컬럼을 자유롭게 중첩
    render_tabs_and_body()


if __name__ == "__main__":
    main()
