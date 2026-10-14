# -*- coding: utf-8 -*-
# =============================================================================
# 마가렛트 — AI 기반 제조 현장 전력량·탄소 배출량 예측 및 모니터링 대시보드
# -----------------------------------------------------------------------------
# 실행 방법:
#   1) 의존성 설치 (버전 고정)
#        pip install streamlit==1.39.0 plotly==5.24.1 pandas==2.2.3
#   2) 앱 실행
#        streamlit run dashboard_app.py
#
# 주의:
#   - 원본 데이터 파일(data/rtu_ground_truth_may.csv, rtu_data_full_flagged.csv 등)은
#     읽기 전용으로만 접근하며 절대 수정/덮어쓰기하지 않습니다.
#   - 명세 경로의 데이터가 없으면 같은 구조의 더미 데이터를 생성해 화면을 채웁니다.
#
# 구현 범위(현재): Task 1~3
#   - 4영역 고정 레이아웃(①헤더 ②필터·설비목록 ③탭·테마 ④본문)
#   - 데이터 로딩/더미 생성 계층, 파생 계산(kWh/요금/탄소)
#   - 전역 상태(st.session_state) 초기화
#   각 탭 본문은 플레이스홀더이며 이후 Task에서 채웁니다.
#
# 기준 해상도: 1366 x 768 (노트북). 세로 스크롤 없이 한 화면.
# 고정 기준일("오늘"): 2025-05-31
# =============================================================================

from __future__ import annotations

import calendar
from datetime import date
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

TODAY = date(2025, 5, 31)  # 고정 기준일

# 요금/탄소 환산 계수 (명세)
WON_PER_KWH = 180.0
KGCO2_PER_KWH = 0.424

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

# 점검 보고서 양식 (에이전트 페이지). source: 'auto'=데이터 자동 채움 / 'manual'=사람 입력
REPORT_FORM_NAME = "제조현장 안전 점검 보고서"
REPORT_FORM = [
    ("점검 일시", "manual", True),
    ("점검 구분", "manual", True),
    ("점검자", "manual", True),
    ("점검 대상 설비", "auto", True),
    ("설비 위치", "manual", False),
    ("정격 용량", "auto", False),
    ("R상 전압", "auto", True),
    ("S상 전압", "auto", True),
    ("T상 전압", "auto", True),
    ("부적합 건수", "auto", True),
    ("전력 소비량", "auto", True),
    ("조치 내용", "manual", False),
]

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
    컬럼: id(datetime), hourly_pow, may_bill, may_carbon, agg_pow
    """
    path = _first_existing(DATA_CANDIDATES["ground_truth_may"])
    if path is not None:
        df = pd.read_csv(path)  # 읽기 전용
        df["id"] = pd.to_datetime(df["id"])
        return df, f"file:{path.name}"
    return _dummy_ground_truth_may(), "dummy"


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


@st.cache_data(show_spinner=False)
def make_dummy_equipment_daily(day: date) -> pd.DataFrame:
    """설비 13개의 하루치(0~24시, 1시간 단위) 전압/역률/전력 더미 데이터.
    명세 구조(전압 R/S/T, 선간 RS/ST/TR, 역률 R/S/T, 전력)를 따른다.
    실제 원시 CSV가 있으면 이후 Task에서 이 함수를 실측 집계로 교체한다.
    """
    seed = int(day.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    hours = np.arange(0, 25)
    rows = []
    for idx_eq, (ch, name) in enumerate(EQUIPMENT_LIST):
        # 설비마다 기준 전력 수준을 다르게
        base_kw = rng.uniform(40, 320)
        # 날짜+설비에 따라 결정적으로 이상 항목/시각을 주입(예시 화면과 유사하게)
        eq_seed = (seed * 131 + idx_eq * 17) % 1000
        inject_phase = (eq_seed % 7) == 0      # 상전압 이상 설비
        inject_line = (eq_seed % 11) == 0      # 선간전압 이상 설비
        inject_pf = (eq_seed % 5) == 0         # 역률 이상 설비
        bad_hours = {(eq_seed + k) % 24 for k in range(2)}  # 이상 발생 시각들
        for h in hours:
            load = max(0.2, np.sin((h - 6) / 24 * 2 * np.pi) * 0.5 + 0.7)
            active_kw = base_kw * load + rng.normal(0, base_kw * 0.03)
            vR = 220 + rng.normal(0, 3)
            vS = 220 + rng.normal(0, 3)
            vT = 220 + rng.normal(0, 3)
            vRS = 380 + rng.normal(0, 4)
            vST = 380 + rng.normal(0, 4)
            vTR = 380 + rng.normal(0, 4)
            pfR = 92 + rng.normal(0, 2)
            pfS = 92 + rng.normal(0, 2)
            pfT = 92 + rng.normal(0, 2)
            if h in bad_hours:
                if inject_phase:
                    vR = 204 + rng.normal(0, 1)   # 210V 미만 → 상전압 이상
                if inject_line:
                    vRS = 354 + rng.normal(0, 1)  # 360V 미만 → 선간전압 이상
                if inject_pf:
                    pfR = 80 + rng.normal(0, 1)   # 85% 미만 → 역률 이상
            rows.append(
                {
                    "channel": ch,
                    "equipment": name,
                    "hour": int(h),
                    "voltageR": vR,
                    "voltageS": vS,
                    "voltageT": vT,
                    "voltageRS": vRS,
                    "voltageST": vST,
                    "voltageTR": vTR,
                    "powerFactorR": pfR,
                    "powerFactorS": pfS,
                    "powerFactorT": pfT,
                    "activePower": max(0.0, active_kw),
                }
            )
    return pd.DataFrame(rows)


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
@st.cache_data(show_spinner=False)
def forecast_series(_actual: pd.Series, horizon_hours: int, seed: int = 7) -> np.ndarray:
    """실측 시계열을 바탕으로 한 간이 예측(명세의 LSTM/XGBoost/Prophet/SARIMA 중
    MAE 최적 모델을 대체하는 베이스라인). 실데이터 연동 시 모델 산출로 교체한다.
    - 최근 24시간 평균 + 약한 잡음으로 1시간 단위 예측값 생성.
    """
    rng = np.random.default_rng(seed)
    base = float(_actual.tail(24).mean()) if len(_actual) else 0.0
    hours = np.arange(horizon_hours)
    daily = np.sin((hours % 24 - 6) / 24 * 2 * np.pi).clip(min=-0.6)
    return base * (0.85 + 0.25 * (daily + 0.6)) + rng.normal(0, base * 0.02, horizon_hours)


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    n = min(len(actual), len(pred))
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(np.asarray(actual[:n]) - np.asarray(pred[:n]))))


@st.cache_data(show_spinner=False)
def monthly_power_series(year: int, month: int) -> pd.DataFrame:
    """월별(1~말일) 일 단위 전력량/탄소 실제·예측 추이.
    실측(ground truth)의 시간당 전력량을 일 합계로 집계하고, 예측은 간이 모델로 생성.
    """
    gt, _ = load_ground_truth_may()
    g = gt[(gt["id"].dt.year == year) & (gt["id"].dt.month == month)].copy()
    g["day"] = g["id"].dt.day
    daily = g.groupby("day")["hourly_pow"].sum()
    days = list(range(1, calendar.monthrange(year, month)[1] + 1))
    actual = np.array([daily.get(d, np.nan) for d in days], dtype=float)
    # 예측: 실제의 이동평균 기반(없는 날은 NaN 유지 → 선 끊김)
    valid = actual[~np.isnan(actual)]
    base = float(valid.mean()) if len(valid) else 0.0
    rng = np.random.default_rng(month)
    pred = np.where(np.isnan(actual), np.nan,
                    actual * 0.98 + rng.normal(0, base * 0.02, len(actual)))
    return pd.DataFrame({
        "day": days,
        "power_actual": actual,
        "power_pred": pred,
        "carbon_actual": actual * KGCO2_PER_KWH,
        "carbon_pred": pred * KGCO2_PER_KWH,
    })


@st.cache_data(show_spinner=False)
def hourly_power_series(day: date) -> pd.DataFrame:
    """선택일의 시간별(0~24시) 전력량 실제·예측.
    실측 ground truth에서 해당 날짜의 시간당 전력량을 사용하고,
    예측은 간이 모델. 데이터 없으면 NaN(선 끊김)."""
    gt, _ = load_ground_truth_may()
    d = gt[gt["id"].dt.date == day].copy()
    hours = list(range(0, 24))
    if len(d):
        d["hour"] = d["id"].dt.hour
        s = d.groupby("hour")["hourly_pow"].mean()
        actual = np.array([s.get(h, np.nan) for h in hours], dtype=float)
    else:
        actual = np.full(len(hours), np.nan)
    base = np.nanmean(actual) if not np.all(np.isnan(actual)) else 0.0
    rng = np.random.default_rng(day.day)
    pred = actual * 0.97 + rng.normal(0, (base or 1) * 0.02, len(actual))
    return pd.DataFrame({"hour": hours, "actual": actual, "pred": pred})


@st.cache_data(show_spinner=False)
def make_dummy_equipment_monthly(equipment: str, year: int, month: int) -> pd.DataFrame:
    """특정 설비의 월별(1~말일) 상별 '그 날 최저값' 더미 데이터.
    상전압 R/S/T, 선간전압 RS/ST/TR, 역률 R/S/T. (한달 기준 그래프용)
    """
    total_days = calendar.monthrange(year, month)[1]
    rows = []
    for d in range(1, total_days + 1):
        day = date(year, month, d)
        g = make_dummy_equipment_daily(day)
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
    """설비별 하루 전력 소비량(kWh) 내림차순 + 비율(%). (더미 하루치 기반)"""
    df = make_dummy_equipment_daily(day)
    rows = []
    for name, g in df.groupby("equipment", sort=False):
        # 시간당 평균 kW를 하루(24h)로 환산한 간이 소비량
        day_kwh = float(g["activePower"].mean() * 24.0)
        rows.append({"equipment": name, "kwh": day_kwh})
    out = pd.DataFrame(rows).sort_values("kwh", ascending=False).reset_index(drop=True)
    total = out["kwh"].sum() or 1.0
    out["ratio"] = out["kwh"] / total * 100.0
    out["rank"] = out.index + 1
    return out


def monthly_kpis(year: int, month: int, day: date, thresholds: dict) -> dict:
    """현황·예측 KPI: 금월 예상(요금/전력/탄소), 전월 대비(%), 일일 누적, MAE."""
    mdf = monthly_power_series(year, month)
    month_power = float(np.nansum(mdf["power_actual"]))
    # 금월 '예상' = 실측 합 + 남은 일자 평균 추정(간이)
    valid = mdf["power_actual"].dropna()
    avg_day = float(valid.mean()) if len(valid) else 0.0
    total_days = calendar.monthrange(year, month)[1]
    month_power_est = month_power + avg_day * (total_days - len(valid))

    # 전월(간이): 금월 추정의 92.4% 가정 → +8.2% 증가 패턴(예시 화면과 유사)
    prev_power = month_power_est / 1.082
    delta_pct = (month_power_est - prev_power) / (prev_power or 1) * 100.0

    # 일일 누적(선택일)
    hdf = hourly_power_series(day)
    day_power = float(np.nansum(hdf["actual"]))

    # MAE (시간당 kWh)
    pair = hdf.dropna()
    day_mae = mae(pair["actual"].to_numpy(), pair["pred"].to_numpy())

    return {
        "month_bill": bill_from_kwh(month_power_est),
        "month_power": month_power_est,
        "month_carbon": carbon_from_kwh(month_power_est),
        "delta_pct": delta_pct,
        "day_bill": bill_from_kwh(day_power),
        "day_power": day_power,
        "day_carbon": carbon_from_kwh(day_power),
        "mae": day_mae,
    }


# -----------------------------------------------------------------------------
# 이상 판정 (Task 6 상태 원 / Task 10 그래프 공용)
# -----------------------------------------------------------------------------
def analyze_equipment_anomalies(day: date, thresholds: dict) -> dict[str, dict]:
    """선택 날짜의 설비별 이상 여부 판정.
    반환: {설비명: {"phase": bool, "line": bool, "pf": bool, "any": bool}}
    - phase: 상전압 R/S/T 중 하나라도 임계 미만
    - line : 선간전압 RS/ST/TR 중 하나라도 임계 미만
    - pf   : 역률 R/S/T 중 하나라도 임계 미만
    """
    df = make_dummy_equipment_daily(day)
    ph_th = thresholds["phase_voltage"]
    ln_th = thresholds["line_voltage"]
    pf_th = thresholds["power_factor"]
    result: dict[str, dict] = {}
    for name, g in df.groupby("equipment", sort=False):
        phase = bool((g[["voltageR", "voltageS", "voltageT"]] < ph_th).any().any())
        line = bool((g[["voltageRS", "voltageST", "voltageTR"]] < ln_th).any().any())
        pf = bool((g[["powerFactorR", "powerFactorS", "powerFactorT"]] < pf_th).any().any())
        result[name] = {"phase": phase, "line": line, "pf": pf,
                        "any": phase or line or pf}
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
    return " / ".join(items) if items else "이상 없음"


def data_available_days(year: int, month: int) -> list[int]:
    """해당 월에 '데이터가 있는' 일자 목록.
    실측(ground truth)은 2025-05-10~31만 존재하므로 그 범위로 제한한다.
    그 외 월은 (더미가 매일 생성되므로) 해당 월 전체 일자를 반환한다.
    """
    gt, _ = load_ground_truth_may()
    gt_days = sorted({d.day for d in gt["id"] if d.year == year and d.month == month})
    if gt_days:
        return gt_days
    return list(range(1, calendar.monthrange(year, month)[1] + 1))


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
    ss.setdefault("thresholds", dict(DEFAULT_THRESHOLDS))
    ss.setdefault("chat_history", [])
    ss.setdefault("report_values", {})
    ss.setdefault("threshold_history", [])


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
            f'<div class="mg-maker">제작<br><b>{MAKERS}</b></div>',
            unsafe_allow_html=True,
        )
    st.divider()


# -----------------------------------------------------------------------------
# ② 필터 · 설비 목록 (좌측)
# -----------------------------------------------------------------------------
def render_sidebar(gt_source: str) -> None:
    ss = st.session_state
    st.markdown("**조회 기간**")

    # 월 선택
    months = [f"{ss['selected_year']}년 {m}월" for m in range(1, 13)]
    sel_m = st.selectbox("월 선택", months, index=ss["selected_month"] - 1,
                         label_visibility="collapsed")
    new_month = months.index(sel_m) + 1
    if new_month != ss["selected_month"]:
        ss["selected_month"] = new_month  # 월만 바꾸면 월별 수치/그래프 갱신(Task 8/10)

    # 일 선택 — 데이터 있는 일자만 노출(없는 날은 선택 불가)
    days = data_available_days(ss["selected_year"], ss["selected_month"])
    if ss["selected_day"] not in days:
        ss["selected_day"] = days[-1]
    sel_d = st.selectbox("일 선택", [f"{d}일" for d in days],
                         index=days.index(ss["selected_day"]),
                         label_visibility="collapsed")
    ss["selected_day"] = int(sel_d.replace("일", ""))

    # 선택 날짜 기준 설비 이상 판정
    sel_date = date(ss["selected_year"], ss["selected_month"], ss["selected_day"])
    anomalies = analyze_equipment_anomalies(sel_date, ss["thresholds"])
    bad_count = sum(1 for f in anomalies.values() if f["any"])
    st.caption(f"설비 현황 {len(EQUIPMENT_NAMES)} · 이상 {bad_count} · 출처: {gt_source}")

    # 설비 목록 — 상태 원(실제 이상 반영) + 클릭 시 전역 선택 + 이상탐지 탭 전환
    for name in EQUIPMENT_NAMES:
        flags = anomalies.get(name, {"any": False})
        is_bad = flags.get("any", False)
        selected = ss["selected_equipment"] == name
        c_btn, c_dot = st.columns([5, 1], vertical_alignment="center")
        with c_btn:
            if st.button(name, key=f"equip_{name}",
                         use_container_width=True,
                         type="primary" if selected else "secondary",
                         help=anomaly_tooltip(flags)):
                ss["selected_equipment"] = name   # 전역 공유 상태
                ss["_pending_tab"] = "이상탐지"     # 다음 런에서 탭 라디오에 반영
                # 위젯 key가 남아있으면 index와 충돌하므로 제거 후 재생성되게 함
                ss.pop("w_active_tab", None)
                st.rerun()
        with c_dot:
            dot = "bad" if is_bad else "ok"
            st.markdown(
                f'<div style="padding-top:8px;" title="{anomaly_tooltip(flags)}">'
                f'<span class="mg-dot {dot}"></span></div>',
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# ③ 탭 · 테마 토글  +  ④ 본문
# -----------------------------------------------------------------------------
def _delta_html(pct: float) -> str:
    """전월 대비: 증가=빨강↑, 감소=초록↓."""
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

    # (1) KPI 카드 4개 (가로 25%씩)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_kpi_card(
            "금월 예상 전기 요금", won(k["month_bill"]),
            _delta_html(k["delta_pct"]), f"오늘 {won(k['day_bill'])}"),
            unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi_card(
            "금월 예상 전력 소비량", kwh(k["month_power"]),
            _delta_html(k["delta_pct"]), f"오늘 {kwh(k['day_power'])}"),
            unsafe_allow_html=True)
    with c3:
        st.markdown(_kpi_card(
            "금월 예상 탄소 배출량", kgco2(k["month_carbon"]),
            _delta_html(k["delta_pct"]), f"오늘 {kgco2(k['day_carbon'])}"),
            unsafe_allow_html=True)
    with c4:
        st.markdown(_kpi_card(
            "AI 예측 오차 범위 (MAE)", f"{k['mae']:,.1f} kWh/시간",
            '<span style="font-size:12px;color:#6b7280;">예측과 실제의 시간당 차이</span>',
            f"예측이 실제와 시간당 평균 {k['mae']:,.1f} kWh 차이"),
            unsafe_allow_html=True)

    st.write("")

    # (2) 꺾은선 그래프 3개 (실제=녹색 실선, 예측=파랑 점선, marker)
    hdf = hourly_power_series(sel_date)
    mdf = monthly_power_series(year, month)

    # 선택일에 실측 시간이 거의 없으면(예: 2025-05-31은 00시 1개만 존재) 안내
    valid_hours = int((~np.isnan(hdf["actual"])).sum())
    if valid_hours <= 1:
        st.caption(f"※ 선택일({sel_date:%Y-%m-%d})은 실측이 {valid_hours}시간만 존재해 "
                   "시간별 그래프가 비어 보일 수 있습니다. 월별 그래프/설비 현황은 정상 표시됩니다.")

    def line_fig(title, x, actual, pred, unit):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=actual, mode="lines+markers", name="실제",
                                 line={"color": SEMANTIC_COLORS["actual"]},
                                 connectgaps=False))
        fig.add_trace(go.Scatter(x=x, y=pred, mode="lines+markers", name="예측",
                                 line={"color": SEMANTIC_COLORS["forecast"], "dash": "dot"},
                                 connectgaps=False))
        lay = plotly_layout(theme)
        lay["margin"] = {"l": 44, "r": 8, "t": 34, "b": 26}
        lay["legend"] = {"orientation": "h", "x": 1, "xanchor": "right",
                         "y": 1.18, "yanchor": "top"}
        fig.update_layout(height=240, title=f"{title} ({unit})", **lay)
        return fig

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(line_fig("시간별 전력량", hdf["hour"],
                                 hdf["actual"], hdf["pred"], "kWh"),
                        use_container_width=True)
    with g2:
        st.plotly_chart(line_fig("월별 전력 추이", mdf["day"],
                                 mdf["power_actual"], mdf["power_pred"], "kWh"),
                        use_container_width=True)
    with g3:
        st.plotly_chart(line_fig("월별 탄소 배출량 추이", mdf["day"],
                                 mdf["carbon_actual"], mdf["carbon_pred"], "kgCO2"),
                        use_container_width=True)

    # (3) 하단: 좌 55% 설비 막대 / 우 45% 도넛 3개
    anomalies = analyze_equipment_anomalies(sel_date, ss["thresholds"])
    left, right = st.columns([55, 45])

    with left:
        st.markdown("**설비별 전력 사용 현황**")
        rank = equipment_power_ranking(sel_date)
        fig = go.Figure()
        colors = [SEMANTIC_COLORS["bad"] if anomalies.get(n, {}).get("any")
                  else "#64748b" for n in rank["equipment"]]
        fig.add_trace(go.Bar(
            x=rank["kwh"], y=rank["equipment"], orientation="h",
            marker={"color": colors},
            text=[f"{v:,.0f} kWh ({r:.1f}%)" for v, r in zip(rank["kwh"], rank["ratio"])],
            textposition="auto",
        ))
        lay = plotly_layout(theme)
        lay["margin"] = {"l": 8, "r": 8, "t": 6, "b": 20}
        fig.update_layout(height=270, yaxis={"autorange": "reversed",
                          **lay["yaxis"]}, **{k2: v for k2, v in lay.items()
                                              if k2 != "yaxis"})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**일일 이상 현황**")
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


def _donut(theme: str, title: str, bad: int):
    """정상/이상 도넛. 합=13, 중앙에 이상 건수. 이상 0건이면 전체 초록 + '이상 없음'."""
    import plotly.graph_objects as go
    total = len(EQUIPMENT_NAMES)
    normal = total - bad
    if bad == 0:
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
    daily = make_dummy_equipment_daily(sel_date)
    g = daily[daily["equipment"] == equip]
    hours = g["hour"].tolist() if len(g) else list(range(25))

    if ss["anomaly_view"] == "daily":
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
        mdf = make_dummy_equipment_monthly(equip, year, month)
        if mdf.empty:
            st.info("데이터 없음")
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


def _auto_report_values(equip: str, sel_date: date, th: dict) -> dict:
    """데이터에서 자동으로 채워지는 보고서 항목."""
    daily = make_dummy_equipment_daily(sel_date)
    g = daily[daily["equipment"] == equip]
    an = analyze_equipment_anomalies(sel_date, th).get(equip, {})
    bad = sum(1 for key in ("phase", "line", "pf") if an.get(key))
    day_kwh = float(g["activePower"].mean() * 24.0) if len(g) else 0.0
    return {
        "점검 대상 설비": equip,
        "정격 용량": "22 kW",
        "R상 전압": f"{g['voltageR'].min():.1f} V" if len(g) else "-",
        "S상 전압": f"{g['voltageS'].min():.1f} V" if len(g) else "-",
        "T상 전압": f"{g['voltageT'].min():.1f} V" if len(g) else "-",
        "부적합 건수": f"{bad} 건",
        "전력 소비량": kwh(day_kwh),
    }


def render_agent_tab() -> None:
    ss = st.session_state
    equip = ss["selected_equipment"]
    year, month = ss["selected_year"], ss["selected_month"]
    sel_date = date(year, month, ss["selected_day"])
    th = ss["thresholds"]

    auto_vals = _auto_report_values(equip, sel_date, th)
    # 보고서 값 = 자동 채움 + 사람이 입력한 값(session)
    report = dict(auto_vals)
    report.update(ss["report_values"])

    manual_items = [n for n, src, _ in REPORT_FORM if src == "manual"]
    required_items = [n for n, _, req in REPORT_FORM if req]
    done_manual = sum(1 for n in manual_items if ss["report_values"].get(n))
    total_manual = len(manual_items)

    left, right = st.columns([45, 55])

    # ---- 좌: 체크리스트 + 대화 ----
    with left:
        with st.expander(f"일일 점검 체크리스트  ·  {done_manual} / {total_manual} 완료",
                         expanded=False):
            for n in manual_items:
                v = ss["report_values"].get(n)
                if v:
                    st.markdown(f"- ✅ **{n}**: {v}")
                else:
                    st.markdown(f"- ⬜ {n}")

        st.markdown("**대화**")
        # 진입 시 오늘 이상 요약을 에이전트가 먼저 발화
        if not ss["chat_history"]:
            an = analyze_equipment_anomalies(sel_date, th)
            bad_n = sum(1 for f in an.values() if f["any"])
            ss["chat_history"].append(
                ("agent", f"오늘({sel_date:%Y-%m-%d}) 이상 {bad_n}건이 감지됐습니다. "
                          f"현재 선택된 설비는 '{equip}'입니다."))

        for role, msg in ss["chat_history"][-6:]:
            align = "right" if role == "user" else "left"
            bg = "#dcfce7" if role == "user" else "#f1f5f9"
            st.markdown(
                f'<div style="text-align:{align};"><span style="display:inline-block;'
                f'background:{bg}; color:#111; padding:6px 10px; border-radius:8px;'
                f' margin:2px 0; max-width:90%;">{msg}</span></div>',
                unsafe_allow_html=True)

        prompt = st.chat_input("메시지를 입력하세요 (예: 역률 기준 90%로 올려줘)")
        if prompt:
            ss["chat_history"].append(("user", prompt))
            reply = _agent_reply(prompt, sel_date)
            ss["chat_history"].append(("agent", reply))
            st.rerun()

        # 임계치 변경 이력 + 기본값 복원
        cur = ss["thresholds"]
        st.caption(
            f"임계치 — 상전압 {cur['phase_voltage']:g}V · 선간 {cur['line_voltage']:g}V · "
            f"역률 {cur['power_factor']:g}%")
        if ss.get("threshold_history"):
            with st.expander("임계치 변경 이력", expanded=False):
                for h in ss["threshold_history"][-5:]:
                    st.markdown(f"- {h}")
        if cur != DEFAULT_THRESHOLDS:
            if st.button("임계치 기본값으로 복원", use_container_width=True):
                ss["thresholds"] = dict(DEFAULT_THRESHOLDS)
                ss["threshold_history"].append("기본값으로 복원")
                st.rerun()

    # ---- 우: 보고서 미리보기 ----
    with right:
        head_l, head_r = st.columns([3, 1])
        with head_l:
            st.markdown(f"**{REPORT_FORM_NAME}**")
        with head_r:
            all_required_done = all(report.get(n) for n in required_items)
            st.button("내보내기", disabled=not all_required_done,
                      use_container_width=True)

        for name, src, req in REPORT_FORM:
            val = report.get(name)
            tag = "🔵자동" if src == "auto" else "✏️입력"
            if val:
                st.markdown(f"<div style='font-size:13px;'>{tag} <b>{name}</b> · {val}"
                            f"{' <span style=color:#dc2626>*</span>' if req else ''}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='font-size:13px; color:#9ca3af;'>{tag} {name} · "
                            f"(미입력){' *' if req else ''}</div>",
                            unsafe_allow_html=True)


def _agent_reply(prompt: str, sel_date: date) -> str:
    """대화 처리: 임계치 변경 요청이면 재분석 결과를, 아니면 조치 기록 안내."""
    ss = st.session_state
    th = ss["thresholds"]
    import re
    m = re.search(r"역률.*?(\d{2,3})\s*%", prompt)
    if m:
        new_pf = float(m.group(1))
        old = dict(th)
        before = analyze_equipment_anomalies(sel_date, th)
        before_n = sum(1 for f in before.values() if f["any"])
        before_set = {n for n, f in before.items() if f["any"]}
        th["power_factor"] = new_pf  # 이상탐지 가이드선에 반영
        after = analyze_equipment_anomalies(sel_date, th)
        after_n = sum(1 for f in after.values() if f["any"])
        new_included = sorted(set(n for n, f in after.items() if f["any"]) - before_set)
        ss.setdefault("threshold_history", []).append(
            f"역률 {old['power_factor']:g}% → {new_pf:g}% (사용자 · {sel_date:%Y-%m-%d})")
        inc = ", ".join(new_included) if new_included else "없음"
        return (f"역률 임계치 {old['power_factor']:g}% → {new_pf:g}% 재분석: "
                f"이상 {before_n}건 → {after_n}건. 새로 포함: {inc}. "
                f"(이상탐지 그래프 가이드 선에 반영됨)")
    # 조치 내용 기록(간이)
    ss["report_values"]["조치 내용"] = prompt
    return "조치 내용으로 기록했습니다. 보고서 '조치 내용' 항목에 반영했어요."


TAB_RENDERERS = {
    "현황·예측": render_status_forecast_tab,
    "이상탐지": render_anomaly_tab,
    "에이전트": render_agent_tab,
}


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

    gt_df, gt_source = load_ground_truth_may()
    _ = load_rtu_raw_source()

    render_header()

    # 사이드바(②영역)는 st.sidebar로 분리 → 본문에서 컬럼 중첩 제한(1단계)을 회피
    with st.sidebar:
        render_sidebar(gt_source)

    # 본문(③④영역)은 최상위에 두어 탭 내부에서 컬럼을 자유롭게 중첩
    render_tabs_and_body()

    st.caption(
        f"기준일(오늘): {TODAY:%Y-%m-%d}  ·  실측 데이터 행수: {len(gt_df):,}  ·  "
        "원본 데이터는 읽기 전용으로만 사용됩니다."
    )


if __name__ == "__main__":
    main()
