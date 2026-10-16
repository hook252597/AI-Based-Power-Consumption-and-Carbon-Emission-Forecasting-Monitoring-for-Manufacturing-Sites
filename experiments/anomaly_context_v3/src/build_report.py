"""Build an offline Korean review report from completed experiment outputs.

Run from any directory: python src/build_report.py
Only reads experiment outputs; never calls an API or modifies a dashboard/model.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["fixed_rule", "static_if", "robust_context", "context_if"]
NAMES = {"fixed_rule": "고정 규칙", "static_if": "현재 정보 AI", "robust_context": "시간 맥락 통계", "context_if": "시간 맥락 AI"}
COLORS = {"fixed_rule": "#94a3b8", "static_if": "#5267df", "robust_context": "#b7791f", "context_if": "#087f8c"}
KINDS = {"level_shift": "평소 수준의 급변", "volatility_increase": "흔들림 증가", "phase_imbalance": "상간 차이 증가", "relation_break": "전류·전력 관계 변화", "persistent_drift": "작은 변화의 누적"}
PERIODS = {"march_check": "3월 후반 점검", "april_development": "4월 개발 평가"}
SENSOR_NAMES = {"activePower": "유효전력", **{f"voltage{p}": f"{p}상 전압" for p in "RST"}, **{f"current{p}": f"{p}상 전류" for p in "RST"}, **{f"powerFactor{p}": f"{p}상 역률" for p in "RST"}}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def number(value, decimals=0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def percent(value, decimals=2) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value) * 100:,.{decimals}f}%"


def boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin(["true", "1"])


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def table(headers, rows, caption=None) -> str:
    head = "".join(f"<th scope='col'>{esc(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in rows)
    if not body:
        body = f"<tr><td colspan='{len(headers)}'>해당 결과가 없습니다.</td></tr>"
    caption_html = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f"<p class='scroll-hint'>표를 좌우로 넘기면 나머지 항목을 볼 수 있습니다.</p><div class='table-scroll'><table>{caption_html}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def aggregate_real(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"period", "method", "budget", "alert_count", "flagged_minutes", "persistent_flagged_minutes", "complete_minutes", "observed_equipment_hours"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing real-result columns: {sorted(required - set(frame.columns))}")
    aggregate = frame.groupby(["period", "method", "budget"], as_index=False)[
        ["alert_count", "flagged_minutes", "persistent_flagged_minutes", "complete_minutes", "observed_equipment_hours"]
    ].sum()
    denom = aggregate["complete_minutes"].replace(0, np.nan)
    aggregate["flagged_fraction"] = aggregate.flagged_minutes / denom
    aggregate["persistent_fraction"] = aggregate.persistent_flagged_minutes / denom
    aggregate["alerts_per_observed_day"] = aggregate.alert_count / (aggregate.observed_equipment_hours / 24).replace(0, np.nan)
    return aggregate


def aggregate_synthetic(frame: pd.DataFrame, by_kind=False) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    for col in ["detected_during_injection", "baseline_already_alerting", "incremental_detection_on_clean_control"]:
        work[col] = boolean(work[col])
    groups = ["method", "budget"] + (["kind"] if by_kind else [])
    rows = []
    for key, group in work.groupby(groups, sort=False):
        clean = ~group.baseline_already_alerting
        detected = group.detected_during_injection
        incremental = group.incremental_detection_on_clean_control
        if (incremental & (~detected | ~clean)).any():
            raise ValueError("Incremental detection must have a clean baseline and an injected alert")
        rows.append({**dict(zip(groups, key)), "tests": len(group), "detected": int(detected.sum()),
                     "baseline_alerts": int((~clean).sum()), "clean": int(clean.sum()),
                     "incremental": int(incremental.sum()), "rate": detected.mean(),
                     "incremental_rate": incremental.sum() / clean.sum() if clean.sum() else np.nan,
                     "delay": pd.to_numeric(group.loc[detected, "first_decision_delay_minutes"], errors="coerce").median(),
                     "incremental_delay": pd.to_numeric(group.loc[incremental, "first_decision_delay_minutes"], errors="coerce").median()})
    return pd.DataFrame(rows)


def chart(fig, name, height=420) -> str:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=50, r=14, t=14, b=75),
                      font=dict(family="Malgun Gothic, Segoe UI, sans-serif", size=13, color="#1c3342"),
                      paper_bgcolor="white", plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.23, x=0), hovermode="closest")
    fig.update_yaxes(gridcolor="#e7eef1", zerolinecolor="#cbd5e1")
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=name,
                       config={"responsive": True, "displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]})


def real_chart(real: pd.DataFrame) -> str:
    subset = real[real.period.eq("april_development") & np.isclose(real.budget, 1.0)]
    short = {"fixed_rule": "고정<br>규칙", "static_if": "현재 정보<br>AI", "robust_context": "시간 맥락<br>통계", "context_if": "시간 맥락<br>AI"}
    panels = []
    for name, metric, title, unit in [("real-count", "alert_count", "4월 경보 구간 수", "구간"), ("real-duration", "persistent_flagged_minutes", "4월 누적 경보 구간 분", "설비·분")]:
        fig = go.Figure()
        for method in METHODS:
            selected = subset.loc[subset.method.eq(method)]
            if selected.empty:
                continue
            row = selected.iloc[0]
            fig.add_trace(go.Bar(name=NAMES[method], x=[short[method]], y=[row[metric]], marker_color=COLORS[method],
                                 customdata=[[NAMES[method], row.persistent_fraction * 100]],
                                 hovertemplate="%{customdata[0]}<br>%{y:,.0f} " + unit + "<extra></extra>"))
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=0)
        fig.update_yaxes(title_text=unit)
        panels.append(f"<div class='chart-card'><h3>{title}</h3>{chart(fig, name, 330)}</div>")
    return "<div class='chart-grid'>" + "".join(panels) + "</div>"


def real_table(real: pd.DataFrame, period: str, budget=1.0) -> str:
    subset = real[real.period.eq(period) & np.isclose(real.budget, budget)].set_index("method")
    rows = []
    for method in METHODS:
        if method not in subset.index:
            continue
        row = subset.loc[method]
        rows.append([NAMES[method], number(row.alert_count), number(row.alerts_per_observed_day, 2),
                     number(row.flagged_minutes), number(row.persistent_flagged_minutes),
                     percent(row.persistent_fraction), percent(row.flagged_fraction)])
    return table(["비교 방법", "경보 구간", "설비·관측일당", "모든 탐지 분", "경보 구간 누적 분", "경보 구간 비율", "모든 탐지 비율"], rows)


def synthetic_table(synthetic: pd.DataFrame, budget=1.0) -> str:
    if synthetic.empty:
        return "<p class='note'>유효한 인위적 변화 시험 결과가 없습니다.</p>"
    subset = synthetic[np.isclose(synthetic.budget, budget)].set_index("method")
    rows = []
    for method in METHODS:
        if method not in subset.index:
            continue
        row = subset.loc[method]
        rows.append([NAMES[method], f"{number(row.detected)} / {number(row.tests)} ({percent(row.rate, 1)})",
                     number(row.baseline_alerts), f"{number(row.incremental)} / {number(row.clean)} ({percent(row.incremental_rate, 1)})",
                     number(row.incremental_delay, 1)])
    return table(["비교 방법", "주입 중 검출 / 전체 시험", "원래 경보가 있던 대조군", "깨끗한 대조군 추가 검출", "추가 검출 지연 중앙값(분)"], rows)


def synthetic_chart(by_kind: pd.DataFrame) -> str:
    if by_kind.empty:
        return ""
    fig = go.Figure()
    short = {"level_shift": "수준<br>변화", "volatility_increase": "변동성<br>증가", "phase_imbalance": "상간<br>차이", "relation_break": "관계<br>변화", "persistent_drift": "완만한<br>변화"}
    for method in METHODS:
        subset = by_kind[np.isclose(by_kind.budget, 1.0) & by_kind.method.eq(method)].set_index("kind")
        present = [kind for kind in KINDS if kind in subset.index]
        fig.add_trace(go.Bar(name=NAMES[method], x=[short[k] for k in present],
                             y=[subset.loc[k, "incremental_rate"] * 100 for k in present],
                             customdata=[[subset.loc[k, "incremental"], subset.loc[k, "clean"], subset.loc[k, "tests"], KINDS[k]] for k in present],
                             marker_color=COLORS[method],
                             hovertemplate="%{customdata[3]}<br>추가 검출 %{customdata[0]:.0f}/%{customdata[1]:.0f}<br>종류별 전체 시험 %{customdata[2]:.0f}<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="추가 검출 비율 (%)", range=[0, 105])
    fig.update_xaxes(tickangle=0)
    return "<h3>변화 종류별 · 깨끗한 대조군에서 추가 검출</h3>" + chart(fig, "synthetic-kind", 470)


def decision_section(decision) -> str:
    if not decision:
        return "<div class='notice'><strong>이 보고서는 비교 근거입니다.</strong> 경보가 적거나 인위적 변화를 잘 찾았다는 이유만으로 현장 점검 효과가 입증되지는 않습니다. 운영 반영 여부는 결과와 실제 기록을 검토한 뒤 결정합니다.</div>"
    if isinstance(decision, str):
        return f"<section class='decision'><h2>검토 결론</h2><p>{esc(decision)}</p></section>"
    pieces = [f"<section class='decision'><h2>{esc(decision.get('title', '검토 결론'))}</h2>"]
    for key in ["summary", "recommendation"]:
        if decision.get(key):
            pieces.append(f"<p>{esc(decision[key])}</p>")
    for key, label in [("bullets", "판단 근거"), ("limitations", "남은 확인")]:
        values = decision.get(key, [])
        if values:
            pieces.append(f"<h3>{label}</h3><ul>" + "".join(f"<li>{esc(v)}</li>" for v in values) + "</ul>")
    pieces.append("</section>")
    return "".join(pieces)


def feature_label(feature: str) -> str:
    if feature == "power_minus_vi_pf": return "전력과 V·I·역률 관측식의 차이"
    if feature == "current_pf_weighted_sum": return "역률을 반영한 전류 합"
    suffixes = {"_delta_past5_mean": "· 직전 5분 평균 대비 변화", "_delta_prev": "· 직전 분 대비 변화", "_past5_std": "· 직전 5분 변동성", "_relative_range": "· 상대 상간폭", "_mean": "· 1분 평균", "_std": "· 1분 내 변동성"}
    for suffix, label in suffixes.items():
        if feature.endswith(suffix):
            base = feature[:-len(suffix)]
            return SENSOR_NAMES.get(base, {"voltage": "전압", "current": "전류", "powerFactor": "역률"}.get(base, base)) + label
    match = re.fullmatch(r"(voltage|current|powerFactor)_([RST])_minus_([RST])", feature)
    if match:
        family, left, right = match.groups()
        return {"voltage": "전압", "current": "전류", "powerFactor": "역률"}[family] + f" · {left}상−{right}상"
    return feature


def reviews_section(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>검토 목록에 기록된 시간 맥락 AI 경보가 없습니다.</p>"
    chosen = frame.sort_values(["start", "module"]).drop_duplicates("module").head(5)
    cards = []
    for _, row in chosen.iterrows():
        deviations = []
        for item in str(row.get("related_large_deviations", "")).split("; "):
            name, sep, value = item.partition("=")
            if sep:
                deviations.append(f"<li><strong>{esc(feature_label(name))}</strong>: {esc(value.replace('train median ', '학습 중앙값 '))}</li>")
        start = str(row.start).replace("T", " ")[:16]
        end = str(row.end).replace("T", " ")[:16]
        decision = str(row.decision_time).replace("T", " ")[:16]
        rules = int(row.rule_flagged_minutes)
        overlap = "기존 규칙 이탈과 겹치지 않음" if rules == 0 else f"기존 규칙 이탈 {rules:,}분과 겹침"
        cards.append(f"<article class='review-card'><div class='review-top'><h3>{esc(row.module)}</h3><span class='tag'>{number(row.duration_minutes)}분 구간</span></div><p>{esc(start)} ~ {esc(end)}<br><span class='muted'>첫 경보 가능 시각 {esc(decision)}</span></p><p class='note'>{esc(overlap)}</p><details><summary>그 구간에서 평소와 크게 달랐던 입력값</summary><ul>{''.join(deviations)}</ul><p class='note'>학습 중앙값과의 편차를 보여주는 보조 설명입니다. 모델 판정의 인과적 원인이나 고장 원인 설명은 아닙니다. 각 값의 단위는 입력 항목에 따라 다릅니다.</p></details></article>")
    return "<div class='review-grid'>" + "".join(cards) + "</div>"


def correlation_section(audits: list[dict]) -> str:
    rows = []
    for item in audits:
        correlations = item.get("train_correlations", {})
        power = correlations.get("activePower_mean", {})
        rows.append([item.get("module", ""), percent(item.get("power_mean_cv"), 3),
                     *[number(power.get(f"current{p}_mean"), 3) for p in "RST"],
                     number(power.get("powerFactorR_mean"), 3), number(power.get("voltageR_mean"), 3)])
    return table(["설비", "전력 변동계수", "전력–R상 전류", "전력–S상 전류", "전력–T상 전류", "전력–R상 역률", "전력–R상 전압"], rows)




def relation_section(audit) -> str:
    if not isinstance(audit, dict) or not audit.get("summary_across_equipment"):
        return ""
    names = {"sum_current": "전력 ← R·S·T 전류 합", "phase_currents": "전력 ← R·S·T 전류 각각", "sum_VI": "전력 ← 각 상 V×I의 합", "sum_VI_PF": "전력 ← 각 상 V×I×역률의 합"}
    rows = []
    for method, name in names.items():
        summary = audit["summary_across_equipment"].get(method, {})
        periods = summary.get("periods", {})
        train = periods.get("train_dec_feb", {}).get("r2", {})
        april = periods.get("april_development", {})
        r2, mae = april.get("r2", {}), april.get("mae_W", {})
        if not r2:
            continue
        rows.append([name, number(train.get("median"), 6), number(r2.get("median"), 6),
                     f"{number(r2.get('min'), 6)} ~ {number(r2.get('max'), 6)}", number(mae.get("median"), 3)])
    return "<h3>같은 시각의 센서 관계는 강하게 관측됩니다</h3><p>전류 한 상만 보는 것과 세 상의 전류를 함께 보는 것은 다릅니다. 아래는 12월–2월 자료로 선형 관계의 계수를 맞춘 뒤 그대로 고정해 4월에 적용한 결과입니다. R²는 1에 가까울수록 이 자료의 동시 전력 변동을 잘 설명한다는 뜻입니다.</p>" + table(["동시 센서 관계", "학습 R² 중앙값", "4월 R² 중앙값", "4월 R² 설비별 범위", "4월 MAE 중앙값(W 가정)"], rows) + "<div class='notice'><strong>후속 후보: 평소의 센서 관계에서 벗어나는 잔차</strong><p>같은 시각의 전류·전압으로 설명되는 전력과 실제 관측 전력의 차이를 추가로 점검할 근거가 생겼습니다. 이번 4방법 비교의 특징량·학습 기준을 중간에 바꾼 것은 아닙니다.</p><p class='note'>동시 센서의 일관성을 확인하는 분석이며 미래 전력 예측이 아닙니다. 평균 V×평균 I는 원시 V×I의 평균과 다를 수 있고, 학습 계수는 센서 배율·정의도 흡수합니다. 검증된 물리식이나 고장 기준으로 사용할 수 없습니다. 전력·전류가 함께 비례해 증가하면 관계 잔차가 작을 수 있고, 센서 배율 변경만으로 잔차가 커질 수도 있습니다.</p></div>"


def temporal_section(records) -> str:
    if not records:
        return ""
    items = records if isinstance(records, list) else records.get("rows", [])
    if not items:
        return ""
    rows = [[row.get("module", ""), number(row.get("power_lag1min_correlation"), 5),
             number(row.get("power_lag60min_correlation"), 5), number(row.get("voltage_lag1min_correlation"), 5)] for row in items]
    finite = [abs(float(row["power_lag1min_correlation"])) for row in items if row.get("power_lag1min_correlation") is not None]
    lead = f"추가 점검에서 전력과 직전 1분 전력의 선형 상관은 절댓값 최대 {number(max(finite), 5)}였습니다." if finite else "시간 간 선형 연관을 추가로 점검했습니다."
    return f"<div class='notice warning'><strong>시간 정보를 추가하는 것만으로 유용성이 보장되지는 않습니다.</strong><p>{esc(lead)} 직전 값과의 선형 연관이 약하면 시간 변화 입력이 실제로 도움이 되는지 결과로 확인해야 합니다. 이 점검만으로 모든 시간 패턴이 없다고 단정하지는 않습니다.</p></div><details><summary>설비별 시간 간 선형 상관 보기</summary>{table(['설비', '전력–직전 1분', '전력–60분 전', 'R상 전압–직전 1분'], rows)}<p class='note'>추가 시간 구조 점검 파일의 값입니다. 분 단위 선형 상관만으로 비선형·운전 상태별 패턴까지 배제할 수 없습니다.</p></details>"


def optional_audit_details(title: str, value) -> str:
    if not value:
        return ""
    rendered = json.dumps(value, ensure_ascii=False, indent=2)
    return f"<details><summary>{esc(title)}</summary><pre>{esc(rendered)}</pre></details>"



def relation_experiment_section(output: Path) -> str:
    manifest = read_json(output / "run_manifest.json", {})
    if manifest.get("complete") is not True:
        return ""
    real = aggregate_real(read_csv(output / "real_alert_metrics.csv"))
    source = read_csv(output / "synthetic_results.csv")
    synth = aggregate_synthetic(source)
    kinds = aggregate_synthetic(source, by_kind=True)
    real_rows = []
    for period in PERIODS:
        for _, row in real[real.period.eq(period) & np.isclose(real.budget, 1.0)].iterrows():
            real_rows.append([PERIODS[period], number(row.alert_count), number(row.alerts_per_observed_day, 2),
                              number(row.flagged_minutes), number(row.persistent_flagged_minutes), percent(row.persistent_fraction), percent(row.flagged_fraction)])
    scale_note = ""
    queue_path = output / "review_queue.csv"
    if queue_path.exists():
        queue = read_csv(queue_path)
        if not queue.empty and {"absolute_residual_W", "observed_power_W"}.issubset(queue.columns):
            residual = pd.to_numeric(queue.absolute_residual_W, errors="coerce")
            fraction = residual / pd.to_numeric(queue.observed_power_W, errors="coerce").abs().replace(0, np.nan)
            scale_note = f"<div class='notice warning'><strong>현재 실제 자료에서 잡힌 차이는 작습니다.</strong><p>저장된 경보 예시 {number(len(queue))}개의 대표 잔차는 {number(residual.min(), 2)}~{number(residual.max(), 2)}W이며, 관측 전력 대비 최대 {percent(fraction.max(), 3)}입니다. 이는 전체 경보를 대표하는 무작위 표본은 아닙니다. 합성 검출이 되더라도 이 정도 차이가 현장에서 점검할 가치가 있는지는 아직 확인하지 않았습니다.</p><p class='note'>평균 사이의 관계만 보기 때문에 평균이 그대로인 흔들림 증가를 놓칠 수 있습니다. 실제 이상 탐지의 성공이나 고장 확정으로 해석하지 않습니다.</p></div>"
    synth_rows, kind_rows = [], []
    if not synth.empty:
        for _, row in synth[np.isclose(synth.budget, 1.0)].iterrows():
            synth_rows.append([f"{number(row.detected)}/{number(row.tests)}", number(row.baseline_alerts),
                               f"{number(row.incremental)}/{number(row.clean)}", percent(row.incremental_rate, 1), number(row.incremental_delay, 1)])
    if not kinds.empty:
        for kind in KINDS:
            for _, row in kinds[kinds.kind.eq(kind) & np.isclose(kinds.budget, 1.0)].iterrows():
                kind_rows.append([KINDS[kind], f"{number(row.detected)}/{number(row.tests)}", number(row.baseline_alerts),
                                  f"{number(row.incremental)}/{number(row.clean)}", percent(row.incremental_rate, 1)])
    return "<section class='section' id='relation-extra'><p class='eyebrow'>원래 4방법과 구분한 후속 탐색</p><h2>추가 실험 · 학습한 센서 관계의 불일치</h2><p>센서 관계 점검에서 확인한 전력과 각 상 V×I 합의 관계를 이용했습니다. 12월–2월에 선형 계수를 학습하고, 관측 전력과 이 관계가 설명하는 전력 사이의 차이(잔차)를 점수로 사용합니다. 경보 기준은 3월 초에 정하고 고정했으며, 동일한 3분 연속 조건을 적용했습니다.</p><div class='notice warning'><strong>자료 관계를 확인한 뒤 추가한 탐색 실험입니다.</strong><p>원래 4방법의 결과에 섞지 않았습니다. 이 실험도 이미 살펴본 4월 자료와 통제한 변화 시험을 사용하므로, 새로운 실측 자료에서의 고장 탐지 성능을 입증하지 않습니다.</p></div><h3>기본 목표 1.0 · 실제 자료의 경보 부담</h3>" + table(["기간", "경보 구간", "설비·관측일당", "모든 탐지 분", "경보 구간 누적 분", "경보 구간 비율", "모든 탐지 비율"], real_rows) + "<h3>인위적 변화 시험 · 같은 다섯 종류와 같은 생성 기준</h3>" + table(["주입 중 검출/전체", "원래 경보가 있던 대조군", "깨끗한 대조군 추가 검출", "추가 검출 비율", "추가 검출 지연 중앙값(분)"], synth_rows) + table(["변화 종류", "검출/전체", "원래 경보 있음", "추가 검출/깨끗한 대조군", "추가 검출 비율"], kind_rows) + scale_note + "<p class='note'>현재 시각의 센서 간 일관성 검사입니다. 관계에서 벗어난 값은 센서 배율·데이터 처리 문제일 수도 있고, 관계를 유지하는 동시 변화는 탐지하기 어려울 수 있습니다. 실제 고장 정밀도·재현율은 확인하지 않았습니다. 근거는 별도 relation_experiment 폴더에 저장했습니다.</p></section>"


def build_report(output: Path) -> Path:
    manifest = read_json(output / "run_manifest.json", {})
    if manifest.get("complete") is not True:
        raise RuntimeError("A completed run_manifest.json is required; partial results must not be presented as complete.")
    real_raw = read_csv(output / "real_alert_metrics.csv")
    synthetic_raw = read_csv(output / "synthetic_results.csv")
    reviews = read_csv(output / "review_queue.csv")
    audits = read_json(output / "training_audit.json", [])
    frozen = read_json(output / "protocol.json", {})
    protocol = frozen.get("protocol", frozen)
    probe_manifest = read_json(output / "synthetic_manifest.json", [])
    real = aggregate_real(real_raw)
    synthetic = aggregate_synthetic(synthetic_raw)
    by_kind = aggregate_synthetic(synthetic_raw, by_kind=True)
    decision = read_json(output / "decision.json")
    temporal_audit = read_json(output / "temporal_structure_audit.json")
    relation_audit = read_json(output / "sensor_relation_audit.json")
    probe_amendment = read_json(output / "probe_feasibility_amendment.json")
    relation_extra = relation_experiment_section(output / "relation_experiment")
    equipment_count = len(manifest.get("modules", []))
    observed_minutes = sum(a.get("valid_minutes", 0) for a in audits)
    targets = protocol.get("synthetic_target_per_kind", 13)
    actual_probes = len(probe_manifest)
    feature_count = manifest.get("feature_count", 64)
    static_count = manifest.get("static_feature_count", 34)
    timelines = [
        ("학습", "2024.12.01–2025.02.28", "설비별 평소 분포와 탐지 모델 학습"),
        ("경보 기준 설정", "2025.03.01–03.15", "설비·관측일당 목표 경보 수에 가깝게 기준 고정"),
        ("3월 후반 점검", "2025.03.16–03.31", "기준 고정 후 경보 부담 확인"),
        ("4월 개발 평가", "2025.04.01–04.29", "기존에 확인한 자료로 비교·인위적 변화 시험"),
    ]
    timeline_html = "".join(f"<div class='timeline-item'><span>{esc(a)}</span><strong>{esc(b)}</strong><p>{esc(c)}</p></div>" for a, b, c in timelines)
    method_cards = [
        ("fixed_rule", "기존 하한값", "상전압 210V·선간전압 360V·역률 85% 미만 여부를 확인합니다."),
        ("static_if", f"현재 정보 {static_count}개", "현재 분의 평균·흔들림, 상간 차이와 관측 관계를 Isolation Forest에 입력합니다."),
        ("robust_context", f"시간 맥락 {feature_count}개", "학습 중앙값에서 얼마나 벗어났는지를 단순 통계 점수로 계산합니다."),
        ("context_if", f"시간 맥락 {feature_count}개", "현재 정보에 직전 변화·과거 5분 평균과 변동성을 추가한 Isolation Forest입니다."),
    ]
    method_html = "".join(f"<article class='method-card' style='border-top-color:{COLORS[key]}'><h3>{NAMES[key]}</h3><p class='eyebrow'>{esc(label)}</p><p>{esc(description)}</p></article>" for key, label, description in method_cards)
    sensitivity_rows = []
    for period in PERIODS:
        for budget in [.5, 1., 2.]:
            sub = real[real.period.eq(period) & np.isclose(real.budget, budget)].set_index("method")
            synth_sub = synthetic[np.isclose(synthetic.budget, budget)].set_index("method") if not synthetic.empty else pd.DataFrame()
            for method in METHODS:
                if method not in sub.index:
                    continue
                row = sub.loc[method]
                sr = synth_sub.loc[method] if method in synth_sub.index else None
                sensitivity_rows.append([PERIODS[period], number(budget, 1), NAMES[method], number(row.alert_count),
                                         number(row.persistent_flagged_minutes), percent(row.persistent_fraction),
                                         f"{number(sr.incremental)}/{number(sr.clean)}" if sr is not None and period == "april_development" else "—"])
    kind_rows = []
    if not by_kind.empty:
        for kind in KINDS:
            for method in METHODS:
                selection = by_kind[by_kind.kind.eq(kind) & by_kind.method.eq(method) & np.isclose(by_kind.budget, 1.0)]
                if selection.empty:
                    continue
                row = selection.iloc[0]
                kind_rows.append([KINDS[kind], NAMES[method], f"{number(row.detected)}/{number(row.tests)}", number(row.baseline_alerts), f"{number(row.incremental)}/{number(row.clean)}", percent(row.incremental_rate, 1)])
    created = str(manifest.get("created_utc", "")).replace("T", " ")[:19]
    details = {"생성 시각(UTC)": created, "모델 입력 수": feature_count, "현재 정보 비교군 입력 수": static_count,
               "규칙 파생 라벨 학습": protocol.get("labels_used", False), "5월 정답 사용": protocol.get("may_data_used", False),
               "OpenAI API 호출": manifest.get("api_calls", 0), "대시보드 변경": manifest.get("dashboard_changed", False),
               "실험 설정 해시": manifest.get("protocol_sha256", frozen.get("sha256", ""))}
    css = """
:root{--ink:#193441;--muted:#536b78;--teal:#087f8c;--line:#dce7eb;--paper:#fff;--bg:#f1f6f8;--soft:#eaf7f7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 'Malgun Gothic','Segoe UI',sans-serif}main{max-width:1200px;margin:auto;padding:30px 24px 70px}header{padding:34px;background:#123f4c;color:#fff;border-radius:20px;margin-bottom:24px}header p{max-width:900px;margin-bottom:0;color:#dcecee}h1,h2,h3{word-break:keep-all}h1{font-size:32px;line-height:1.35;margin:10px 0}h2{font-size:23px;line-height:1.4;margin:0 0 16px}h3{font-size:17px;line-height:1.4;margin:0 0 9px}p{margin:9px 0 15px}a{color:var(--teal)}header a{color:#d8ffff}nav{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}nav a{padding:7px 12px;border:1px solid #56818a;border-radius:9px;font-size:14px;text-decoration:none}.eyebrow{font-size:13px;font-weight:700;letter-spacing:.04em;color:var(--teal);margin:0 0 7px}header .eyebrow{color:#90dfdc}.section{margin-top:24px;padding:28px;background:var(--paper);border:1px solid var(--line);border-radius:16px}.notice,.decision{padding:22px;border-radius:14px;background:var(--soft);border:1px solid #b5dede;margin-bottom:20px}.notice p:last-child,.decision p:last-child{margin-bottom:0}.warning{background:#fff7e8;border-color:#edd8ac}.muted,.note{color:var(--muted);font-size:14px}.stats,.methods,.timeline{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.stat{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px}.stat strong{display:block;font-size:27px;color:var(--teal)}.stat span{font-size:14px;color:var(--muted)}.timeline{margin:18px 0}.timeline-item{border-left:3px solid #7bbdc3;padding:0 12px}.timeline-item span{display:block;color:var(--teal);font-size:13px;font-weight:700}.timeline-item strong{font-size:14px}.timeline-item p{font-size:13px;color:var(--muted);margin:5px 0}.method-card{border:1px solid var(--line);border-top:4px solid;border-radius:11px;padding:17px}.method-card p{font-size:14px}.table-scroll{overflow-x:auto;margin:16px 0;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:collapse;font-size:14px}caption{text-align:left;padding:12px;font-weight:700}th,td{padding:12px 13px;text-align:right;border-bottom:1px solid #e7eef1;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf6f7;color:#254b58;font-weight:700}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#f8fbfc}details{border:1px solid var(--line);border-radius:10px;padding:13px 15px;margin-top:18px}summary{cursor:pointer;font-weight:700;color:#225b68}details[open] summary{margin-bottom:14px}.review-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.review-card{border:1px solid var(--line);padding:18px;border-radius:12px;min-width:0}.review-card details{font-size:13px}.review-card li{overflow-wrap:anywhere}.review-top{display:flex;align-items:start;justify-content:space-between;gap:10px}.tag{background:#e4f4f4;color:#0a727b;padding:3px 8px;border-radius:6px;font-size:13px;white-space:nowrap}li{margin:6px 0}ul{padding-left:22px}.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:25px}.chart-card{min-width:0}.chart-card h3{text-align:center}.plotly-graph-div{width:100%}.scroll-hint{display:none;color:var(--muted);font-size:12px;margin:12px 0 -9px}.table-scroll td:first-child{position:sticky;left:0;background:white}.table-scroll th:first-child{position:sticky;left:0;background:#edf6f7;z-index:1}footer{color:var(--muted);font-size:13px;margin-top:25px}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fa;padding:15px;border-radius:8px;font-size:12px}.section-title{display:flex;align-items:baseline;justify-content:space-between;gap:14px;flex-wrap:wrap}
@media(max-width:820px){.chart-grid{grid-template-columns:1fr}.scroll-hint{display:block}main{padding:16px 12px 40px}header{padding:24px}h1{font-size:27px}.section{padding:20px}.stats,.methods,.timeline{grid-template-columns:repeat(2,minmax(0,1fr))}.review-grid{grid-template-columns:1fr}}@media(max-width:480px){body{font-size:15px}h1{font-size:24px}h2{font-size:21px}.section{padding:17px}.methods,.timeline{grid-template-columns:1fr}.stat{padding:13px}.stat strong{font-size:23px}th,td{padding:10px;font-size:12px}nav a{font-size:12px}}
@media print{body{background:white}main{max-width:none;padding:0}.section,header,.notice{break-inside:avoid}details{display:block}nav{display:none}}
"""
    content = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>평소와 다른 패턴 탐지 · 비교 실험</title><style>{css}</style><script>{get_plotlyjs()}</script></head><body><main>
<header><p class="eyebrow">마가렛트 · 로컬 검토용 실험</p><h1>기준 안에 있어도,<br>평소와 다른 변화를 찾을 수 있을까?</h1><p>한 시점의 센서값에 시간 변화 정보를 더하고, 고정 규칙·단순 통계·AI를 같은 조건에서 비교했습니다. 현재 대시보드의 탐지 모델은 교체하지 않았습니다.</p><nav aria-label="보고서 목차"><a href="#design">실험 기준</a><a href="#burden">실제 경보 부담</a><a href="#probes">인위적 변화 시험</a><a href="#data">데이터 관계</a><a href="#review">검토할 기록</a></nav></header>
{decision_section(decision)}
<div class="stats"><div class="stat"><strong>{number(equipment_count)}대</strong><span>비교한 설비</span></div><div class="stat"><strong>{number(observed_minutes)}</strong><span>전체 기간 유효 설비·분</span></div><div class="stat"><strong>{number(actual_probes)}개</strong><span>실제로 구성한 변화 시험</span></div><div class="stat"><strong>미확인</strong><span>실제 고장 정밀도·재현율</span></div></div>
<section class="section" id="design"><h2>1. 무엇을 바꿔서 비교했나</h2><p>기존 전압·역률 라벨을 정답으로 사용하지 않았습니다. 설비별 과거 자료에서 평소 분포를 학습하고, 3월 초 자료로 경보 기준을 정한 뒤 고정했습니다.</p><div class="timeline">{timeline_html}</div><div class="methods">{method_html}</div><div class="notice" style="margin-top:20px"><strong>경보 1건 = 한 설비에서 3분 이상 연속 탐지된 하나의 구간</strong><p>한 분은 5초 기록 12개로 집계합니다. 세 번째 분이 끝나야 경보를 낼 수 있습니다. 1·2분짜리 짧은 탐지는 경보 수에서 제외하되 ‘모든 탐지 분’에는 포함합니다. 기록이 비거나 불완전하면 구간을 끊습니다.</p><p class="note">‘경보 구간 누적 분’은 판단에 쓰인 앞 2분을 포함한 탐지 구간 길이입니다. 실제 고장 지속 시간이나 설비 정지 시간이 아닙니다. 여러 설비를 합친 설비·분이며 서로 겹친 시간도 합산됩니다.</p></div><p class="note">기본 비교 목표는 경보 <strong>1.0구간/설비·관측일</strong>입니다. 이는 3월 1–15일의 기준 선택 목표이며, 이후 기간의 경보 수가 같아진다는 보장은 없습니다. 고정 규칙의 하한값은 이 목표에 맞춰 조정하지 않습니다.</p></section>
<section class="section" id="burden"><div class="section-title"><h2>2. 실제 자료에서 경보 부담은 얼마나 달라졌나</h2><span class="tag">기본 목표 1.0</span></div><p>경보 수가 줄어도 한 번의 경보가 오래 이어지면 점검 부담은 클 수 있습니다. <strong>구간 수와 누적 분·비율을 함께</strong> 확인해야 합니다.</p>{real_chart(real)}<h3>4월 1–29일 · 개발 평가</h3>{real_table(real, 'april_development')}<h3>3월 16–31일 · 기준 고정 후 점검</h3>{real_table(real, 'march_check')}<p class="note">비율의 분모는 각 기간에 모든 방법이 공통으로 사용할 수 있었던 유효 관측 분입니다. 처음 5분, 결측·불완전 분 및 그 뒤의 이력 부족 구간은 제외합니다. 경보가 적다는 사실만으로 정확도가 높다고 판단할 수 없습니다.</p><details><summary>목표 경보 수 0.5 · 1.0 · 2.0일 때 결과 비교</summary><p class="note">각 목표로 3월 초 기준을 따로 고정한 감도 비교입니다. 고정 규칙은 같은 결과가 반복됩니다. 4월 추가 검출은 동일한 인위적 변화 시험에서 측정했습니다.</p>{table(['기간', '설정 목표', '방법', '경보 구간', '누적 경보 구간 분', '경보 구간 비율', '4월 추가 검출/깨끗한 대조군'], sensitivity_rows)}</details></section>
<section class="section" id="probes"><h2>3. 규칙 안의 변화를 추가로 찾았나</h2><p>규칙 하한을 넘지 않는 구간에 급변·흔들림·상간 차이·관계 변화·완만한 변화를 넣었습니다. 같은 구간의 원래 자료와 변화 후 자료를 나란히 검사했습니다. 종류별 최대 {number(targets)}개를 목표로 했으며, 아래는 실제로 생성 가능한 시험만의 결과입니다.</p><div class="notice warning"><strong>실제 고장 성능이 아니라 기능 시험입니다.</strong><p>원시 5초 신호를 재현한 시험이 아니라 <strong>1분 평균·표준편차·최솟값·최댓값을 통제한 시험</strong>입니다. 규칙 범위 안에 머물게 설계했으므로 고정 규칙의 검출이 0이어도 AI가 전반적으로 우월하다는 뜻은 아닙니다.</p></div>{synthetic_table(synthetic)}<p class="note">‘주입 중 검출’은 변화가 들어간 시간 안에 새 경보 판단이 나온 시험입니다. 원래 자료에서도 이미 울리던 대조군은 별도로 표시합니다. ‘깨끗한 대조군 추가 검출’은 <strong>원래 경보가 없던 대조군 중 변화 후 검출한 수</strong>이며 방법마다 분모가 다를 수 있습니다. 이미 울리던 시험을 새로운 성공으로 세지 않습니다.</p>{synthetic_chart(by_kind)}<details><summary>종류별 시험 수와 대조군까지 보기</summary>{table(['변화 종류', '방법', '검출/전체', '원래 경보 있음', '추가 검출/깨끗한 대조군', '추가 검출 비율'], kind_rows)}</details><p class="note">검출 지연은 주입 시작부터 첫 경보 판단까지의 분입니다. 세 번째 연속 탐지 분의 종료 이후에만 판단할 수 있으며, 변화가 끝난 후에 생긴 경보를 성공으로 포함하지 않습니다.</p></section>
<section class="section" id="data"><h2>4. 센서 관계 자체가 학습할 만한 정보인가</h2><p>같은 시각의 여러 센서가 맺는 관계와, 직전 시각부터 이어지는 시간 패턴은 따로 봐야 합니다.</p>{relation_section(relation_audit)}{temporal_section(temporal_audit)}<details><summary>학습 자료의 단일 센서 상관과 전력 변동계수</summary>{correlation_section(audits)}</details><p class="note">전력 변동계수 = 표준편차 ÷ 평균. 상관계수는 −1~1이며 인과관계를 뜻하지 않습니다. —는 계산할 수 없는 값입니다. 특정 센서 조합이 함께 변한다는 가정은 자료에서 확인해야 하며, 물리적 고장 증거로 바로 해석하지 않습니다.</p></section>
{relation_extra}
<section class="section" id="review"><h2>5. 실제로 들여다볼 기록</h2><p>시간 맥락 AI가 찾은 4월 구간 중 서로 다른 설비의 예시 최대 5개입니다. 기존 규칙과 겹치지 않는 기록도 고장으로 확정할 수 없습니다.</p>{reviews_section(reviews)}<p class="note">이 목록은 설비별 저장된 초기 경보 예시에서 뽑았습니다. 위험도 순위나 전체 경보의 대표 표본이 아닙니다. 운전 조건·정비 기록과 대조해 점검 가치가 있는지 사람이 확인해야 합니다.</p></section>
<section class="section"><h2>운영 적용 전에 남은 확인</h2><ul><li>4월은 이전에 여러 번 살펴본 개발 자료입니다. 새 실측 기간에서 기준을 바꾸지 않고 다시 확인해야 합니다.</li><li>실제 고장·정상 판정 자료가 없어 현장 정밀도·재현율과 오경보율은 계산하지 않았습니다.</li><li>인위적 변화 시험의 종류·세기·센서 조합 밖에서도 도움이 되는지는 아직 알 수 없습니다.</li><li>현장 담당자가 각 경보의 점검 가치와 원인을 기록한 뒤, 같은 점검 부담에서 방법들을 비교해야 합니다.</li><li>이번 비교는 기존 배포 모델의 수치를 직접 재현한 것이 아닙니다. 현재 정보 {static_count}개와 시간 맥락 {feature_count}개로 새로 학습한 비교 실험입니다.</li></ul><details><summary>재현용 실험 정보와 산출물</summary>{table(['항목', '값'], details.items())}<p class="note">같은 폴더의 real_alert_metrics.csv, synthetic_results.csv, training_audit.json, review_queue.csv, protocol.json, synthetic_manifest.json에 근거가 저장되어 있습니다. 이 HTML의 그래프는 인터넷 연결 없이 열립니다.</p><p class="note">탐지 입력은 해당 분 종료 시점까지의 정보만 사용합니다. 과거 5분 통계는 현재 분을 제외하며, 표준화와 모델은 학습 기간에서만 맞춥니다. 전력은 W로 해석한 작업 가정을 사용했으며, 원본 단위 정의는 추가 확인이 필요합니다.</p>{optional_audit_details("추가 센서 관계 점검의 상세 수치", relation_audit)}{optional_audit_details("인위적 변화 시험의 구성 가능성 조정 기록", probe_amendment)}</details></section>
<footer>로컬 실험 검토본 · 기존 대시보드/모델 교체 없음 · GitHub 업로드 없음 · API 호출 없음<br>실험 완료 기록: {esc(created)} UTC</footer></main><script>document.querySelectorAll('details').forEach(el=>el.addEventListener('toggle',()=>{{if(el.open){{el.querySelectorAll('.plotly-graph-div').forEach(plot=>Plotly.Plots.resize(plot));}}}}));</script></body></html>"""
    target = output / "context_anomaly_report.html"
    target.write_text(content, encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    print(build_report(args.outputs))


if __name__ == "__main__":
    main()
