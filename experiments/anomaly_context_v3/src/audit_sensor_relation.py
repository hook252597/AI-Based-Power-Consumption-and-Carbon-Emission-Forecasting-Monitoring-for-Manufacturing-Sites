"""Read-only sensor-relation audit; no anomaly labels or model/threshold updates.

Fit simple same-minute linear relationships on Dec-Feb only, then reuse those
coefficients unchanged on March/April. This is a contemporaneous consistency
audit, not a forecast or proof of fault detection.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE = "module(equipment)"
PHASES = "RST"
PERIODS = {
    "train_dec_feb": ("2024-12-01", "2025-03-01"),
    "march_calibration": ("2025-03-01", "2025-03-16"),
    "march_check": ("2025-03-16", "2025-04-01"),
    "april_development": ("2025-04-01", "2025-04-30"),
}
METHODS = {
    "sum_current": ["sum_current_A"],
    "phase_currents": ["currentR_mean", "currentS_mean", "currentT_mean"],
    "sum_VI": ["sum_VI_proxy_VA"],
    "sum_VI_PF": ["sum_VI_PF_proxy_W"],
}
QUANTILES = {"p50": 0.5, "p90": 0.9, "p95": 0.95,
             "p99": 0.99, "p999": 0.999, "p9999": 0.9999}


def metric_summary(y, prediction):
    residual = y - prediction
    sst = np.square(y - y.mean()).sum()
    return {
        "n": len(y),
        "power_mean_W": float(y.mean()),
        "power_std_W": float(y.std(ddof=0)),
        "r2": float(1 - np.square(residual).sum() / sst) if sst else None,
        "mae_W": float(np.abs(residual).mean()),
        "rmse_W": float(np.sqrt(np.square(residual).mean())),
        "residual_mean_W": float(residual.mean()),
        "residual_std_W": float(residual.std(ddof=0)),
        "absolute_residual_quantiles_W": {
            name: float(np.quantile(np.abs(residual), quantile))
            for name, quantile in QUANTILES.items()
        },
        "absolute_residual_max_W": float(np.max(np.abs(residual))),
    }


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    return {"min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max())}


def audit():
    started = time.perf_counter()
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"], "Aggregation must be complete before auditing."
    columns = [MODULE, "timestamp", "complete_minute", "activePower_mean"]
    columns += [f"{sensor}{phase}_mean"
                for sensor in ["voltage", "current", "powerFactor"]
                for phase in PHASES]
    equipment_results = []
    for equipment in manifest["equipment"]:
        data = pd.read_parquet(
            ROOT / "data/minute.parquet", columns=columns,
            filters=[(MODULE, "==", equipment), ("complete_minute", "==", True),
                     ("timestamp", ">=", pd.Timestamp("2024-12-01")),
                     ("timestamp", "<", pd.Timestamp("2025-04-30"))],
        ).sort_values("timestamp").reset_index(drop=True)
        voltage = data[[f"voltage{s}_mean" for s in PHASES]].to_numpy(dtype=float)
        current = data[[f"current{s}_mean" for s in PHASES]].to_numpy(dtype=float)
        power_factor = data[[f"powerFactor{s}_mean" for s in PHASES]].to_numpy(dtype=float) / 100.0
        data["sum_current_A"] = current.sum(axis=1)
        data["sum_VI_proxy_VA"] = (voltage * current).sum(axis=1)
        data["sum_VI_PF_proxy_W"] = (voltage * current * power_factor).sum(axis=1)
        masks = {name: ((data.timestamp >= start) & (data.timestamp < end)).to_numpy()
                 for name, (start, end) in PERIODS.items()}
        training = masks["train_dec_feb"]
        power = data.activePower_mean.to_numpy(dtype=float)
        ytrain = power[training]
        result = {"equipment": equipment, "complete_minutes": len(data),
                  "train_n": int(training.sum()), "train_correlations": {}, "relationships": {}}
        for variable in ["currentR_mean", "currentS_mean", "currentT_mean",
                         "sum_current_A", "sum_VI_proxy_VA", "sum_VI_PF_proxy_W",
                         "powerFactorR_mean", "powerFactorS_mean", "powerFactorT_mean"]:
            result["train_correlations"][variable] = float(
                np.corrcoef(data.loc[training, variable].to_numpy(dtype=float), ytrain)[0, 1]
            )
        for name, features in METHODS.items():
            X = data[features].to_numpy(dtype=float)
            Xtrain = X[training]
            center_x, center_y = Xtrain.mean(axis=0), ytrain.mean()
            coefficients = np.linalg.lstsq(Xtrain - center_x, ytrain - center_y, rcond=None)[0]
            intercept = float(center_y - center_x @ coefficients)
            prediction = X @ coefficients + intercept
            residual_train = ytrain - prediction[training]
            reference_absolute_p999 = float(np.quantile(np.abs(residual_train), 0.999))
            fit = {"features": features,
                   "fit_period": list(PERIODS["train_dec_feb"]),
                   "intercept_W": intercept,
                   "coefficients": {feature: float(coefficient) for feature, coefficient in zip(features, coefficients)},
                   "training_absolute_residual_p999_W": reference_absolute_p999,
                   "periods": {}}
            for period, mask in masks.items():
                metrics = metric_summary(power[mask], prediction[mask])
                absolute_residual = np.abs(power[mask] - prediction[mask])
                metrics["fraction_above_train_abs_residual_p999"] = float(
                    (absolute_residual > reference_absolute_p999).mean()
                )
                fit["periods"][period] = metrics
            april_indices = np.flatnonzero(masks["april_development"])
            top_indices = april_indices[
                np.argsort(np.abs(power[april_indices] - prediction[april_indices]))[-3:][::-1]
            ]
            fit["largest_april_residual_examples"] = [
                {"timestamp": data.timestamp.iloc[idx].isoformat(),
                 "observed_power_W": float(power[idx]), "relationship_expected_power_W": float(prediction[idx]),
                 "residual_W": float(power[idx] - prediction[idx]),
                 "inputs": {feature: float(data[feature].iloc[idx]) for feature in features}}
                for idx in top_indices
            ]
            result["relationships"][name] = fit
        equipment_results.append(result)
    summaries = {}
    for name in METHODS:
        summaries[name] = {"periods": {}}
        if len(METHODS[name]) == 1:
            summaries[name]["coefficient_range_across_equipment"] = summarize_values([
                r["relationships"][name]["coefficients"][METHODS[name][0]]
                for r in equipment_results
            ])
        summaries[name]["intercept_W_range_across_equipment"] = summarize_values([
            r["relationships"][name]["intercept_W"] for r in equipment_results
        ])
        for period in PERIODS:
            summaries[name]["periods"][period] = {
                metric: summarize_values([
                    r["relationships"][name]["periods"][period][metric] for r in equipment_results
                ]) for metric in ["r2", "mae_W", "residual_std_W",
                                  "fraction_above_train_abs_residual_p999"]
            }
    output = {
        "audit_type": "Post-hoc read-only empirical same-minute sensor relation audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"minute_data": "data/minute.parquet", "raw_sha256_from_manifest": manifest["source"]["sha256"],
                   "activePower_unit": manifest["units"]["activePower"]},
        "methods": METHODS,
        "periods": PERIODS,
        "policy": {
            "fit_on_dec_feb_only": True, "post_train_coefficients_fixed": True,
            "anomaly_labels_used": False, "may_data_used": False,
            "model_protocol_or_deployment_changed": False,
            "train_p999_role": "Distribution stability reference only, not a selected operational alert threshold or false-positive rate.",
        },
        "interpretation_limits": [
            "This predicts contemporaneous measured power from contemporaneous sensors; it is not future power forecasting.",
            "sum_VI and sum_VI_PF multiply minute means. Mean(V)*mean(I) is not identical to mean(V*I); residuals include within-minute covariance and aggregation effects.",
            "Strong R2 supports investigating relationship deviations but does not prove that deviations are faults or that a detector has useful precision/recall.",
            "A fitted empirical coefficient absorbs source scaling/definitions. Do not treat it as a validated electrical safety or power-conversion formula.",
            "Post-train data already belongs to a development project. This audit is exploratory and does not create a new untouched final holdout.",
            "Small lag correlations do not exclude every nonlinear temporal pattern; this audit only examines the explicit contemporaneous linear relationships.",
            "A proportional increase in both current and power can remain consistent with this relation, while sensor scaling/measurement changes may create a residual without equipment damage.",
        ],
        "equipment_count": len(equipment_results),
        "summary_across_equipment": summaries,
        "equipment": equipment_results,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    target = ROOT / "outputs/sensor_relation_audit.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"equipment": len(equipment_results), "elapsed_seconds": output["elapsed_seconds"],
                      "summary_across_equipment": summaries}, ensure_ascii=True, indent=2))
    return output


if __name__ == "__main__":
    audit()
