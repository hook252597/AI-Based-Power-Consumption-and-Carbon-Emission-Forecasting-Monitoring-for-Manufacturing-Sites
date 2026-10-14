"""Predeclared March-split comparisons; April is exploratory reevaluation.

The supplied labels are provisional rule-derived markers, not confirmed faults.
All detectors fit December-February sensors. Thresholds use March 1-15 labels;
candidate choice uses March 16-31 labels. No May data is read.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve

from .anomaly import (FEATURES, KEYS, MODULE, SEED, TEST_START, TEST_END,
                      add_counts, confusion, feature_matrix, isolation_score,
                      metrics, model_identity, prepare_cache, rule_parameters,
                      rule_score, safe_id, verify_model_identity)

CALIB_START, CALIB_END = 20250301000000, 20250316000000
SELECT_START, SELECT_END = CALIB_END, 20250401000000
FOCUSED_FEATURES = ["voltageR", "voltageS", "voltageT", "powerFactorR", "powerFactorS", "powerFactorT"]
METHODS = ("if_threshold", "if_voltage_pf", "robust_distance")
NOTICE = "April was inspected in V1 and in error analysis before V2 was designed. April V2 metrics are exploratory reevaluation, not a fresh held-out test. Supplied labels match the educational voltage<210/PF<85 rule throughout audited March samples and April rows, but are not confirmed physical fault labels."
LIMITATIONS = ["The q=0.99999 threshold is close to the largest calibration observations; it is not a guarantee of a future 0.001% false-alarm rate.", "The selected six-feature model focuses on voltage and power factor; power/current-only faults are outside its input scope.", "The saved model missed synthetic single-sensor voltage/power-factor deviations, so the improved proxy agreement does not establish sensitivity to unfamiliar sensor combinations. A separate documented_rule_anomaly reference is provided at inference and is not merged with the learned model flag."]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def march_masks(localtime):
    t = np.asarray(localtime, dtype=np.int64)
    return (t >= CALIB_START) & (t < CALIB_END), (t >= SELECT_START) & (t < SELECT_END)


def candidate_specs():
    specs = []
    descriptions = {"if_threshold": "기존 10개 입력 IF의 판정 기준 보정", "if_voltage_pf": "전압·역률 6개 입력으로 IF 재학습", "robust_distance": "10개 센서의 학습 중앙값·IQR 기반 최대 통계거리"}
    for stage, method in enumerate(METHODS, 1):
        for beta in (1, 2):
            for q in (0.9999, 0.99999):
                specs.append({"name": f"{method}_f{beta}_q{str(q).replace('.', '')}", "method": method, "stage": stage, "stage_name": {1: "임계값 보정", 2: "입력 변수 개선", 3: "탐지 방법 비교"}[stage], "description": descriptions[method], "positive_f_beta": beta, "zero_positive_quantile": q, "reference_only": False})
    return specs


def build_protocol(manifest):
    return {"version": 1, "seed": SEED, "input_features": FEATURES, "focused_features": FOCUSED_FEATURES,
            "source_signature": manifest["signature"], "source_fingerprint": manifest["verification"]["sha256_parsed_selected_columns"],
            "train": {"start_inclusive": 20241201000000, "end_exclusive": 20250301000000, "rows_per_equipment": 51000, "labels_used": False},
            "calibration": {"start_inclusive": CALIB_START, "end_exclusive": CALIB_END, "source": "cached March random sample"},
            "selection": {"start_inclusive": SELECT_START, "end_exclusive": SELECT_END, "source": "cached March random sample", "policy": "Among candidates meeting precision>=0.70 and recall>=0.80, maximize pooled all-equipment F1; if none meet both, maximize F1 over all. Ties: recall, precision, simpler stage, listed candidate order."},
            "candidates": candidate_specs(), "positive_threshold_policy": "Maximize specified F-beta over finite calibration score thresholds; ties prefer higher recall, precision, then higher threshold.",
            "no_positive_policy": "Use specified empirical calibration score quantile: finite, equipment remains active. This is an alarm-budget reference, not an assumption about the true fault rate.",
            "forest_parameters": {"n_estimators": 100, "max_samples": 2048, "random_state": SEED, "n_jobs": 2},
            "references": [{"name": "v1_frozen_if", "reference_only": True, "notice": "Thresholds previously tuned on all March, so not eligible for March16-31 selection."}, {"name": "documented_rule", "reference_only": True, "rule": "any voltageR/S/T < 210 OR any powerFactorR/S/T < 85", "notice": "Educational rule reference, not a machine learning fault ground truth."}],
            "evaluation_notice": NOTICE, "april_policy": "Freeze selection and serialized bundle before reading any April cache. Evaluate all 13 equipment and retain all rows. No threshold/model changes based on April.", "may_access": "forbidden"}


def write_protocol(path, protocol):
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["protocol"] != protocol:
            raise ValueError("Existing V2 protocol differs; use a fresh output directory.")
        return existing
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "sha256": digest_json(protocol), "protocol": protocol}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def calibrated_threshold(y, scores, beta, zero_positive_quantile):
    y, score = np.asarray(y, bool), np.asarray(scores, float)
    if len(score) == 0 or not np.isfinite(score).all():
        raise ValueError("Calibration scores must be finite and nonempty")
    if not y.any():
        return float(np.quantile(score, zero_positive_quantile)), "finite_calibration_alarm_budget"
    p, r, thresholds = precision_recall_curve(y, score)
    b2 = beta * beta
    denominator = b2 * p[:-1] + r[:-1]
    f = np.divide((1+b2)*p[:-1]*r[:-1], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    idx = np.lexsort((thresholds, p[:-1], r[:-1], f))[-1]
    return float(thresholds[idx]), f"calibration_proxy_max_f{beta}"


def documented_rule(x):
    voltage_idx = [FEATURES.index(f) for f in ("voltageR", "voltageS", "voltageT")]
    pf_idx = [FEATURES.index(f) for f in ("powerFactorR", "powerFactorS", "powerFactorT")]
    return (x[:, voltage_idx] < 210).any(axis=1) | (x[:, pf_idx] < 85).any(axis=1)


def detector_scores(detectors, x):
    focus = x[:, [FEATURES.index(f) for f in FOCUSED_FEATURES]]
    return {"if_threshold": isolation_score(detectors["original_if"], x), "if_voltage_pf": isolation_score(detectors["focused_if"], focus), "robust_distance": rule_score(x, detectors["robust"])}


def choose_candidate(candidates):
    eligible = [c for c in candidates if not c.get("reference_only")]
    if not eligible:
        raise ValueError("No eligible candidate")
    qualified = [c for c in eligible if c["march_selection"]["precision"] >= .7 and c["march_selection"]["recall"] >= .8]
    pool = qualified or eligible
    return max(pool, key=lambda c: (c["march_selection"]["f1"], c["march_selection"]["recall"], c["march_selection"]["precision"], -c["stage"], -eligible.index(c)))


def fit_and_select(root, cache, output, model_dir, manifest, protocol_record):
    specs = candidate_specs()
    totals = {c["name"]: {} for c in specs}
    references = {"v1_frozen_if": {}, "documented_rule": {}}
    equipment, threshold_rows, split_counts = {}, [], {}
    source_model_id = model_identity(manifest, 100, 2048)
    identity = {"protocol_sha256": protocol_record["sha256"], "source_fingerprint": manifest["verification"]["sha256_parsed_selected_columns"], "seed": SEED}
    for module in manifest["samples"]:
        old = joblib.load(root / "models/anomaly" / f"{safe_id(module)}.joblib")
        verify_model_identity(old, source_model_id)
        with np.load(cache / f"sample_{safe_id(module)}.npz") as d:
            train_x, train_t = d["train_x"], d["train_localtime"]
            if not ((train_t >= 20241201000000) & (train_t < 20250301000000)).all():
                raise ValueError("Training sample crosses December-February boundary")
            x, y, ts = d["tune_x"], d["tune_label"], d["tune_localtime"]
            cm, sm = march_masks(ts)
            if (cm & sm).any() or not (cm | sm).all():
                raise ValueError("Invalid calibration/selection split")
            focus_idx = [FEATURES.index(f) for f in FOCUSED_FEATURES]
            detector_path = model_dir / f"detectors_{safe_id(module)}.joblib"
            if detector_path.exists():
                detectors = joblib.load(detector_path)
                if detectors["identity"] != identity:
                    raise ValueError("Cached V2 detectors have stale source/protocol identity")
            else:
                detectors = {"identity": identity, "original_if": old["model"], "focused_if": IsolationForest(n_estimators=100, max_samples=2048, random_state=SEED, n_jobs=2, contamination="auto").fit(train_x[:, focus_idx]), "robust": rule_parameters(train_x)}
                joblib.dump(detectors, detector_path, compress=3)
            scores = detector_scores(detectors, x)
            thresholds = {}
            for spec in specs:
                score = scores[spec["method"]]
                threshold, how = calibrated_threshold(y[cm], score[cm], spec["positive_f_beta"], spec["zero_positive_quantile"])
                thresholds[spec["name"]] = threshold
                add_counts(totals[spec["name"]], confusion(y[sm], score[sm] >= threshold))
                threshold_rows.append({"module": module, "candidate": spec["name"], "threshold": threshold, "policy": how, "calibration_rows": int(cm.sum()), "calibration_proxy_positives": int(y[cm].sum()), "selection_rows": int(sm.sum()), "selection_proxy_positives": int(y[sm].sum())})
            old_threshold = old["thresholds"]["isolation_forest"]["threshold"]
            add_counts(references["v1_frozen_if"], confusion(y[sm], scores["if_threshold"][sm] >= old_threshold))
            add_counts(references["documented_rule"], confusion(y[sm], documented_rule(x[sm])))
            split_counts[module] = {"train": len(train_x), "calibration": int(cm.sum()), "selection": int(sm.sum()), "calibration_proxy_positive": int(y[cm].sum()), "selection_proxy_positive": int(y[sm].sum())}
            equipment[module] = {"detectors": detectors, "thresholds": thresholds, "v1_threshold": old_threshold}
            print(f"V2 March fit/calibrate/selection {module}", flush=True)
    candidates = [{**c, "march_selection": metrics(totals[c["name"]])} for c in specs]
    chosen = choose_candidate(candidates)
    stage_best = {str(stage): choose_candidate([c for c in candidates if c["stage"] == stage])["name"] for stage in (1, 2, 3)}
    references_records = [{"name": name, "stage": 0, "reference_only": True, "description": "기존 IF 동결 참고값" if name == "v1_frozen_if" else "강의자료 210V/85% 고정 규칙", "march_selection": metrics(counts)} for name, counts in references.items()]
    selection = {"selected_candidate": chosen["name"], "selected_method": chosen["method"], "selected_march": chosen["march_selection"], "candidates": candidates + references_records, "stage_best": stage_best, "split_rows": split_counts, "policy": protocol_record["protocol"]["selection"]["policy"], "identity": identity, "frozen_utc": datetime.now(timezone.utc).isoformat(), "before_april_read": True, "evaluation_notice": NOTICE}
    selection_path = output / "selection.json"
    if selection_path.exists():
        previous = json.loads(selection_path.read_text(encoding="utf-8"))
        for key in ("selected_candidate", "candidates", "identity"):
            if previous[key] != selection[key]:
                raise ValueError("Frozen March selection differs on rerun; refuse to evaluate April")
        selection = previous
    else:
        selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(threshold_rows).to_csv(output / "calibration_thresholds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{**{k: c[k] for k in ("name", "stage", "reference_only")}, **c["march_selection"]} for c in selection["candidates"]]).to_csv(output / "march_selection_comparison.csv", index=False, encoding="utf-8-sig")
    bundles = {}
    for candidate_name in sorted(set(stage_best.values()) | {chosen["name"]}):
        spec = next(c for c in specs if c["name"] == candidate_name)
        bundle = {"schema_version": 1, "candidate": candidate_name, "method": spec["method"], "features": FEATURES, "focused_features": FOCUSED_FEATURES, "identity": identity, "evaluation_notice": NOTICE, "equipment": {module: {"detectors": item["detectors"], "threshold": item["thresholds"][candidate_name]} for module, item in equipment.items()}}
        path = model_dir / f"{candidate_name}.joblib"
        joblib.dump(bundle, path, compress=3)
        bundles[candidate_name] = path
    selected_path = model_dir / "selected_model.joblib"
    joblib.dump(joblib.load(bundles[chosen["name"]]), selected_path, compress=3)
    return equipment, selection, selected_path


def load_selected(path):
    bundle = joblib.load(path)
    if bundle.get("schema_version") != 1 or bundle.get("features") != FEATURES or bundle.get("focused_features") != FOCUSED_FEATURES or bundle.get("method") not in METHODS:
        raise ValueError("Invalid V2 sensor/model schema")
    if len(bundle["equipment"]) != 13:
        raise ValueError("Expected all 13 equipment in selected bundle")
    if not all(np.isfinite(d["threshold"]) for d in bundle["equipment"].values()):
        raise ValueError("Nonfinite anomaly threshold")
    return bundle


def predict_observations(frame, bundle):
    missing = set(KEYS + FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required five-second sensor columns: {sorted(missing)}")
    unknown = set(frame[MODULE].astype(str)) - set(bundle["equipment"])
    if unknown:
        raise ValueError(f"No fitted V2 detector for equipment: {sorted(unknown)}")
    result = frame[KEYS].copy()
    result["anomaly_score"] = np.nan
    result["is_anomaly"] = False
    result["threshold"] = np.nan
    result["candidate"] = bundle["candidate"]
    result["documented_rule_anomaly"] = documented_rule(feature_matrix(frame))
    for module, positions in frame.groupby(MODULE, sort=False).indices.items():
        item = bundle["equipment"][str(module)]
        x = feature_matrix(frame.iloc[positions])
        method = bundle["method"]
        if method == "if_threshold":
            scores = isolation_score(item["detectors"]["original_if"], x)
        elif method == "if_voltage_pf":
            scores = isolation_score(item["detectors"]["focused_if"], x[:, [FEATURES.index(f) for f in FOCUSED_FEATURES]])
        elif method == "robust_distance":
            scores = rule_score(x, item["detectors"]["robust"])
        else:
            raise ValueError("Unsupported selected detector")
        result.iloc[positions, result.columns.get_loc("anomaly_score")] = scores
        result.iloc[positions, result.columns.get_loc("is_anomaly")] = scores >= item["threshold"]
        result.iloc[positions, result.columns.get_loc("threshold")] = item["threshold"]
    return result


def evaluate_april(cache, output, equipment, selection, manifest):
    # This is intentionally reached only AFTER selection.json and selected_model.joblib
    # exist. Every method and threshold is already frozen by March evidence.
    if not (output / "selection.json").exists():
        raise ValueError("March selection must be frozen before April evaluation")
    specs = candidate_specs()
    names = [c["name"] for c in specs] + ["v1_frozen_if", "documented_rule"]
    totals, by_module, by_reason = {name: {} for name in names}, {}, {}
    selected_name = selection["selected_candidate"]
    selected_method = selection["selected_method"]
    hourly_parts, example_parts = [], []
    rows = 0
    start = time.monotonic()
    for i, part in enumerate(manifest["test_parts"]):
        module = part["module"]
        item = equipment[module]
        with np.load(cache / part["file"]) as d:
            x, y, ts, reason = d["x"], d["label"], d["localtime"], d["reason"]
            if not ((ts >= TEST_START) & (ts < TEST_END)).all():
                raise ValueError("Non-April row in exploratory cache")
            scores = detector_scores(item["detectors"], x)
            predictions = {s["name"]: scores[s["method"]] >= item["thresholds"][s["name"]] for s in specs}
            predictions["v1_frozen_if"] = scores["if_threshold"] >= item["v1_threshold"]
            predictions["documented_rule"] = documented_rule(x)
            for name, pred in predictions.items():
                counts = confusion(y, pred)
                add_counts(totals[name], counts)
                add_counts(by_module.setdefault((module, name), {}), counts)
                for code, rname in ((1, "voltage"), (2, "power_factor")):
                    rr = by_reason.setdefault((module, name, rname), {"proxy_positive": 0, "detected": 0})
                    mask = reason == code
                    rr["proxy_positive"] += int(mask.sum())
                    rr["detected"] += int((mask & pred).sum())
            pred = predictions[selected_name]
            score = scores[selected_method]
            hf = pd.DataFrame({"module": module, "hour": ts // 10000, "observations": 1, "proxy_flags": y.astype(int), "selected_flags": pred.astype(int), "v1_flags": predictions["v1_frozen_if"].astype(int), "selected_score_max": score})
            agg = {c: ("max" if c.endswith("score_max") else "sum") for c in hf.columns if c not in ("module", "hour")}
            hourly_parts.append(hf.groupby(["module", "hour"], as_index=False).agg(agg))
            categories = {"true_positive_proxy": y & pred, "false_positive_proxy": ~y & pred, "false_negative_proxy": y & ~pred}
            for category, mask in categories.items():
                eligible = np.flatnonzero(mask)
                if len(eligible) == 0:
                    continue
                chosen = eligible[np.argsort(score[eligible])[-min(30, len(eligible)):]]
                ef = pd.DataFrame(x[chosen], columns=FEATURES)
                ef.insert(0, "localtime", ts[chosen])
                ef.insert(0, "module", module)
                ef["category"] = category
                ef["anomaly_score"] = score[chosen]
                ef["threshold"] = item["thresholds"][selected_name]
                ef["proxy_reason"] = np.array(["normal", "voltage", "power_factor"])[reason[chosen]]
                example_parts.append(ef)
            rows += len(x)
        if i % 5 == 0:
            print(f"V2 April exploratory scored {rows:,} rows in {time.monotonic()-start:.1f}s", flush=True)
    hourly = pd.concat(hourly_parts, ignore_index=True)
    agg = {c: ("max" if c.endswith("score_max") else "sum") for c in hourly.columns if c not in ("module", "hour")}
    hourly = hourly.groupby(["module", "hour"], as_index=False).agg(agg)
    hourly["localtime"] = pd.to_datetime(hourly.pop("hour").astype(str), format="%Y%m%d%H")
    hourly.to_csv(output / "april_hourly_anomalies.csv", index=False, encoding="utf-8-sig")
    if example_parts:
        examples = pd.concat(example_parts, ignore_index=True).sort_values("anomaly_score", ascending=False).groupby(["module", "category"]).head(30)
        examples.to_csv(output / "april_observation_examples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"module": module, "candidate": name, **metrics(counts)} for (module, name), counts in by_module.items()]).to_csv(output / "april_equipment_proxy_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"module": module, "candidate": name, "proxy_reason": reason, **counts, "recall": counts["detected"] / counts["proxy_positive"] if counts["proxy_positive"] else None} for (module, name, reason), counts in by_reason.items()]).to_csv(output / "april_reason_proxy_recall.csv", index=False, encoding="utf-8-sig")
    return {name: metrics(counts) for name, counts in totals.items()}, rows, len(hourly), time.monotonic()-start


def run(root):
    root = Path(root)
    output, model_dir, cache = root / "outputs/anomaly_v2", root / "models/anomaly_v2", root / ".cache/anomaly"
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not (cache / "manifest.json").exists():
        raise ValueError("V2 requires the previously verified V1 cache; run train_anomaly.py first. V2 will not scan April source rows before declaring its protocol.")
    # Existing source check only verifies metadata and cache files; it does not
    # read May or inspect April labels before the protocol/selection is frozen.
    manifest = prepare_cache(root.parent / "프로젝트/data/rtu_data_full.csv", root.parent / "프로젝트/data/rtu_data_full_flagged.csv", output, cache_dir=cache)
    protocol = write_protocol(output / "protocol.json", build_protocol(manifest))
    equipment, selection, selected_path = fit_and_select(root, cache, output, model_dir, manifest, protocol)
    bundle = load_selected(selected_path)
    # Verify saved-model equivalence using March-only observations before April.
    checked = 0
    for module in manifest["samples"]:
        with np.load(cache / f"sample_{safe_id(module)}.npz") as d:
            frame = pd.DataFrame(d["tune_x"][:128], columns=FEATURES)
            frame[MODULE] = module
            frame["localtime"] = d["tune_localtime"][:128]
        prediction = predict_observations(frame, bundle)
        expected = detector_scores(equipment[module]["detectors"], feature_matrix(frame))[selection["selected_method"]]
        np.testing.assert_array_equal(prediction["anomaly_score"].to_numpy(), expected)
        np.testing.assert_array_equal(prediction["is_anomaly"].to_numpy(), expected >= equipment[module]["thresholds"][selection["selected_candidate"]])
        checked += len(frame)
    model_hash_before = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    selection_hash_before = hashlib.sha256((output / "selection.json").read_bytes()).hexdigest()
    (output / "april_evaluation_gate.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "selected_model_sha256": model_hash_before, "selection_file_sha256": selection_hash_before, "notice": "Written after March selection and before April scoring."}, indent=2), encoding="utf-8")
    april, rows, hourly_rows, seconds = evaluate_april(cache, output, equipment, selection, manifest)
    if hashlib.sha256(selected_path.read_bytes()).hexdigest() != model_hash_before or hashlib.sha256((output / "selection.json").read_bytes()).hexdigest() != selection_hash_before:
        raise ValueError("Selected model or thresholds changed during April evaluation")
    records = [{**c, "april": april[c["name"]]} for c in selection["candidates"]]
    selected_metrics = april[selection["selected_candidate"]]
    old_metrics = april["v1_frozen_if"]
    summary = {"selected_candidate": selection["selected_candidate"], "selected_method": selection["selected_method"], "candidates": records, "stage_best": selection["stage_best"], "selected_march": selection["selected_march"], "selected_april": selected_metrics, "baseline_april": old_metrics, "delta_vs_v1": {k: selected_metrics[k] - old_metrics[k] for k in ("precision", "recall", "f1")}, "both_precision_recall_improved": selected_metrics["precision"] > old_metrics["precision"] and selected_metrics["recall"] > old_metrics["recall"], "rows_scored": rows, "expected_rows": manifest["source_split_rows"]["test"], "all_test_rows_scored": rows == manifest["source_split_rows"]["test"], "hourly_rows": hourly_rows, "selected_bundle_path": str(selected_path.relative_to(root)), "evaluation_notice": NOTICE, "acceptance_status": "Exploratory proxy agreement only; confirmed fault acceptance remains undetermined", "policy": selection["policy"], "protocol_sha256": protocol["sha256"], "selection_frozen_utc": selection["frozen_utc"], "reload_validation": {"equipment": 13, "march_rows": checked, "scores_and_flags_exactly_equal": True}, "freeze_validation": {"selected_model_sha256": model_hash_before, "selection_file_sha256": selection_hash_before, "unchanged_before_after_april": True}, "score_seconds": seconds}
    summary["limitations"] = LIMITATIONS
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"name": c["name"], "stage": c["stage"], "reference_only": c["reference_only"], **c["april"]} for c in records]).to_csv(output / "april_comparison.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k: summary[k] for k in ("selected_candidate", "selected_april", "delta_vs_v1", "both_precision_recall_improved", "rows_scored")}, ensure_ascii=False, indent=2), flush=True)
    return summary
