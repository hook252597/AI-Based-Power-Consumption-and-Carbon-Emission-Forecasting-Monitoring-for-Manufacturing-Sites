"""Time-separated equipment anomaly detection with explicitly provisional labels.

Labels in the supplied flagged file are never model inputs. Their provenance is
unknown: every reported precision/recall number measures agreement with these
proxy labels, not verified physical faults.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve

MODULE = "module(equipment)"
FEATURES = ["activePower", "voltageR", "voltageS", "voltageT", "currentR", "currentS", "currentT", "powerFactorR", "powerFactorS", "powerFactorT"]
KEYS = [MODULE, "localtime"]
SEED = 20261012
TRAIN_START, TRAIN_END = 20241201000000, 20250301000000
TUNE_START, TUNE_END = 20250301000000, 20250401000000
TEST_START, TEST_END = 20250401000000, 20250430000000
PROXY_NOTICE = "is_outlier/outlier_reason have unknown provenance. Label-based tuning and test metrics measure agreement with provisional labels, not validated real-world anomaly accuracy."


def split_identity():
    return {"train": [TRAIN_START, TRAIN_END], "tune": [TUNE_START, TUNE_END], "test": [TEST_START, TEST_END], "interval": "half_open"}


def model_identity(manifest, trees, max_samples, seed=SEED):
    """Bind models to source, sampling policy, feature order, and split policy."""
    identity = {"source_signature": manifest["signature"], "source_fingerprint": manifest["verification"]["sha256_parsed_selected_columns"], "samples": manifest["samples"], "features": FEATURES, "seed": seed, "splits": split_identity(), "parameters": {"trees": trees, "max_samples": max_samples}, "threshold_policy": "march_proxy_max_f1_or_q999_v1", "schema_version": 2}
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"details": identity, "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def verify_model_identity(bundle, expected):
    if bundle.get("training_identity") != expected:
        raise ValueError("Cached anomaly model training identity differs or is absent; use a fresh model directory and retrain. Source, seed, features, sampling, or split policy may have changed.")


def time_masks(localtime):
    """Half-open splits; the isolated April 30 midnight sample is excluded."""
    t = np.asarray(localtime, dtype=np.int64)
    return ((t >= TRAIN_START) & (t < TRAIN_END), (t >= TUNE_START) & (t < TUNE_END), (t >= TEST_START) & (t < TEST_END))


def feature_matrix(frame):
    """An explicit allowlist prevents accidental use of labels or identifiers."""
    x = frame.loc[:, FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Input features contain NaN/infinity; resolve before training.")
    return x


def rule_parameters(x):
    q25, center, q75 = np.quantile(x, [0.25, 0.5, 0.75], axis=0)
    return {"center": center, "scale": np.maximum(q75 - q25, 1e-6)}


def rule_score(x, params):
    """Largest sensor deviation in training IQR units; larger means abnormal."""
    return np.max(np.abs((x - params["center"]) / params["scale"]), axis=1)


def isolation_score(model, x):
    """Invert sklearn score_samples so larger consistently means abnormal."""
    return -model.score_samples(x)


def predict_observations(frame, bundles):
    """Score a new batch using frozen models and thresholds, without labels.

    ``bundles`` maps the original equipment name to its loaded joblib bundle.
    Measurements must use the same raw sensor units as training (power is W).
    The five-second sensor schema is required; hourly May totals are unsuitable.
    """
    missing = set(KEYS + FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required five-second sensor columns: {sorted(missing)}")
    unknown = set(frame[MODULE].astype(str)) - set(bundles)
    if unknown:
        raise ValueError(f"No fitted anomaly model for equipment: {sorted(unknown)}")
    result = frame.loc[:, KEYS].copy()
    for method in ("isolation_forest", "rules"):
        result[f"{method}_score"] = np.nan
        result[f"{method}_anomaly"] = False
    for module, positions in frame.groupby(MODULE, sort=False).indices.items():
        bundle = bundles[str(module)]
        if bundle["features"] != FEATURES:
            raise ValueError("Saved feature schema is incompatible")
        x = feature_matrix(frame.iloc[positions])
        scores = {"isolation_forest": isolation_score(bundle["model"], x), "rules": rule_score(x, bundle["rules"])}
        for method, score in scores.items():
            result.iloc[positions, result.columns.get_loc(f"{method}_score")] = score
            result.iloc[positions, result.columns.get_loc(f"{method}_anomaly")] = score >= bundle["thresholds"][method]["threshold"]
    return result


def threshold_from_proxy(labels, scores):
    """Tune F1 on March only; ties prefer higher precision then threshold.

    Equipment without a positive March proxy label instead uses a predeclared
    99.9th-percentile March score reference. That fallback is NOT a learned true
    fault rate, and its April false positives are retained in the results.
    """
    y = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if not y.any():
        return float(np.quantile(scores, 0.999)), "unlabeled_equipment_march_score_q999"
    p, r, t = precision_recall_curve(y, scores)
    f1 = np.divide(2 * p[:-1] * r[:-1], p[:-1] + r[:-1], out=np.zeros_like(t), where=(p[:-1] + r[:-1]) > 0)
    idx = np.lexsort((t, p[:-1], f1))[-1]
    return float(t[idx]), "march_proxy_max_f1"


def confusion(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    return {"tp": int(np.count_nonzero(y & pred)), "fp": int(np.count_nonzero(~y & pred)), "fn": int(np.count_nonzero(y & ~pred)), "tn": int(np.count_nonzero(~y & ~pred))}


def metrics(counts):
    tp, fp, fn, tn = (int(counts[k]) for k in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {**counts, "rows": tp + fp + fn + tn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0, "proxy_target_met": precision >= 0.70 and recall >= 0.80}


def add_counts(destination, new):
    for key in ("tp", "fp", "fn", "tn"):
        destination[key] = destination.get(key, 0) + new[key]


def safe_id(module):
    return re.sub(r"[^0-9A-Za-z_-]", "_", module)


def _priority_sample(existing, x, localtime, labels, reason, limit, rng):
    new = {"x": x, "localtime": localtime, "label": labels, "reason": reason, "priority": rng.random(len(x))}
    if existing is not None:
        new = {k: np.concatenate([existing[k], v]) for k, v in new.items()}
    if len(new["priority"]) > limit:
        idx = np.argpartition(new["priority"], limit - 1)[:limit]
        new = {k: v[idx] for k, v in new.items()}
    return new


def prepare_cache(raw_path, flagged_path, output, chunksize=250_000, monthly_sample=17_000, tune_sample=100_000, cache_dir=None):
    """Single paired scan, full relevant-sensor equality check, compact cache."""
    output = Path(output)
    cache = Path(cache_dir) if cache_dir is not None else output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    signature = {"raw_size": raw_path.stat().st_size, "raw_mtime": raw_path.stat().st_mtime_ns, "flagged_size": flagged_path.stat().st_size, "flagged_mtime": flagged_path.stat().st_mtime_ns, "features": FEATURES, "monthly_sample": monthly_sample, "tune_sample": tune_sample, "seed": SEED, "splits": split_identity(), "sampling_policy": "per_equipment_month_priority_reservoir_v1", "version": 2}
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] == signature and all((cache / f).exists() for f in manifest["files"]):
            print("Reusing fully verified anomaly cache", flush=True)
            return manifest
        raise ValueError("Anomaly cache signature differs; use a fresh output directory.")
    dtypes = {MODULE: "string", "localtime": "int64", **{f: "float32" for f in FEATURES}}
    raw_iter = pd.read_csv(raw_path, usecols=KEYS + FEATURES, dtype=dtypes, chunksize=chunksize)
    flag_iter = pd.read_csv(flagged_path, usecols=KEYS + FEATURES + ["is_outlier", "outlier_reason"], dtype={**dtypes, "is_outlier": bool, "outlier_reason": "string"}, chunksize=chunksize)
    rng = np.random.default_rng(SEED)
    reservoirs, tune, files, parts, count_by_split = {}, {}, [], [], {"train": 0, "tune": 0, "test": 0, "excluded": 0}
    reasons = {"normal": 0, "voltage": 1, "power_factor": 2}
    reason_counts, rows = {}, 0
    digest = hashlib.sha256()
    start = time.monotonic()
    for part, raw in enumerate(raw_iter):
        try:
            flagged = next(flag_iter)
        except StopIteration as exc:
            raise ValueError("Flagged file ended early") from exc
        if len(raw) != len(flagged):
            raise ValueError("Source row counts differ")
        for col in KEYS + FEATURES:
            if not np.array_equal(raw[col].to_numpy(), flagged[col].to_numpy()):
                raise ValueError(f"Raw/flagged mismatch in {col} at chunk {part}")
        digest.update(pd.util.hash_pandas_object(raw[KEYS + FEATURES], index=False).to_numpy().tobytes())
        x = feature_matrix(raw)
        ts = raw["localtime"].to_numpy()
        label = flagged["is_outlier"].to_numpy(bool)
        mapped = flagged["outlier_reason"].map(reasons)
        if mapped.isna().any():
            raise ValueError("Unknown proxy reason in flagged source")
        reason = mapped.to_numpy(dtype=np.uint8)
        if not np.array_equal(label, reason != 0):
            raise ValueError("Proxy label/reason conflict")
        for name, n in flagged["outlier_reason"].value_counts().items():
            reason_counts[str(name)] = reason_counts.get(str(name), 0) + int(n)
        masks = time_masks(ts)
        for name, mask in zip(("train", "tune", "test"), masks):
            count_by_split[name] += int(mask.sum())
        count_by_split["excluded"] += int((~(masks[0] | masks[1] | masks[2])).sum())
        for module, indexes in raw.groupby(MODULE, sort=False).indices.items():
            module = str(module)
            for month in (202412, 202501, 202502):
                idx = indexes[(ts[indexes] // 100000000) == month]
                if len(idx):
                    key = (module, month)
                    reservoirs[key] = _priority_sample(reservoirs.get(key), x[idx], ts[idx], label[idx], reason[idx], monthly_sample, rng)
            idx = indexes[masks[1][indexes]]
            if len(idx):
                tune[module] = _priority_sample(tune.get(module), x[idx], ts[idx], label[idx], reason[idx], tune_sample, rng)
            idx = indexes[masks[2][indexes]]
            if len(idx):
                filename = f"test_{safe_id(module)}_{part:04d}.npz"
                np.savez_compressed(cache / filename, x=x[idx], localtime=ts[idx], label=label[idx], reason=reason[idx])
                files.append(filename)
                parts.append({"module": module, "file": filename, "rows": len(idx)})
        rows += len(raw)
        if part % 8 == 0:
            print(f"Verified/cache {rows:,} rows in {time.monotonic()-start:.1f}s", flush=True)
    try:
        next(flag_iter)
        raise ValueError("Flagged file has extra rows")
    except StopIteration:
        pass
    sample_counts = {}
    for module in sorted(tune):
        samples = [reservoirs[(module, month)] for month in (202412, 202501, 202502)]
        train_x = np.concatenate([s["x"] for s in samples])
        train_ts = np.concatenate([s["localtime"] for s in samples])
        filename = f"sample_{safe_id(module)}.npz"
        t = tune[module]
        np.savez_compressed(cache / filename, train_x=train_x, train_localtime=train_ts, tune_x=t["x"], tune_localtime=t["localtime"], tune_label=t["label"], tune_reason=t["reason"])
        files.append(filename)
        sample_counts[module] = {"train": len(train_x), "train_per_month": monthly_sample, "tune": len(t["x"]), "tune_proxy_positive": int(t["label"].sum())}
    manifest = {"signature": signature, "files": files, "test_parts": parts, "samples": sample_counts, "source_rows": rows, "source_split_rows": count_by_split, "source_proxy_reason_counts": reason_counts, "verification": {"all_rows_identical_on": KEYS + FEATURES, "sha256_parsed_selected_columns": digest.hexdigest(), "hash_note": "SHA256 of aligned parsed row hashes, not a byte-level file hash"}, "proxy_notice": PROXY_NOTICE, "scan_seconds": time.monotonic()-start}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def train_models(output, model_dir, manifest, trees=100, max_samples=2048, cache_dir=None):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir is not None else Path(output) / "cache"
    trained = {}
    records = []
    identity = model_identity(manifest, trees, max_samples)
    for module, sample_info in manifest["samples"].items():
        model_path = model_dir / f"{safe_id(module)}.joblib"
        if model_path.exists():
            bundle = joblib.load(model_path)
            verify_model_identity(bundle, identity)
            if bundle["parameters"] != {"trees": trees, "max_samples": max_samples}:
                raise ValueError("Existing model parameters differ")
            print(f"Reusing model {module}", flush=True)
        else:
            with np.load(cache / f"sample_{safe_id(module)}.npz") as sample:
                x, tx = sample["train_x"], sample["tune_x"]
                if not time_masks(sample["train_localtime"])[0].all() or not time_masks(sample["tune_localtime"])[1].all():
                    raise ValueError("Split overlap in training cache")
                forest = IsolationForest(n_estimators=trees, max_samples=max_samples, contamination="auto", random_state=SEED, n_jobs=2).fit(x)
                rules = rule_parameters(x)
                tune_y = sample["tune_label"]
                info = {}
                for method, score in (("isolation_forest", isolation_score(forest, tx)), ("rules", rule_score(tx, rules))):
                    threshold, how = threshold_from_proxy(tune_y, score)
                    info[method] = {"threshold": threshold, "selection": how, "tune_proxy_metrics": metrics(confusion(tune_y, score >= threshold)), "tune_proxy_average_precision": float(average_precision_score(tune_y, score)) if tune_y.any() else None}
                bundle = {"module": module, "features": FEATURES, "model": forest, "rules": rules, "thresholds": info, "parameters": {"trees": trees, "max_samples": max_samples}, "seed": SEED, "samples": sample_info, "proxy_notice": PROXY_NOTICE, "train_period": [TRAIN_START, TRAIN_END], "tune_period": [TUNE_START, TUNE_END], "training_identity": identity}
                joblib.dump(bundle, model_path, compress=3)
            print(f"Trained {module}: IF threshold {bundle['thresholds']['isolation_forest']['threshold']:.4f}", flush=True)
        trained[module] = bundle
        for method, info in bundle["thresholds"].items():
            records.append({"module": module, "method": method, "threshold": info["threshold"], "threshold_selection": info["selection"], **info["tune_proxy_metrics"], "average_precision": info["tune_proxy_average_precision"]})
    pd.DataFrame(records).to_csv(Path(output) / "march_tuning_proxy_metrics.csv", index=False, encoding="utf-8-sig")
    return trained


def evaluate_test(output, trained, manifest, cache_dir=None):
    output = Path(output)
    cache = Path(cache_dir) if cache_dir is not None else output / "cache"
    totals = {m: {} for m in ("isolation_forest", "rules")}
    by_module, by_reason, hourly, examples = {}, {}, [], []
    rng = np.random.default_rng(SEED + 1)
    ap_y, ap_scores = [], {m: [] for m in totals}
    rows = 0
    start = time.monotonic()
    # Exactly 1000 top-score cases per chunk are considered, then reduced to
    # 100 per equipment/method across the entire April test period.
    for part_index, part in enumerate(manifest["test_parts"]):
        module = part["module"]
        bundle = trained[module]
        with np.load(cache / part["file"]) as data:
            x, ts, y, reasons = data["x"], data["localtime"], data["label"], data["reason"]
            if not time_masks(ts)[2].all():
                raise ValueError("Non-test row found in April cache")
            scores = {"isolation_forest": isolation_score(bundle["model"], x), "rules": rule_score(x, bundle["rules"])}
            # Fixed Bernoulli sample for compact, explicitly sampled test AP.
            ap_sample = rng.random(len(x)) < 0.05
            ap_y.append(y[ap_sample])
            frame = pd.DataFrame({"module": module, "hour": ts // 10000, "observations": 1, "proxy_flags": y.astype(np.int8)})
            for method, score in scores.items():
                pred = score >= bundle["thresholds"][method]["threshold"]
                counts = confusion(y, pred)
                add_counts(totals[method], counts)
                add_counts(by_module.setdefault((module, method), {}), counts)
                for reason_code, reason_name in ((1, "voltage"), (2, "power_factor")):
                    reason_mask = reasons == reason_code
                    r = by_reason.setdefault((module, method, reason_name), {"proxy_positive": 0, "detected": 0})
                    r["proxy_positive"] += int(reason_mask.sum())
                    r["detected"] += int((reason_mask & pred).sum())
                frame[f"{method}_flags"] = pred.astype(np.int8)
                frame[f"{method}_score_max"] = score
                ap_scores[method].append(score[ap_sample])
                n = min(1000, len(score))
                idx = np.argpartition(score, -n)[-n:]
                local_examples = pd.DataFrame(x[idx], columns=FEATURES)
                local_examples.insert(0, "localtime", ts[idx])
                local_examples.insert(0, "module", module)
                local_examples["method"] = method
                local_examples["score"] = score[idx]
                local_examples["threshold"] = bundle["thresholds"][method]["threshold"]
                local_examples["predicted_anomaly"] = pred[idx]
                local_examples["proxy_label"] = y[idx]
                local_examples["proxy_reason"] = np.array(["normal", "voltage", "power_factor"])[reasons[idx]]
                # Explain the largest normalized observed sensor deviation;
                # this is supporting evidence, not a causal attribution to IF.
                residual = np.abs((x[idx] - bundle["rules"]["center"]) / bundle["rules"]["scale"])
                local_examples["largest_observed_deviation_sensor"] = np.array(FEATURES)[residual.argmax(axis=1)]
                examples.append(local_examples)
            aggregates = {col: ("max" if col.endswith("score_max") else "sum") for col in frame.columns if col not in ("module", "hour")}
            hourly.append(frame.groupby(["module", "hour"], as_index=False).agg(aggregates))
            rows += len(x)
        if part_index % 4 == 0:
            print(f"April full-test scored {rows:,} rows in {time.monotonic()-start:.1f}s", flush=True)
    h = pd.concat(hourly, ignore_index=True)
    aggregates = {col: ("max" if col.endswith("score_max") else "sum") for col in h.columns if col not in ("module", "hour")}
    h = h.groupby(["module", "hour"], as_index=False).agg(aggregates)
    h["localtime"] = pd.to_datetime(h.pop("hour").astype(str), format="%Y%m%d%H")
    h.to_csv(output / "april_hourly_anomalies.csv", index=False, encoding="utf-8-sig")
    e = pd.concat(examples, ignore_index=True).sort_values("score", ascending=False).groupby(["module", "method"], sort=False).head(100)
    e.sort_values(["module", "method", "localtime"]).to_csv(output / "april_top_observations.csv", index=False, encoding="utf-8-sig")
    module_metrics = [{"module": module, "method": method, **metrics(value)} for (module, method), value in by_module.items()]
    pd.DataFrame(module_metrics).to_csv(output / "april_equipment_proxy_metrics.csv", index=False, encoding="utf-8-sig")
    reason_metrics = [{"module": module, "method": method, "proxy_reason": reason, **value, "recall": value["detected"] / value["proxy_positive"] if value["proxy_positive"] else None} for (module, method, reason), value in by_reason.items()]
    pd.DataFrame(reason_metrics).to_csv(output / "april_reason_proxy_recall.csv", index=False, encoding="utf-8-sig")
    sampled_y = np.concatenate(ap_y)
    result = {"proxy_notice": PROXY_NOTICE, "rows_scored": rows, "expected_rows": manifest["source_split_rows"]["test"], "all_test_rows_scored": rows == manifest["source_split_rows"]["test"], "hourly_rows": len(h), "hours_per_equipment": h.groupby("module").size().to_dict(), "methods": {method: {**metrics(counts), "sampled_average_precision": float(average_precision_score(sampled_y, np.concatenate(ap_scores[method])))} for method, counts in totals.items()}, "ap_sample": {"method": "fixed_seed_5_percent_Bernoulli", "rows": len(sampled_y), "positives": int(sampled_y.sum()), "notice": "Only AP is sampled; confusion/precision/recall/F1 are from every April row."}, "test_period": {"start_inclusive": "2025-04-01 00:00:00", "end_exclusive": "2025-04-30 00:00:00"}, "physical_anomaly_success": "undetermined_pending_label_provenance", "score_seconds": time.monotonic()-start}
    (output / "test_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run(raw_path, flagged_path, output, model_dir, chunksize=250_000, cache_dir=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = prepare_cache(Path(raw_path), Path(flagged_path), output, chunksize=chunksize, cache_dir=cache_dir)
    trained = train_models(output, model_dir, manifest, cache_dir=cache_dir)
    result = evaluate_test(output, trained, manifest, cache_dir=cache_dir)
    result["training"] = {"period": "2024-12-01 <= localtime < 2025-03-01", "features": FEATURES, "equipment_models": len(trained), "samples": manifest["samples"], "isolation_forest": {"n_estimators": 100, "max_samples": 2048, "contamination": "auto", "n_jobs": 2}, "threshold_tuning_period": "2025-03-01 <= localtime < 2025-04-01", "frozen_before_test": True}
    result["source_validation"] = {"rows": manifest["source_rows"], **manifest["verification"]}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows_scored": result["rows_scored"], "methods": result["methods"], "proxy_notice": PROXY_NOTICE}, ensure_ascii=False, indent=2), flush=True)
    return result
