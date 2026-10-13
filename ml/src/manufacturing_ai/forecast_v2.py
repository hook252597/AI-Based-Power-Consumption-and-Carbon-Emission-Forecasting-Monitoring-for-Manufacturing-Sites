"""AI-only forecast selection on March; April/May are retrospective evaluations.

Mean/median remain explicit comparators and can never become the selected model.
Artifacts reuse forecast.ForecastModel for compatibility with existing inference.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits

from .forecast import (CARBON, MODULE, RATE, SEED, ForecastModel, calendar_features,
                       evaluate, export_prediction, fit_model, grid_for_dates,
                       load_hourly, save_json, validate_reference)

FAMILIES = ("xgboost", "random_forest", "extra_trees", "hist_gradient_boosting")
NOTICE = "April and May were inspected during earlier experiments. These results are retrospective reevaluations, not fresh unseen tests. This run freezes the AI choice using March factory MAE before evaluating April/May; no April/May-based reselection is allowed."
SELECTION_POLICY = "Select the AI candidate with the smallest factory-total MAE on March 10 00:00 through March 31 00:00 inclusive (505 hours). Mean/median are reference-only. Ties use fixed candidate order. April/May never select or replace the winner."
UNIT_CONTRACT = {"sensor_power_assumption": "activePower is W, supported by the supplied course rules and accumulator increments", "physical_energy_formula": "sum of 5-second mean activePower(W) / 720000 = kWh", "course_native_formula": "course_native hourly_pow = physical kWh * 1000", "course_native_is_physical_kWh": False, "physical_cost_formula": "physical kWh * 180 KRW/kWh", "physical_carbon_formula": "physical kWh * 0.424 kgCO2/kWh", "reference_cost_formula": "course-native hourly_pow * 180; this is not a physical KRW bill under the W assumption", "reference_carbon_formula": "course-native hourly_pow * 0.424; this is not physical kgCO2 under the W assumption", "provider_unit_confirmation_pending": True}
METRIC_DEFINITIONS = {"factory_mae_kWh": "Mean absolute error of hourly sums across all equipment, in physical kWh for the observed training-series aggregation", "normalized_mae": "Hourly factory MAE divided by the mean actual hourly factory total; a ratio, not a percentage", "factory_bias_kWh": "Mean(predicted factory kWh - actual factory kWh)", "module_mae_mean_kWh": "Unweighted mean of equipment-specific hourly MAE values", "all": "All hours in the evaluation month available in the source", "focus": "Multiweek task-like evaluation window: March10-March31 00:00 or April10-April29 23:00, both inclusive", "mae_course_native": "MAE for supplied reference columns hourly_pow, may_bill, may_carbon using the supplied numerical convention", "mae_kWh_if_reference_is_Wh": "MAE of reference hourly_pow divided by1000; physical interpretation is conditional on the provider confirming hourly_pow is Wh", "may_normalized_mae": "Course-native hourly_pow MAE divided by mean reference hourly_pow", "basic_error_target_passed": "Normalized hourly factory MAE <=0.01; a proposed internal goal, not an official acceptance criterion"}


def candidate_specs():
    grids = {
        "xgboost": [
            {"max_depth": 2, "n_estimators": 100, "learning_rate": .03, "min_child_weight": 100, "reg_lambda": 20},
            {"max_depth": 3, "n_estimators": 300, "learning_rate": .03, "min_child_weight": 200, "reg_lambda": 50},
            {"max_depth": 4, "n_estimators": 300, "learning_rate": .03, "min_child_weight": 200, "reg_lambda": 50}],
        "random_forest": [{"n_estimators": 128, "max_depth": None, "min_samples_leaf": leaf, "max_features": 1.0, "bootstrap": True} for leaf in (50, 200, 800)],
        "extra_trees": [{"n_estimators": 128, "max_depth": None, "min_samples_leaf": leaf, "max_features": 1.0, "bootstrap": False} for leaf in (50, 200, 800)],
        "hist_gradient_boosting": [
            {"loss": "absolute_error", "learning_rate": .05, "max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 10.0, "early_stopping": False},
            {"loss": "absolute_error", "learning_rate": .05, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 200, "l2_regularization": 30.0, "early_stopping": False},
            {"loss": "absolute_error", "learning_rate": .05, "max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 800, "l2_regularization": 100.0, "early_stopping": False}],
    }
    return [{"name": f"{family}_{i+1}", "family": family, "parameters": p, "reference_only": False} for family in FAMILIES for i, p in enumerate(grids[family])]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_io_paths(hourly_path, reference_path, output_dir, model_dir, project_root=None):
    hourly, reference, output, models = (Path(p).resolve() for p in (hourly_path, reference_path, output_dir, model_dir))
    if output == models or output.is_relative_to(models) or models.is_relative_to(output):
        raise ValueError("Output and model directories must be separate and nonoverlapping")
    for source in (hourly, reference):
        if source.is_relative_to(output) or source.is_relative_to(models):
            raise ValueError("Read-only input source must not be inside an output/model directory")
    project_root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    protected = (project_root / "outputs/forecast", project_root / "models/forecast")
    for destination in (output, models):
        for existing_v1 in protected:
            if destination == existing_v1 or destination.is_relative_to(existing_v1) or existing_v1.is_relative_to(destination):
                raise ValueError("V1 forecast directories are protected; use dedicated V2 directories")
    return hourly, reference, output, models


def validate_training_grid(data):
    expected = pd.date_range("2024-12-01 00:00", "2025-04-29 23:00", freq="h")
    actual = pd.DatetimeIndex(data.datetime.drop_duplicates().sort_values())
    if not actual.equals(expected) or len(data) != 3600*13 or data[MODULE].nunique() != 13:
        raise ValueError("Expected exactly3600 complete hourly timestamps from Dec1 00:00 through Apr29 23:00 and46800 equipment-hour rows")


def fit_ai_model(train, family, parameters):
    if family not in FAMILIES:
        raise ValueError("AI family must be xgboost/random_forest/extra_trees/hist_gradient_boosting")
    if train.empty or not np.isfinite(train.energy_kWh).all():
        raise ValueError("Training energy must be nonempty and finite")
    with threadpool_limits(limits=2):
        if family == "xgboost":
            return fit_model(train, family, parameters)
        factories = {"random_forest": RandomForestRegressor, "extra_trees": ExtraTreesRegressor, "hist_gradient_boosting": HistGradientBoostingRegressor}
        options = {"random_state": SEED, **parameters}
        if family in ("random_forest", "extra_trees"):
            options.update(n_jobs=2, criterion="squared_error")
        estimator = factories[family](**options)
        modules = sorted(train[MODULE].unique())
        estimator.fit(calendar_features(train, modules), train.energy_kWh)
        if family in ("random_forest", "extra_trees"):
            # Fixed tree summation order makes serialized inference bit-exact.
            # Training uses two workers; inference uses one worker within budget.
            estimator.set_params(n_jobs=1)
    return ForecastModel(family, modules, str(train.datetime.max()), estimator=estimator, parameters=parameters)


def select_ai_candidate(records):
    eligible = [r for r in records if r["family"] in FAMILIES and not r.get("reference_only", False)]
    if not eligible:
        raise ValueError("No AI candidates available")
    # min() preserves declared order for exact ties; no April/May field is read.
    return min(eligible, key=lambda r: r["march"]["focus"]["factory_mae_kWh"])


def evaluate_reference(native, reference):
    joined = reference[["hourly_pow", "may_bill", "may_carbon"]].join(native, lsuffix="_actual", rsuffix="_predicted", validate="one_to_one")
    if len(joined) != 505 or joined.isna().any().any():
        raise ValueError("Reference/prediction must match all fixed505 timestamps")
    mae = {key: float(np.abs(joined[key+"_actual"] - joined[key+"_predicted"]).mean()) for key in ("hourly_pow", "may_bill", "may_carbon")}
    normalized = mae["hourly_pow"] / float(reference.hourly_pow.mean())
    result = {"rows": len(joined), "mae_course_native": mae, "mae_kWh_if_reference_is_Wh": mae["hourly_pow"] / 1000.0, "normalized_mae": normalized, "basic_error_target_passed": bool(normalized <= .01), "unit_confirmation_pending": True, "retrospective_reevaluation": True}
    return result, joined


def forecast_schema(model):
    probe = grid_for_dates("2025-05-10", "2025-05-10", model.modules)
    return {"serialized_class": "manufacturing_ai.forecast.ForecastModel", "kind": model.kind, "module_count": len(model.modules), "modules": model.modules, "required_inference_columns": ["datetime", MODULE], "feature_columns": list(calendar_features(probe, model.modules).columns), "feature_policy": "hour, weekday, weekend, equipment one-hot only", "forbidden_future_inputs": ["future activePower", "future current/voltage", "actual future energy", "May reference values or lags"], "prediction_unit": "physical kWh per equipment-hour", "course_native_scale": 1000.0, "last_training_time": model.last_training_time, "nonnegative_output": True}


def validate_saved_model(path, original):
    loaded = joblib.load(path)
    if not isinstance(loaded, ForecastModel) or loaded.kind != original.kind or loaded.modules != original.modules:
        raise ValueError("Serialized ForecastModel schema mismatch")
    probe = grid_for_dates("2025-05-10", "2025-05-31", original.modules)
    with threadpool_limits(limits=2):
        expected, actual = original.predict(probe), loaded.predict(probe)
    np.testing.assert_array_equal(expected, actual)
    assert expected.shape == (505*len(original.modules),)
    assert np.isfinite(expected).all() and (expected >= 0).all()
    return {"class_compatible": True, "grid_rows": len(probe), "hours": 505, "equipment": len(original.modules), "scores_exactly_equal_after_reload": True, "schema": forecast_schema(loaded)}


def run(hourly_path, reference_path, output_dir, model_dir):
    start = time.monotonic()
    hourly_path, reference_path, output, models = validate_io_paths(hourly_path, reference_path, output_dir, model_dir)
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256_file(hourly_path)
    data = load_hourly(hourly_path)
    validate_training_grid(data)
    train = data[(data.datetime >= "2024-12-01") & (data.datetime < "2025-03-01")]
    march = data[(data.datetime >= "2025-03-01") & (data.datetime < "2025-04-01")]
    april = data[(data.datetime >= "2025-04-01") & (data.datetime < "2025-04-30")]
    if len(data) != len(train)+len(march)+len(april) or not all(len(x) for x in (train, march, april)):
        raise ValueError("Expected complete December-February training, March selection, and April1-29 source rows only")
    if len(data[MODULE].unique()) != 13:
        raise ValueError("Expected all13 equipment")
    specs = candidate_specs()
    protocol = {"version": 1, "seed": SEED, "thread_limit": 2, "hourly_source_name": hourly_path.name, "hourly_source_sha256": source_hash_before, "training_window": "2024-12-01 <=datetime<2025-03-01", "selection_window": "2025-03-10 00:00<=datetime<=2025-03-31 00:00", "selection_policy": SELECTION_POLICY, "candidates": specs, "baselines": [{"name": k, "reference_only": True} for k in ("mean", "median")], "features": ["hour", "weekday", "weekend", "equipment one-hot"], "future_sensor_inputs": False, "lag_inputs": False, "random_time_split": False, "hist_gradient_early_stopping": False, "xgboost_fixed_options": {"objective": "reg:absoluteerror", "eval_metric": "mae", "tree_method": "hist", "random_state": SEED, "n_jobs": 2, "subsample": 1.0, "colsample_bytree": 1.0}, "ensemble_fixed_options": {"random_state": SEED, "n_jobs": 2, "criterion": "squared_error"}, "unit_contract": UNIT_CONTRACT, "evaluation_notice": NOTICE, "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}}
    protocol["forest_inference_workers"] = 1
    protocol["forest_inference_worker_reason"] = "Deterministic tree summation order for exact serialized prediction replay; training uses2 workers."
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old["protocol"] != protocol:
            raise ValueError("Existing forecast V2 protocol differs; use a fresh output directory")
        protocol_record = old
    else:
        protocol_record = {"created_utc": datetime.now(timezone.utc).isoformat(), "sha256": stable_digest(protocol), "protocol": protocol, "metric_definitions": METRIC_DEFINITIONS}
        save_json(protocol_path, protocol_record)
    records = []
    for spec in specs + [{"name": k, "family": k, "parameters": None, "reference_only": True} for k in ("mean", "median")]:
        model = fit_model(train, spec["family"]) if spec["reference_only"] else fit_ai_model(train, spec["family"], spec["parameters"])
        with threadpool_limits(limits=2):
            measured, _, _ = evaluate(model, march, "2025-03-10", "2025-03-31")
        records.append({**spec, "march": measured})
        print(f"March {spec['name']}: factory MAE={measured['focus']['factory_mae_kWh']:.8f} physical kWh", flush=True)
    winner = select_ai_candidate(records)
    family_best = {family: select_ai_candidate([r for r in records if r["family"] == family]) for family in FAMILIES}
    baselines = {r["family"]: r for r in records if r["reference_only"]}
    selection = {"selected_candidate": winner["name"], "selected_family": winner["family"], "selected_parameters": winner["parameters"], "selected_march": winner["march"], "family_best_candidates": {family: spec["name"] for family, spec in family_best.items()}, "selection_policy": SELECTION_POLICY, "protocol_sha256": protocol_record["sha256"], "march_candidates": records, "frozen_utc": datetime.now(timezone.utc).isoformat(), "frozen_before_current_run_april_evaluation": True, "frozen_before_current_run_reference_read": True, "evaluation_notice": NOTICE}
    selection_path = output / "selection.json"
    if selection_path.exists():
        old = json.loads(selection_path.read_text(encoding="utf-8"))
        for key in ("selected_candidate", "selected_family", "selected_parameters", "family_best_candidates", "protocol_sha256"):
            if old[key] != selection[key]:
                raise ValueError("Frozen March AI selection changed on rerun")
        selection = old
    else:
        save_json(selection_path, selection)
    selection_hash = sha256_file(selection_path)
    pd.DataFrame([{"candidate": r["name"], "family": r["family"], "reference_only": r["reference_only"], "march_mae_kWh": r["march"]["focus"]["factory_mae_kWh"], "march_normalized_mae": r["march"]["focus"]["normalized_mae"], "parameters_json": json.dumps(r["parameters"], sort_keys=True)} for r in records]).to_csv(output / "march_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"FROZEN AI winner: {winner['name']} (mean/median ineligible)", flush=True)
    benchmarks = {**family_best, **baselines}
    train_april = data[data.datetime < "2025-04-01"]
    for family, spec in benchmarks.items():
        model = fit_model(train_april, family) if spec["reference_only"] else fit_ai_model(train_april, family, spec["parameters"])
        with threadpool_limits(limits=2):
            measured, _, totals = evaluate(model, april, "2025-04-10", "2025-04-29 23:00")
        spec["april"] = measured
        totals.to_csv(output / f"april_{family}.csv", encoding="utf-8-sig")
        print(f"April retrospective {family}: {measured['focus']['factory_mae_kWh']:.8f} kWh", flush=True)
    final_models, reload_checks, paths, native_predictions = {}, {}, {}, {}
    for family, spec in benchmarks.items():
        final = fit_model(data, family) if spec["reference_only"] else fit_ai_model(data, family, spec["parameters"])
        path = models / f"{family}_model.joblib"
        joblib.dump(final, path, compress=3)
        reload_checks[family] = validate_saved_model(path, final)
        paths[family] = path.name
        final_models[family] = final
        with threadpool_limits(limits=2):
            _, native_predictions[family] = export_prediction(final, "2025-05-10", "2025-05-31", output / "families" / family)
    chosen_final = final_models[winner["family"]]
    chosen_path = models / "selected_model.joblib"
    joblib.dump(chosen_final, chosen_path, compress=3)
    reload_checks["selected"] = validate_saved_model(chosen_path, chosen_final)
    paths["selected"] = chosen_path.name
    final_models["xgboost"].estimator.save_model(models / "xgboost_native.json")
    with threadpool_limits(limits=2):
        export_prediction(chosen_final, "2025-05-10", "2025-05-31", output / "selected")
        export_prediction(final_models["xgboost"], "2025-05-10", "2025-05-31", output / "xgboost")
    frozen_hashes = {key: sha256_file(models / value) for key, value in paths.items()}
    freeze = {"selected_candidate": winner["name"], "selected_family": winner["family"], "model_files": paths, "model_sha256": frozen_hashes, "selection_file_sha256": selection_hash, "protocol_sha256": protocol_record["sha256"], "final_training_start": str(data.datetime.min()), "final_training_end": str(data.datetime.max()), "final_training_rows": len(data), "equipment": len(chosen_final.modules), "frozen_utc": datetime.now(timezone.utc).isoformat(), "frozen_before_current_run_reference_read": True, "evaluation_notice": NOTICE, "unit_contract": UNIT_CONTRACT}
    save_json(output / "model_freeze.json", freeze)
    save_json(output / "model_schema.json", {"selected": forecast_schema(chosen_final), "family_schemas": {k: forecast_schema(v) for k,v in final_models.items()}, "unit_contract": UNIT_CONTRACT})
    # Current-run first read of May values occurs after immutable model choice,
    # model serialization, reload verification, and all505 forecasts are saved.
    reference_hash_before = sha256_file(reference_path)
    reference = validate_reference(pd.read_csv(reference_path))
    for family, spec in benchmarks.items():
        measured, joined = evaluate_reference(native_predictions[family], reference)
        spec["may"] = measured
        joined.to_csv(output / f"may_comparison_{family}.csv", encoding="utf-8-sig")
        print(f"May retrospective {family}: native MAE={measured['mae_course_native']['hourly_pow']:.6f}, conditional kWh={measured['mae_kWh_if_reference_is_Wh']:.8f}", flush=True)
    _, selected_joined = evaluate_reference(native_predictions[winner["family"]], reference)
    selected_joined.to_csv(output / "may_comparison_selected.csv", encoding="utf-8-sig")
    for key, filename in paths.items():
        if sha256_file(models / filename) != frozen_hashes[key]:
            raise ValueError("Serialized model changed during May evaluation")
    if sha256_file(selection_path) != selection_hash or sha256_file(hourly_path) != source_hash_before or sha256_file(reference_path) != reference_hash_before:
        raise ValueError("Selection or input source changed during evaluation")
    chosen_results = benchmarks[winner["family"]]
    best_may_baseline = min(baselines.values(), key=lambda r: r["may"]["mae_course_native"]["hourly_pow"])
    may_selected = chosen_results["may"]["mae_course_native"]["hourly_pow"]
    may_baseline = best_may_baseline["may"]["mae_course_native"]["hourly_pow"]
    comparison_rows = [{"family": family, "candidate": spec["name"], "selected": family == winner["family"], "reference_only": spec["reference_only"], "march_mae_kWh": spec["march"]["focus"]["factory_mae_kWh"], "april_mae_kWh": spec["april"]["focus"]["factory_mae_kWh"], "may_mae_course_native": spec["may"]["mae_course_native"]["hourly_pow"], "may_mae_kWh_if_reference_is_Wh": spec["may"]["mae_kWh_if_reference_is_Wh"], "may_normalized_mae": spec["may"]["normalized_mae"]} for family, spec in benchmarks.items()]
    pd.DataFrame(comparison_rows).to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    summary = {"selected_family": winner["family"], "selected_candidate": winner["name"], "selected_parameters": winner["parameters"], "selection_policy": SELECTION_POLICY, "selected_march": chosen_results["march"], "selected_april": chosen_results["april"], "selected_may": chosen_results["may"], "family_best": family_best, "baselines": baselines, "march_candidates": records, "model_paths": paths, "metric_definitions": METRIC_DEFINITIONS, "unit_contract": UNIT_CONTRACT, "evaluation_notice": NOTICE, "success": {"final_model_is_ai": chosen_final.kind in FAMILIES, "functional_505_hours_13_equipment": len(native_predictions[winner["family"]]) == 505 and len(chosen_final.modules) == 13, "selected_basic_error_goal": chosen_results["may"]["basic_error_target_passed"], "selected_beats_best_may_baseline": bool(may_selected < may_baseline), "improvement_vs_best_may_baseline_percent": float((may_baseline-may_selected)/may_baseline*100), "baseline_comparison_did_not_change_selection": True, "official_physical_unit_acceptance": "pending provider confirmation"}, "reload_validation": reload_checks, "freeze_validation": {"model_hashes_unchanged": True, "selection_hash_unchanged": True, "hourly_source_unchanged": True, "reference_source_unchanged": True, "reference_sha256": reference_hash_before, "protocol_sha256": protocol_record["sha256"]}, "rows": {"training_dec_feb": len(train), "march_all_equipment_hours": len(march), "april_all_equipment_hours": len(april), "final_training": len(data), "may_factory_hours": 505, "may_equipment_hours": 505*13}, "elapsed_seconds": time.monotonic()-start}
    save_json(output / "summary.json", summary)
    print(json.dumps({"selected_family": summary["selected_family"], "selected_candidate": summary["selected_candidate"], "selected_may": summary["selected_may"], "success": summary["success"]}, ensure_ascii=False, indent=2), flush=True)
    return summary
