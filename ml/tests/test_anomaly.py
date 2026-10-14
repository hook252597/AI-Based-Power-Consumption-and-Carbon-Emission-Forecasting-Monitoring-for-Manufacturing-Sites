import numpy as np
import pandas as pd
import joblib
import pytest
from sklearn.ensemble import IsolationForest

from manufacturing_ai.anomaly import FEATURES, MODULE, SEED, confusion, feature_matrix, isolation_score, metrics, model_identity, predict_observations, rule_parameters, rule_score, threshold_from_proxy, time_masks, verify_model_identity


def test_splits_are_disjoint_and_exclude_april30_partial():
    times = np.array([20241130235955, 20241201000000, 20250228235955, 20250301000000, 20250331235955, 20250401000000, 20250429235955, 20250430000000, 20250501000000])
    train, tune, test = time_masks(times)
    assert np.flatnonzero(train).tolist() == [1, 2]
    assert np.flatnonzero(tune).tolist() == [3, 4]
    assert np.flatnonzero(test).tolist() == [5, 6]
    assert not (train & tune | tune & test | train & test).any()


def test_label_fields_cannot_enter_features():
    df = pd.DataFrame({f: [1.0, 2.0] for f in FEATURES})
    original = feature_matrix(df)
    df["is_outlier"] = [True, False]
    df["outlier_reason"] = ["voltage", "normal"]
    df["localtime"] = [20250301000000, 20250301000005]
    np.testing.assert_array_equal(original, feature_matrix(df))
    assert feature_matrix(df).shape == (2, 10)


def test_isolation_score_direction_and_anomaly_detection():
    rng = np.random.default_rng(24)
    normal = rng.normal(0, 1, (1000, 2)).astype(np.float32)
    model = IsolationForest(n_estimators=40, random_state=24).fit(normal)
    normal_scores = isolation_score(model, normal)
    extreme_scores = isolation_score(model, np.full((5, 2), 25, dtype=np.float32))
    assert extreme_scores.min() > np.quantile(normal_scores, 0.99)


def test_threshold_selection_is_march_input_only_and_uses_ge():
    labels = np.array([False, False, True, True])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, selection = threshold_from_proxy(labels, scores)
    assert threshold == 0.8
    assert selection == "march_proxy_max_f1"
    assert metrics(confusion(labels, scores >= threshold))["f1"] == 1


def test_no_positive_proxy_uses_declared_score_reference():
    threshold, selection = threshold_from_proxy(np.zeros(100, dtype=bool), np.arange(100))
    assert threshold == np.quantile(np.arange(100), 0.999)
    assert selection == "unlabeled_equipment_march_score_q999"


def test_rules_fit_training_and_handle_constant_features():
    train = np.column_stack([np.arange(100), np.ones(100)])
    params = rule_parameters(train)
    assert np.isfinite(rule_score(train, params)).all()
    assert rule_score(np.array([[1000, 1]]), params)[0] > rule_score(train, params).max()


def test_metrics_zero_denominators_are_defined():
    result = metrics(confusion([False, False], [False, False]))
    assert result["rows"] == 2
    assert result["precision"] == result["recall"] == result["f1"] == 0


def test_saved_bundle_scores_unseen_measurements_without_labels(tmp_path):
    rng = np.random.default_rng(100)
    x = rng.normal(0, 1, (500, len(FEATURES))).astype(np.float32)
    model = IsolationForest(n_estimators=30, random_state=100).fit(x)
    rules = rule_parameters(x)
    bundle = {"features": FEATURES, "model": model, "rules": rules, "thresholds": {"isolation_forest": {"threshold": float(np.quantile(isolation_score(model, x), .99))}, "rules": {"threshold": float(np.quantile(rule_score(x, rules), .99))}}}
    destination = tmp_path / "bundle.joblib"
    joblib.dump(bundle, destination)
    new = pd.DataFrame([np.zeros(len(FEATURES)), np.full(len(FEATURES), 100)], columns=FEATURES)
    new[MODULE] = "machine"
    new["localtime"] = [20250501000000, 20250501000005]
    predicted = predict_observations(new, {"machine": joblib.load(destination)})
    assert predicted["isolation_forest_anomaly"].tolist() == [False, True]
    assert predicted["rules_anomaly"].tolist() == [False, True]
    new["is_outlier"] = [True, False]
    pd.testing.assert_frame_equal(predicted, predict_observations(new, {"machine": bundle}))
    new[MODULE] = "unknown"
    with pytest.raises(ValueError, match="No fitted anomaly model"):
        predict_observations(new, {"machine": bundle})


def test_cached_model_rejects_changed_source_and_seed():
    manifest = {"signature": {"raw_size": 100, "monthly_sample": 10}, "verification": {"sha256_parsed_selected_columns": "first-source"}, "samples": {"machine": {"train": 30, "tune": 10}}}
    identity = model_identity(manifest, 100, 2048)
    bundle = {"training_identity": identity}
    verify_model_identity(bundle, identity)
    with pytest.raises(ValueError, match="training identity differs"):
        verify_model_identity(bundle, model_identity(manifest, 100, 2048, seed=SEED + 1))
    manifest["verification"]["sha256_parsed_selected_columns"] = "changed-source"
    with pytest.raises(ValueError, match="training identity differs"):
        verify_model_identity(bundle, model_identity(manifest, 100, 2048))
    with pytest.raises(ValueError, match="training identity differs"):
        verify_model_identity({}, identity)
