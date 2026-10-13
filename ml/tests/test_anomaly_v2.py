import json
import numpy as np
import pandas as pd
import pytest

from manufacturing_ai.anomaly import FEATURES, MODULE
from manufacturing_ai.anomaly_v2 import (calibrated_threshold, choose_candidate,
    documented_rule, march_masks, predict_observations, run, write_protocol)


def test_march_calibration_selection_are_disjoint():
    t = np.array([20250228235955, 20250301000000, 20250315235955, 20250316000000, 20250331235955, 20250401000000])
    calibration, selection = march_masks(t)
    assert np.flatnonzero(calibration).tolist() == [1, 2]
    assert np.flatnonzero(selection).tolist() == [3, 4]
    assert not (calibration & selection).any()


def test_no_positive_threshold_is_finite_and_keeps_detector_active():
    score = np.linspace(0, 10, 10000)
    threshold, policy = calibrated_threshold(np.zeros(len(score), bool), score, 1, .99999)
    assert np.isfinite(threshold)
    assert 9.9 < threshold < 10
    assert 11 >= threshold
    assert policy == "finite_calibration_alarm_budget"


def test_positive_threshold_detects_boundary_with_greater_equal():
    y = np.array([False, False, True, True])
    score = np.array([.1, .2, .8, .9])
    threshold, _ = calibrated_threshold(y, score, 2, .99999)
    np.testing.assert_array_equal(score >= threshold, y)


def test_reference_and_april_metrics_cannot_choose_candidate():
    def record(name, f1, reference=False, april_f1=0):
        return {"name": name, "stage": 1, "reference_only": reference, "march_selection": {"precision": f1, "recall": f1, "f1": f1}, "april": {"f1": april_f1}}
    records = [record("reference", 1, True), record("march_winner", .95, april_f1=.1), record("april_winner", .85, april_f1=1)]
    assert choose_candidate(records)["name"] == "march_winner"


def test_protocol_refuses_changed_identity(tmp_path):
    path = tmp_path / "protocol.json"
    first = write_protocol(path, {"source": "one", "seed": 4})
    assert write_protocol(path, {"source": "one", "seed": 4}) == first
    with pytest.raises(ValueError, match="differs"):
        write_protocol(path, {"source": "two", "seed": 4})


def test_missing_prior_cache_does_not_trigger_undeclared_source_scan(tmp_path):
    with pytest.raises(ValueError, match="requires the previously verified V1 cache"):
        run(tmp_path)


def test_documented_rule_strict_boundaries():
    x = np.full((3, len(FEATURES)), 1000, dtype=float)
    for f in ("voltageR", "voltageS", "voltageT"):
        x[:, FEATURES.index(f)] = 210
    for f in ("powerFactorR", "powerFactorS", "powerFactorT"):
        x[:, FEATURES.index(f)] = 85
    x[1, FEATURES.index("voltageR")] = 209.99
    x[2, FEATURES.index("powerFactorS")] = 84.99
    np.testing.assert_array_equal(documented_rule(x), [False, True, True])


def test_inference_ignores_labels_and_preserves_input_order():
    frame = pd.DataFrame(np.zeros((2, len(FEATURES))), columns=FEATURES, index=[8, 4])
    frame[MODULE] = ["b", "a"]
    frame["localtime"] = [20250501000000, 20250501000005]
    frame.loc[4, "voltageR"] = 9
    equipment = {m: {"detectors": {"robust": {"center": np.zeros(len(FEATURES)), "scale": np.ones(len(FEATURES))}}, "threshold": 5} for m in ("a", "b")}
    bundle = {"candidate": "test", "method": "robust_distance", "equipment": equipment}
    result = predict_observations(frame, bundle)
    assert result.index.tolist() == [8, 4]
    assert result["is_anomaly"].tolist() == [False, True]
    shuffled = predict_observations(frame.iloc[::-1], bundle).loc[result.index]
    pd.testing.assert_frame_equal(result, shuffled)
    frame["is_outlier"] = [True, False]
    frame["outlier_reason"] = ["voltage", "normal"]
    pd.testing.assert_frame_equal(result, predict_observations(frame, bundle))
    frame[MODULE] = "unknown"
    with pytest.raises(ValueError, match="No fitted V2 detector"):
        predict_observations(frame, bundle)
