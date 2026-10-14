import joblib
import numpy as np
import pandas as pd
import pytest

from manufacturing_ai.forecast import MODULE, ForecastModel, calendar_features, grid_for_dates
from manufacturing_ai.forecast_v2 import (FAMILIES, candidate_specs, evaluate_reference,
                                         fit_ai_model, select_ai_candidate,
                                         validate_io_paths, validate_training_grid)


def small_training():
    data = grid_for_dates("2025-01-01", "2025-01-03 23:00", ["equipment-a", "equipment-b"])
    data["energy_kWh"] = data[MODULE].map({"equipment-a": 3.0, "equipment-b": 8.0}) + data.datetime.dt.hour * .01
    return data


def test_grid_is_four_ai_families_and_no_random_hist_validation():
    specs = candidate_specs()
    assert len(specs) == 12
    assert {c["family"] for c in specs} == set(FAMILIES)
    assert all(not c["reference_only"] for c in specs)
    assert all(c["parameters"]["early_stopping"] is False for c in specs if c["family"] == "hist_gradient_boosting")


def test_ai_selection_never_chooses_better_mean_or_may_winner():
    def record(name, family, march, may, reference=False):
        return {"name": name, "family": family, "reference_only": reference, "march": {"focus": {"factory_mae_kWh": march}}, "may": {"mae": may}}
    items = [record("mean", "mean", 0, 0, True), record("xgb", "xgboost", .3, .01), record("forest", "random_forest", .2, 1.0)]
    assert select_ai_candidate(items)["name"] == "forest"
    with pytest.raises(ValueError, match="No AI"):
        select_ai_candidate(items[:1])


@pytest.mark.parametrize("family", FAMILIES)
def test_all_families_serialize_as_compatible_forecastmodel(tmp_path, family):
    params = {
        "xgboost": {"n_estimators": 8, "max_depth": 2, "learning_rate": .2},
        "random_forest": {"n_estimators": 8, "max_depth": 4, "min_samples_leaf": 2},
        "extra_trees": {"n_estimators": 8, "max_depth": 4, "min_samples_leaf": 2},
        "hist_gradient_boosting": {"max_iter": 8, "min_samples_leaf": 5, "early_stopping": False},
    }[family]
    data = small_training()
    model = fit_ai_model(data, family, params)
    assert isinstance(model, ForecastModel)
    assert model.kind == family
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    future = grid_for_dates("2025-05-10", "2025-05-10 03:00", model.modules)
    np.testing.assert_array_equal(model.predict(future), loaded.predict(future))
    original = model.predict(future)
    future["activePower"] = 10000000
    future["energy_kWh"] = -999
    future["hourly_pow"] = -999
    np.testing.assert_array_equal(original, model.predict(future))
    assert np.isfinite(original).all() and (original >= 0).all()
    future[MODULE] = "unknown"
    with pytest.raises(ValueError, match="Unknown equipment"):
        loaded.predict(future)


def test_native_and_physical_reference_mae_are_not_confused():
    index = pd.date_range("2025-05-10", "2025-05-31", freq="h")
    actual = pd.DataFrame({"hourly_pow": 1000., "may_bill": 180000., "may_carbon": 424.}, index=index)
    pred = pd.DataFrame({"hourly_pow": 1100., "may_bill": 198000., "may_carbon": 466.4}, index=index)
    result, _ = evaluate_reference(pred, actual)
    assert result["rows"] == 505
    assert result["mae_course_native"]["hourly_pow"] == 100
    assert result["mae_kWh_if_reference_is_Wh"] == .1
    assert result["normalized_mae"] == .1
    assert result["unit_confirmation_pending"]


def test_incomplete_or_offhour_calendar_is_rejected():
    data = grid_for_dates("2024-12-01", "2025-04-29 23:00", [f"module-{i}" for i in range(13)])
    validate_training_grid(data)
    with pytest.raises(ValueError, match="3600 complete"):
        validate_training_grid(data.iloc[13:])
    with pytest.raises(ValueError, match="3600 complete"):
        validate_training_grid(data.iloc[:-13])
    shifted = data.copy()
    shifted["datetime"] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="3600 complete"):
        validate_training_grid(shifted)


def test_paths_cannot_overwrite_inputs_or_v1_models(tmp_path):
    hourly, reference = tmp_path / "source/hourly.csv", tmp_path / "source/reference.csv"
    output, models = tmp_path / "outputs/forecast_v2", tmp_path / "models/forecast_v2"
    validate_io_paths(hourly, reference, output, models, project_root=tmp_path)
    with pytest.raises(ValueError, match="input source"):
        validate_io_paths(output / "comparison.csv", reference, output, models, project_root=tmp_path)
    with pytest.raises(ValueError, match="nonoverlapping"):
        validate_io_paths(hourly, reference, output, output / "models", project_root=tmp_path)
    with pytest.raises(ValueError, match="protected"):
        validate_io_paths(hourly, reference, output, tmp_path / "models/forecast", project_root=tmp_path)
