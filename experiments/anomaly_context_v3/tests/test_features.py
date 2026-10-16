"""Causality, isolation, and data completeness checks; no source labels needed."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_features import ALL_FEATURES, CONTEXT_FEATURES, SENSORS, STATIC_FEATURES, build_features


def minute_frame(count=15, start="2025-04-01 10:00", module="A"):
    data = {"module": [module] * count, "timestamp": pd.date_range(start, periods=count, freq="min"), "n": [12] * count}
    for index, sensor in enumerate(SENSORS):
        data[f"{sensor}_mean"] = 100.0 + index + np.arange(count, dtype=float)
        data[f"{sensor}_std"] = np.ones(count)
    return pd.DataFrame(data)


class FeaturesTest(unittest.TestCase):
    def test_feature_contract_and_causal_window(self):
        source = minute_frame()
        features = build_features(source)
        self.assertEqual(len(STATIC_FEATURES), 34)
        self.assertEqual(len(CONTEXT_FEATURES), 30)
        self.assertEqual(len(set(ALL_FEATURES)), 64)
        self.assertEqual(list(features.columns), ALL_FEATURES + ["valid"])
        self.assertEqual(features.valid.tolist(), [False] * 5 + [True] * 10)
        self.assertEqual(features.loc[5, "activePower_delta_prev"], 1)
        self.assertEqual(features.loc[5, "activePower_delta_past5_mean"], 3)
        self.assertAlmostEqual(features.loc[5, "activePower_past5_std"], np.std(np.arange(5)))

    def test_future_changes_cannot_change_prefix(self):
        source = minute_frame(20)
        prefix = build_features(source.iloc[:11])
        future_changed = source.copy()
        future_changed.loc[11:, [f"{s}_mean" for s in SENSORS]] = 1e9
        full = build_features(future_changed)
        assert_frame_equal(prefix, full.iloc[:11])

    def test_current_minute_does_not_change_past_window(self):
        source = minute_frame()
        altered = source.copy()
        altered.loc[8, "activePower_mean"] = 900
        original = build_features(source)
        changed = build_features(altered)
        self.assertEqual(original.loc[8, "activePower_past5_std"], changed.loc[8, "activePower_past5_std"])
        self.assertEqual(changed.loc[8, "activePower_delta_past5_mean"], 900 - np.mean(np.arange(103, 108)))

    def test_gap_resets_history(self):
        source = minute_frame(20).drop(index=8)
        result = build_features(source)
        self.assertTrue(result.loc[7, "valid"])
        self.assertFalse(result.loc[9:13, "valid"].any())
        self.assertTrue(result.loc[14, "valid"])
        self.assertTrue(pd.isna(result.loc[9, "activePower_delta_prev"]))

    def test_incomplete_minute_resets_history(self):
        source = minute_frame(20)
        source.loc[8, "n"] = 11
        result = build_features(source)
        self.assertFalse(result.loc[8:13, "valid"].any())
        self.assertTrue(result.loc[14, "valid"])
        self.assertEqual(result.loc[8, "activePower_mean"], source.loc[8, "activePower_mean"])

    def test_twelve_rows_with_incomplete_phase_reset_history(self):
        source = minute_frame(20)
        source["complete_minute"] = True
        source.loc[8, "complete_minute"] = False
        self.assertEqual(source.loc[8, "n"], 12)
        result = build_features(source)
        self.assertTrue(result.loc[7, "valid"])
        self.assertFalse(result.loc[8:13, "valid"].any())
        self.assertTrue(result.loc[14, "valid"])

    def test_missing_completeness_is_not_assumed_complete(self):
        source = minute_frame(20)
        source["complete_minute"] = pd.Series([True] * len(source), dtype="boolean")
        source.loc[8, "complete_minute"] = pd.NA
        result = build_features(source)
        self.assertFalse(result.loc[8:13, "valid"].any())
        self.assertTrue(result.loc[14, "valid"])

    def test_equipment_column_alias_and_conflict(self):
        source = minute_frame()
        aliased = source.rename(columns={"module": "module(equipment)"})
        assert_frame_equal(build_features(source), build_features(aliased))
        aliased["module"] = "B"
        with self.assertRaises(ValueError):
            build_features(aliased)

    def test_missing_sensor_is_not_zero_filled(self):
        source = minute_frame(20)
        source.loc[8, "voltageR_mean"] = np.nan
        result = build_features(source)
        self.assertTrue(pd.isna(result.loc[8, "voltageR_mean"]))
        self.assertTrue(pd.isna(result.loc[8, "voltage_relative_range"]))
        self.assertFalse(result.loc[8:13, "valid"].any())
        self.assertTrue(result.loc[14, "valid"])

    def test_midnight_resets_history(self):
        result = build_features(minute_frame(20, start="2025-04-01 23:50"))
        self.assertTrue(result.loc[9, "valid"])
        self.assertFalse(result.loc[10:14, "valid"].any())
        self.assertTrue(result.loc[15, "valid"])

    def test_equipment_isolation_and_input_order(self):
        a = minute_frame(module="A")
        b = minute_frame(module="B")
        b[[f"{s}_mean" for s in SENSORS]] *= 1000
        combined = pd.concat([a, b], ignore_index=True).sample(frac=1, random_state=5)
        output = build_features(combined)
        self.assertTrue(output.index.equals(combined.index))
        assert_frame_equal(output.loc[a.index], build_features(a))
        self.assertTrue(output.loc[20, "valid"])
        self.assertEqual(output.loc[20, "activePower_delta_prev"], 1000)

    def test_duplicate_input_index_is_preserved(self):
        source = minute_frame(12)
        source.index = ["same"] * len(source)
        result = build_features(source)
        self.assertTrue(result.index.equals(source.index))
        self.assertEqual(int(result.valid.sum()), 7)

    def test_labels_have_no_effect(self):
        source = minute_frame()
        source["is_outlier"] = np.arange(len(source)) % 2
        assert_frame_equal(build_features(source), build_features(source.drop(columns="is_outlier")))

    def test_all_zero_observed_phases_are_valid(self):
        source = minute_frame()
        for sensor in SENSORS:
            source[f"{sensor}_mean"] = 0.0
            source[f"{sensor}_std"] = 0.0
        result = build_features(source)
        self.assertTrue(result.loc[5:, "valid"].all())
        self.assertEqual(result.loc[5, "current_relative_range"], 0.0)
        self.assertEqual(result.loc[5, "power_minus_vi_pf"], 0.0)

    def test_zero_mean_nonzero_spread_is_undefined(self):
        source = minute_frame()
        source["currentR_mean"] = -1.0
        source["currentS_mean"] = 0.0
        source["currentT_mean"] = 1.0
        result = build_features(source)
        self.assertTrue(result.current_relative_range.isna().all())
        self.assertFalse(result.valid.any())

    def test_infinite_and_negative_std_are_invalid(self):
        source = minute_frame(20)
        source.loc[7, "currentR_mean"] = np.inf
        source.loc[14, "voltageT_std"] = -1
        result = build_features(source)
        self.assertFalse(result.loc[7:12, "valid"].any())
        self.assertTrue(result.loc[13, "valid"])
        self.assertFalse(result.loc[14:, "valid"].any())

    def test_invalid_timestamps_and_duplicates_are_rejected(self):
        for case in ("duplicate", "off_minute", "aware", "missing"):
            with self.subTest(case=case):
                source = minute_frame()
                if case == "duplicate": source.loc[1, "timestamp"] = source.loc[0, "timestamp"]
                if case == "off_minute": source.loc[1, "timestamp"] += pd.Timedelta(seconds=5)
                if case == "aware": source["timestamp"] = source.timestamp.dt.tz_localize("Asia/Seoul")
                if case == "missing": source.loc[1, "timestamp"] = pd.NaT
                with self.assertRaises(ValueError): build_features(source)

    def test_empty_frame(self):
        result = build_features(minute_frame().iloc[:0])
        self.assertEqual(result.shape, (0, 65))
        self.assertEqual(str(result.valid.dtype), "bool")


if __name__ == "__main__":
    unittest.main()
