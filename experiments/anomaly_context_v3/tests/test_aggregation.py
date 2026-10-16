"""Behavioral checks for contiguous aggregation, not implementation mirroring."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_minute_data import MODULE, SENSORS, LINE_SENSORS, chunk_moments, merge_moments


def raw_frame(times, sensor_values=None):
    timestamps = pd.to_datetime(times)
    n = len(timestamps)
    frame = pd.DataFrame({
        MODULE: ["machine"] * n, "localtime": timestamps.strftime("%Y%m%d%H%M%S"),
        "operation": np.ones(n),
    })
    values = np.arange(n, dtype=float) if sensor_values is None else np.asarray(sensor_values, dtype=float)
    for sensor in SENSORS + LINE_SENSORS:
        frame[sensor] = values + 100
    return frame


class AggregationTests(unittest.TestCase):
    def test_chunk_boundary_preserves_mean_population_std_and_count(self):
        raw = raw_frame(pd.date_range("2025-01-01", periods=24, freq="5s"))
        pieces = [chunk_moments(raw.iloc[:7]), chunk_moments(raw.iloc[7:17]), chunk_moments(raw.iloc[17:])]
        actual = merge_moments(pd.concat(pieces, ignore_index=True))
        self.assertEqual(actual.n.tolist(), [12, 12])
        self.assertTrue(actual.complete_minute.all())
        np.testing.assert_allclose(actual.activePower_mean, [105.5, 117.5])
        np.testing.assert_allclose(actual.activePower_std, [np.std(np.arange(12))] * 2, rtol=1e-6)

    def test_duplicate_across_chunks_not_complete_even_with_twelve_rows(self):
        times = list(pd.date_range("2025-01-01", periods=11, freq="5s"))
        times.append(times[0])
        raw = raw_frame(times)
        actual = merge_moments(pd.concat([chunk_moments(raw.iloc[:6]), chunk_moments(raw.iloc[6:])]))
        self.assertEqual(int(actual.n.iloc[0]), 12)
        self.assertEqual(int(actual.duplicate_records.iloc[0]), 1)
        self.assertEqual(int(actual.missing_sample_slots.iloc[0]), 1)
        self.assertFalse(bool(actual.complete_minute.iloc[0]))

    def test_gaps_and_partial_endpoint_are_not_filled(self):
        times = list(pd.date_range("2025-01-01", periods=12, freq="5s"))
        times += [pd.Timestamp("2025-01-01 00:02")]
        actual = merge_moments(chunk_moments(raw_frame(times)))
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual.timestamp.dt.minute.tolist(), [0, 2])
        self.assertEqual(actual.complete_minute.tolist(), [True, False])
        self.assertEqual(actual.n.tolist(), [12, 1])

    def test_missing_sensor_is_not_zero_or_complete(self):
        raw = raw_frame(pd.date_range("2025-01-01", periods=12, freq="5s"))
        raw.loc[0, "activePower"] = np.nan
        actual = merge_moments(chunk_moments(raw))
        self.assertFalse(bool(actual.complete_minute.iloc[0]))
        self.assertTrue(bool(actual.has_missing_sensor.iloc[0]))
        self.assertAlmostEqual(float(actual.activePower_mean.iloc[0]), 106.0)

    def test_equipment_isolation_and_off_grid(self):
        one = raw_frame(pd.date_range("2025-01-01", periods=12, freq="5s"), np.zeros(12))
        two = raw_frame(pd.date_range("2025-01-01 00:00:01", periods=12, freq="5s"), np.ones(12) * 50)
        two[MODULE] = "second"
        actual = merge_moments(chunk_moments(pd.concat([one, two])))
        self.assertEqual(actual.activePower_mean.tolist(), [100, 150])
        self.assertEqual(actual.complete_minute.tolist(), [True, False])
        self.assertEqual(actual.off_grid_records.tolist(), [0, 12])


if __name__ == "__main__":
    unittest.main()
