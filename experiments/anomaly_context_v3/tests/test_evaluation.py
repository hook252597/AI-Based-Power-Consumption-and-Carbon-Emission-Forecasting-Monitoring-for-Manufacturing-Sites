from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation import (SENSORS, PROBE_KINDS, group_alert_events, summarize_alerts,
    fixed_rule_flags, assert_aggregate_consistency, generate_synthetic_probes,
    evaluate_synthetic_probe, summarize_synthetic_results)


def minutes(n=12):
    return pd.DataFrame({'module': ['1'] * n, 'timestamp': pd.date_range('2025-04-01', periods=n, freq='min'), 'n': [12] * n})


def sensors(n=1200, modules=('1', '2')):
    rows = []
    for module in modules:
        df = minutes(n)
        df['module'] = module
        df['operation_mean'] = 1.0
        for sensor in SENSORS:
            value = 225.0 if sensor.startswith('voltage') else 95.0 if sensor.startswith('powerFactor') else 30.0 if sensor.startswith('current') else 4000.0
            df[f'{sensor}_mean'] = value
            df[f'{sensor}_std'] = 0.2
            df[f'{sensor}_min'] = value - 1
            df[f'{sensor}_max'] = value + 1
        for phase in ('RS', 'ST', 'TR'):
            df[f'voltage{phase}_min'] = 380.0
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


class AlertEventsTests(unittest.TestCase):
    def test_three_minutes_required_and_causal_decision(self):
        df = minutes(8)
        events = group_alert_events(df, [1,1,1,1,0,1,1,0])
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]['duration_minutes'], 4)
        self.assertEqual(events.iloc[0]['decision_time'], pd.Timestamp('2025-04-01 00:03'))
        self.assertEqual(events.iloc[0]['end'], pd.Timestamp('2025-04-01 00:04'))
        self.assertTrue(group_alert_events(df.iloc[:2], [1,1]).empty)
        self.assertEqual(group_alert_events(df.iloc[:3], [1,1,1]).iloc[0]['decision_time'], events.iloc[0]['decision_time'])

    def test_missing_minute_breaks_and_module_separated(self):
        df = minutes(7).drop(index=3)
        events = group_alert_events(df, np.ones(len(df), dtype=bool))
        self.assertEqual(len(events), 2)
        second = df.copy(); second['module'] = '2'
        both = pd.concat([df,second], ignore_index=True)
        self.assertEqual(len(group_alert_events(both, [1]*len(both))), 4)

    def test_incomplete_minute_breaks(self):
        df = minutes(7); df.loc[3, 'n'] = 11
        events = group_alert_events(df, [1]*7)
        self.assertEqual(events['duration_minutes'].tolist(), [3,3])
        summary = summarize_alerts(df, [1]*7)
        self.assertEqual(summary['flagged_minutes'], 6)
        self.assertEqual(summary['complete_minutes'], 6)
        self.assertEqual(summary['observed_samples'], 83)
        self.assertEqual(summary['alert_count'], 2)
        self.assertEqual(summary['alerts_per_24_equipment_hours'], 480)

    def test_complete_minute_false_breaks_even_with_twelve_records(self):
        df = minutes(7).rename(columns={'module':'module(equipment)'})
        df['complete_minute'] = True; df.loc[3,'complete_minute'] = False
        events = group_alert_events(df, [1]*7)
        self.assertEqual(events['duration_minutes'].tolist(), [3,3])
        self.assertEqual(summarize_alerts(df,[1]*7)['complete_minutes'], 6)

    def test_unsorted_input_flags_stay_row_aligned(self):
        df = minutes(5).iloc[[4,0,3,1,2]]
        events = group_alert_events(df, [0,1,0,1,1])
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]['start'], pd.Timestamp('2025-04-01'))

    def test_duplicate_or_wrong_timestamp_rejected(self):
        df = minutes(3); df.loc[1,'timestamp'] = df.loc[0,'timestamp']
        with self.assertRaises(ValueError): group_alert_events(df, [1]*3)
        df = minutes(3); df.loc[1,'timestamp'] += pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError): group_alert_events(df, [1]*3)
        with self.assertRaises(ValueError): group_alert_events(minutes(), [1])

    def test_empty_result_has_safe_summary(self):
        df = minutes(0)
        self.assertTrue(group_alert_events(df, []).empty)
        self.assertIsNone(summarize_alerts(df, [])['alerts_per_24_equipment_hours'])


class SyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = sensors()
        cls.original = cls.source.copy(deep=True)
        cls.scales = {module: {sensor: 1.0 for sensor in SENSORS} for module in ('1','2')}
        cls.probes = generate_synthetic_probes(cls.source, cls.scales, max_per_type=2)

    def test_five_kinds_balanced_and_original_unchanged(self):
        self.assertEqual(len(self.probes), 10)
        self.assertEqual(set(p['kind'] for p in self.probes), set(PROBE_KINDS))
        for kind in PROBE_KINDS:
            self.assertEqual(set(p['module'] for p in self.probes if p['kind'] == kind), {'1','2'})
        pd.testing.assert_frame_equal(self.source, self.original)

    def test_aggregate_bounds_rule_safety_and_no_pre_injection_changes(self):
        for probe in self.probes:
            base, injected = probe['baseline'], probe['injected']
            assert_aggregate_consistency(injected, require_rule_safe=True)
            self.assertFalse(fixed_rule_flags(injected).any())
            pd.testing.assert_frame_equal(base.iloc[:60], injected.iloc[:60])
            pd.testing.assert_frame_equal(base.iloc[90:], injected.iloc[90:])
            self.assertFalse(base.iloc[60:90].equals(injected.iloc[60:90]))
            self.assertEqual(probe['truth']['end'] - probe['truth']['start'], pd.Timedelta(minutes=30))
            self.assertIn('One-minute aggregate', probe['notice'])

    def test_deterministic_and_no_windows_overlap(self):
        repeat = generate_synthetic_probes(self.source, self.scales, max_per_type=2)
        for first, second in zip(self.probes, repeat):
            self.assertEqual(first['truth'], second['truth'])
            pd.testing.assert_frame_equal(first['injected'], second['injected'])
        for i, first in enumerate(self.probes):
            for second in self.probes[i+1:]:
                if first['module'] == second['module']:
                    self.assertFalse(first['baseline']['timestamp'].isin(second['baseline']['timestamp']).any())

    def test_three_minute_delay_and_baseline_contamination_reported(self):
        probe = self.probes[0]
        base = np.zeros(120, dtype=bool); inject = base.copy(); inject[60:90] = True
        result = evaluate_synthetic_probe(probe, base, inject)
        self.assertTrue(result['detected_during_injection'])
        self.assertEqual(result['first_decision_delay_minutes'], 3)
        self.assertTrue(result['incremental_detection_on_clean_control'])
        base[58:90] = True
        result = evaluate_synthetic_probe(probe, base, inject)
        self.assertTrue(result['baseline_already_alerting'])
        self.assertFalse(result['incremental_detection_on_clean_control'])

    def test_post_injection_decision_does_not_count_but_last_minute_end_does(self):
        probe = self.probes[0]; base = np.zeros(120,dtype=bool)
        late = base.copy(); late[89:92] = True
        self.assertFalse(evaluate_synthetic_probe(probe,base,late)['detected_during_injection'])
        immediate = base.copy(); immediate[60:90] = True
        result = evaluate_synthetic_probe(probe,late,immediate)
        self.assertFalse(result['baseline_already_alerting'])
        self.assertTrue(result['incremental_detection_on_clean_control'])
        last = base.copy(); last[87:90] = True
        self.assertTrue(evaluate_synthetic_probe(probe,base,last)['detected_during_injection'])
        earlier = base.copy(); earlier[55:90] = True
        self.assertFalse(evaluate_synthetic_probe(probe,base,earlier)['detected_during_injection'])

    def test_recall_is_synthetic_only_and_clean_controls_separate(self):
        base = np.zeros(120,dtype=bool); inject = base.copy(); inject[60:90] = True
        clean = evaluate_synthetic_probe(self.probes[0],base,inject)
        noisy = evaluate_synthetic_probe(self.probes[1],inject,inject)
        result = summarize_synthetic_results([clean,noisy])
        self.assertEqual(result['synthetic_event_recall'],1.0)
        self.assertEqual(result['clean_control_count'],1)
        self.assertEqual(result['baseline_already_alerting_count'],1)
        self.assertIn('not a raw 5-second',result['notice'])

    def test_clipped_voltage_lower_bound_has_documented_upward_alternative(self):
        df = sensors(1200,('1',))
        for phase in 'RST': df[f'voltage{phase}_min'] = 210.0
        audit = {}
        probes = generate_synthetic_probes(df,self.scales,max_per_type=1,generation_audit=audit)
        self.assertEqual(len(probes),5)
        self.assertGreater(audit['by_kind']['level_shift']['original_direction_rejected'],0)
        self.assertFalse(audit['model_scores_inspected'])
        for probe in probes:
            before,after = probe['baseline'].iloc[60:90],probe['injected'].iloc[60:90]
            if probe['kind'] in ('level_shift','persistent_drift','phase_imbalance'):
                self.assertTrue((after['voltageR_mean'] > before['voltageR_mean']).all())
                self.assertTrue((after['voltageR_max'] < 300).all())
            if probe['kind'] == 'phase_imbalance':
                pd.testing.assert_series_equal(before['voltageS_mean'],after['voltageS_mean'])
            if probe['kind'] == 'volatility_increase':
                self.assertTrue((after['activePower_std'] > before['activePower_std']).all())
                self.assertTrue((after['currentR_min'] >= 0).all())
            assert_aggregate_consistency(probe['injected'],require_rule_safe=True)

    def test_float32_source_is_preserved_without_assignment_warning(self):
        import warnings
        df = sensors(1200,('1',))
        float_columns = df.select_dtypes(include='floating').columns
        df = df.astype({c:'float32' for c in float_columns})
        original = df.copy(deep=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error',FutureWarning)
            probes = generate_synthetic_probes(df,self.scales,max_per_type=1)
        self.assertEqual(len(probes),5)
        pd.testing.assert_frame_equal(df,original)
        for probe in probes:
            pd.testing.assert_frame_equal(probe['baseline'].iloc[:60],probe['injected'].iloc[:60])

    def test_skipped_probes_leave_audit(self):
        audit = {}
        probes = generate_synthetic_probes(sensors(100,('1',)),self.scales,max_per_type=1,generation_audit=audit)
        self.assertEqual(probes,[])
        self.assertEqual(audit['candidate_limit_per_equipment_kind'],500)
        self.assertTrue(all(item['skipped']==1 for item in audit['by_kind'].values()))

    def test_incomplete_minute_excluded_from_probe_selection(self):
        df = sensors(119,('1',)).rename(columns={'module':'module(equipment)'})
        df['complete_minute'] = True
        self.assertEqual(generate_synthetic_probes(df,self.scales,max_per_type=1),[])
        df = sensors(120,('1',)).rename(columns={'module':'module(equipment)'})
        df['complete_minute'] = True; df.loc[90,'complete_minute'] = False
        self.assertEqual(generate_synthetic_probes(df,self.scales,max_per_type=1),[])

    def test_inconsistent_standard_deviation_rejected(self):
        df = sensors(4,('1',)); df['voltageR_std'] = 100
        with self.assertRaises(ValueError): assert_aggregate_consistency(df)
        df = sensors(4,('1',)); df['powerFactorR_max'] = 101
        with self.assertRaises(ValueError): assert_aggregate_consistency(df)


if __name__ == '__main__':
    unittest.main()
