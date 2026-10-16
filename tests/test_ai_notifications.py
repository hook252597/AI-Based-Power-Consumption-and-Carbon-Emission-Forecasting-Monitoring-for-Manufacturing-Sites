"""Behavioral tests for causal policy replay and immutable model detections."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from monitoring.ai_notifications import (DEFAULT_POLICY, JSON_HASH_METHOD, json_content_sha256, build_detection_samples,
    evaluate_notifications, get_ai_notifications, normalize_policy)
from monitoring.alert_service import MODULE, sha256


def raw(scores, flags=None, seconds=None, start='2025-04-01 00:00:00'):
    seconds = list(range(0, 5 * len(scores), 5)) if seconds is None else seconds
    return pd.DataFrame({MODULE: '13(3호기)', 'datetime': pd.Timestamp(start) + pd.to_timedelta(seconds, unit='s'),
                         'ai_score': scores, 'ai_flag': [True] * len(scores) if flags is None else flags})


def sample(frame):
    return build_detection_samples(frame, {'13(3호기)': 0.6})


def bursts(starts, max_seconds=None, score=.7):
    last = max_seconds if max_seconds is not None else max(starts)
    seconds = list(range(0, last + 1, 5))
    return sample(raw([score] * len(seconds), [s in starts for s in seconds], seconds))


def test_every_ai_detection_retained_even_below_new_score_gate():
    data = sample(raw([.61, .9, .7], [True, False, True]))
    result = evaluate_notifications(data)
    assert len(data) == 2
    assert result['summary']['ai_record_count'] == 2
    assert result['summary']['qualified_record_count'] == 1
    assert result['summary']['original_event_count'] == 2


def test_source_and_policy_unchanged():
    frame = raw([.7] * 12)
    original = frame.copy(deep=True)
    data = sample(frame)
    before = data.copy(deep=True)
    policy = dict(DEFAULT_POLICY)
    evaluate_notifications(data, policy)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(data, before)
    assert policy == DEFAULT_POLICY


def test_score_gate_inclusive_exact_boundary():
    result = evaluate_notifications(sample(raw([.62] * 12)))
    assert result['summary']['qualified_record_count'] == 12
    assert result['notifications'][0]['first_trigger_at'] == '2025-04-01T00:01:00'


def test_duration_requires_full_completed_60_seconds():
    data = sample(raw([.7] * 12))
    assert not evaluate_notifications(data, cutoff='2025-04-01 00:00:59')['notifications']
    assert evaluate_notifications(data, cutoff='2025-04-01 00:01:00')['notifications'][0]['first_trigger_at'].endswith('00:01:00')


def test_one_high_point_in_long_original_event_is_not_long_high_score():
    result = evaluate_notifications(sample(raw([.61] * 5 + [.8] + [.61] * 20)))
    assert result['all_logs'][0]['duration_seconds'] == 130
    assert result['all_logs'][0]['max_qualifying_duration_seconds'] == 5
    assert not result['notifications']


def test_normal_sample_breaks_duration():
    result = evaluate_notifications(sample(raw([.7] * 13, [True] * 6 + [False] + [True] * 6)))
    assert len(result['all_logs']) == 2
    assert not result['notifications']


def test_missing_raw_observation_breaks_duration_and_repeat():
    frame = raw([.7] * 12, seconds=list(range(0, 30, 5)) + list(range(35, 65, 5)))
    data = sample(frame)
    assert data.coverage_id.nunique() == 2
    assert not evaluate_notifications(data, {'repeat_count': 2})['notifications']


def test_score_oscillation_within_one_original_event_is_not_repetition():
    result = evaluate_notifications(sample(raw([.7, .61] * 30)), {'repeat_count': 2})
    assert len(result['all_logs']) == 1
    assert result['all_logs'][0]['max_qualifying_duration_seconds'] == 5
    assert not result['notifications']


def test_repeat_trigger_on_third_distinct_event_after_sample_end():
    data = bursts([0, 20, 40])
    result = evaluate_notifications(data)
    assert result['notifications'][0]['first_trigger_at'] == '2025-04-01T00:00:45'
    assert result['summary']['notified_original_event_count'] == 3
    assert result['trigger_candidates'][0]['repeat_event_count'] == 3
    assert not evaluate_notifications(data, cutoff='2025-04-01 00:00:44')['notifications']


def test_repeat_window_expiration_and_inclusive_boundary():
    assert evaluate_notifications(bursts([0, 30, 60]), {'repeat_window_minutes': 1})['notifications']
    assert not evaluate_notifications(bursts([0, 30, 65]), {'repeat_window_minutes': 1})['notifications']


def test_repeat_resets_at_day_boundary_even_contiguous_source():
    data = sample(raw([.7, .7, .7, .7, .7], [True, False, True, False, True], start='2025-04-01 23:59:50'))
    assert data.coverage_id.nunique() == 2
    assert not evaluate_notifications(data)['notifications']


def test_score_gate_first_qualified_not_original_event_start():
    frame = raw([.61, .7, .7, .7, .61, .7, .7], [True, True, False, True, True, False, True])
    result = evaluate_notifications(sample(frame))
    assert result['notifications'][0]['first_trigger_at'] == '2025-04-01T00:00:35'


def test_no_future_max_leaks_into_first_notification():
    data = sample(raw([.7] * 12 + [.99] * 3))
    result = evaluate_notifications(data)
    assert result['all_logs'][0]['score_max'] == .99
    assert result['notifications'][0]['first_score_at_trigger'] == .7
    assert result['trigger_candidates'][0]['score_at_trigger'] == .7


def test_quiet_gap_is_sliding_and_exact_boundary_starts_new_group():
    # Candidates at 65s, 365s and 665s remain one rolling group despite >10min span.
    result = evaluate_notifications(bursts([0, 60, 360, 660]), {'repeat_count': 2})
    assert result['summary']['notification_count'] == 1
    assert result['notifications'][0]['candidate_count'] == 3
    exact = evaluate_notifications(bursts([0, 60, 660]), {'repeat_count': 2})
    assert exact['summary']['notification_count'] == 2


def test_merge_zero_preserves_each_candidate():
    result = evaluate_notifications(bursts([0, 20, 40, 60]), {'merge_window_minutes': 0})
    assert result['summary']['candidate_count'] == result['summary']['notification_count'] == 2


def test_original_log_partition_is_disjoint_complete():
    result = evaluate_notifications(bursts([0, 20, 40, 4000]))
    original = {r['event_id'] for r in result['all_logs']}
    kept = {r['event_id'] for r in result['candidate_events']}
    only = {r['event_id'] for r in result['log_only']}
    assert not kept & only
    assert kept | only == original
    assert len(original) == result['summary']['notified_original_event_count'] + result['summary']['log_only_event_count']


def test_duplicate_raw_or_portable_observations_rejected():
    frame = raw([.7, .8], seconds=[0, 0])
    with pytest.raises(ValueError):
        sample(frame)
    data = sample(raw([.7]))
    with pytest.raises(ValueError):
        evaluate_notifications(pd.concat([data, data]))


@pytest.mark.parametrize('policy', [ {'score_margin': -1}, {'score_margin': float('nan')},
    {'duration_seconds': 61}, {'duration_seconds': 0}, {'repeat_count': 1},
    {'repeat_window_minutes': 0}, {'merge_window_minutes': -1}, {'repeat_count': True}, {'unknown': 3}])
def test_invalid_policy_rejected(policy):
    with pytest.raises(ValueError):
        normalize_policy(policy)


def cache(tmp_path):
    data = bursts([0, 20, 40])
    target = tmp_path / 'ai_notification_samples.parquet'
    data.to_parquet(target, index=False)
    hourly = tmp_path / 'dashboard_hourly.parquet'
    events = tmp_path / 'alert_events.parquet'
    hourly.write_bytes(b'fixture-hourly')
    events.write_bytes(b'fixture-events')
    parent_meta = {'complete': True, 'ai_thresholds': {'13(3호기)': .6}, 'source_manifest_sha256': 'source', 'source_model_sha256': 'model'}
    (tmp_path / 'dashboard_hourly.json').write_text(json.dumps(parent_meta), encoding='utf-8')
    (tmp_path / 'alert_events.json').write_text(json.dumps(parent_meta), encoding='utf-8')
    meta = {'ai_thresholds': parent_meta['ai_thresholds'], 'source_manifest_sha256': 'source', 'source_model_sha256': 'model',
        'json_metadata_hash_method': JSON_HASH_METHOD,
        'hourly_metadata_sha256': json_content_sha256(tmp_path / 'dashboard_hourly.json'),
        'event_metadata_sha256': json_content_sha256(tmp_path / 'alert_events.json'), 'format_version': 1, 'complete': True, 'sample_seconds': 5, 'sample_rows': len(data),
        'samples_sha256': sha256(target), 'hourly_sha256': sha256(hourly), 'events_sha256': sha256(events),
        'dates': {'2025-04-01': {'per_equipment': {'13(3호기)': 9}, 'first_observation_at': '2025-04-01T00:00:00'}}}
    (tmp_path / 'ai_notification_samples.json').write_text(json.dumps(meta), encoding='utf-8')
    return tmp_path


def test_missing_or_corrupt_cache_fail_closed(tmp_path):
    assert not get_ai_notifications(cache_dir=tmp_path)['data_available']
    root = cache(tmp_path)
    (root / 'ai_notification_samples.parquet').write_bytes(b'broken')
    result = get_ai_notifications(cache_dir=root)
    assert result['status'] == 'unavailable'
    assert result['summary'] == {}


@pytest.mark.parametrize('filename', ['dashboard_hourly.parquet', 'alert_events.parquet', 'dashboard_hourly.json', 'alert_events.json'])
def test_changed_upstream_fails_closed(tmp_path, filename):
    root = cache(tmp_path)
    (root / filename).write_bytes(b'changed')
    assert not get_ai_notifications(cache_dir=root)['data_available']


def test_may_and_unfinished_first_observation_are_unavailable(tmp_path):
    root = cache(tmp_path)
    assert not get_ai_notifications('2025-05-01', cache_dir=root)['data_available']
    assert not get_ai_notifications('2025-04-01', cutoff='2025-04-01T00:00:02', cache_dir=root)['data_available']
    assert get_ai_notifications('2025-04-01', cutoff='2025-04-01T00:00:05', cache_dir=root)['data_available']


def test_equipment_coverage_and_independent_setting_changes(tmp_path):
    root = cache(tmp_path)
    result = get_ai_notifications('2025-04-01', cache_dir=root)
    assert len(result['per_equipment']) == 13
    assert sum(row['data_available'] for row in result['per_equipment']) == 1
    assert not get_ai_notifications('2025-04-01', equipment='4호기', cache_dir=root)['data_available']
    changed = get_ai_notifications('2025-04-01', policy={'score_margin': .2}, cache_dir=root)
    assert result['summary']['ai_record_count'] == changed['summary']['ai_record_count'] == 3
    assert changed['summary']['notification_count'] == 0


@pytest.mark.parametrize('filename', ['dashboard_hourly.json', 'alert_events.json'])
def test_parent_json_formatting_bom_and_crlf_do_not_invalidate_cache(tmp_path, filename):
    root = cache(tmp_path)
    before = get_ai_notifications('2025-04-01', cache_dir=root)
    path = root / filename
    content = json.loads(path.read_text(encoding='utf-8'))
    # Reverse key order, pretty print, retain Unicode, prepend BOM and use CRLF.
    content = dict(reversed(list(content.items())))
    formatted = json.dumps(content, ensure_ascii=False, indent=4).replace('\n', '\r\n')
    path.write_bytes(b'\xef\xbb\xbf' + formatted.encode('utf-8'))
    after = get_ai_notifications('2025-04-01', cache_dir=root)
    assert after['data_available']
    assert after['summary'] == before['summary']


@pytest.mark.parametrize('filename', ['dashboard_hourly.json', 'alert_events.json'])
def test_parent_json_semantic_change_still_fails_closed(tmp_path, filename):
    root = cache(tmp_path)
    path = root / filename
    content = json.loads(path.read_text(encoding='utf-8'))
    content['ai_thresholds']['13(3호기)'] = .61
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    result = get_ai_notifications('2025-04-01', cache_dir=root)
    assert not result['data_available']
    assert result['summary'] == {}
