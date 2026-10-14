import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_runtime.agent_data import (
    DataService, MODULE, RULE_ITEMS, _sha256, _source_meta,
    _verify_source, prepare_data_cache,
)


@pytest.fixture
def tiny_bundle(tmp_path):
    source = tmp_path / 'data/rtu_data_full.csv'
    source.parent.mkdir()
    source.write_bytes(b'synthetic source for integrity checks\n')
    model = tmp_path / 'models/anomaly_v2/selected_model.joblib'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'synthetic frozen model identity')
    cache = tmp_path / 'outputs/agent_data'
    cache.mkdir(parents=True)
    modules = [f'{i}(test-{i})' for i in range(1, 14)]
    rows = []
    for i, module in enumerate(modules):
        for second in (0, 5):
            rows.append({MODULE: module, 'datetime': pd.Timestamp('2025-04-29') + pd.Timedelta(seconds=second),
                         **{column: 200.0 + i + second for column in RULE_ITEMS}})
    frame = pd.DataFrame(rows)
    partition = cache / '2025-04-29.parquet'
    frame.to_parquet(partition, index=False)
    manifest = {
        'complete': True, 'source_file': 'data/rtu_data_full.csv', 'source_sha256': _sha256(source),
        'model_file': 'models/anomaly_v2/selected_model.joblib', 'equipment': modules,
        'signature': {'format_version': 1, 'source_metadata': _source_meta(source),
                      'model_sha256': _sha256(model), 'period_start': '2025-04-01',
                      'period_end': '2025-04-29', 'sample_seconds': 5},
        'dates': {'2025-04-29': {'file': partition.name, 'rows': len(frame),
                              'size': partition.stat().st_size, 'sha256': _sha256(partition), 'complete': True}},
    }
    (cache / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return tmp_path, source, model, manifest


@pytest.mark.parametrize('item,sensors', [
    ('phase_voltage', ['phase_voltage_R', 'phase_voltage_S', 'phase_voltage_T']),
    ('line_voltage', ['line_voltage_RS', 'line_voltage_ST', 'line_voltage_TR']),
    ('power_factor', ['power_factor_R', 'power_factor_S', 'power_factor_T']),
    ('phase_voltage_R', ['phase_voltage_R']),
])
def test_group_preserves_all_equipment_and_reads_day_once(tiny_bundle, monkeypatch, item, sensors):
    data = DataService(tiny_bundle[0])
    calls = []
    original = data.daily_frame
    def read(date):
        calls.append(date)
        return original(date)
    monkeypatch.setattr(data, 'daily_frame', read)
    result = data.get_measurements_group('2025-04-29', '전체', item)
    assert calls == ['2025-04-29']
    assert result['data_available'] and result['equipment_count'] == 13
    for i, row in enumerate(result['items']):
        assert row['equipment'] == data.modules[i]
        assert [r['item'] for r in row['measurements']] == sensors
        for reading in row['measurements']:
            assert reading['minimum_observed'] == 200 + i
            assert reading['maximum_observed'] == 205 + i
            assert reading['minimum_observed_timestamp'] == '2025-04-29T00:00:00'
            assert reading['record_count'] == 2


def test_group_missing_date_is_not_replaced_with_latest(tiny_bundle):
    data = DataService(tiny_bundle[0])
    result = data.get_measurements_group('2025-05-31', '전체')
    assert not result['data_available'] and result['items'] == []
    assert result['date'] == '2025-05-31'
    assert result['latest_available_date'] == '2025-04-29'


def test_group_does_not_turn_missing_or_nonfinite_sensors_into_zero(tiny_bundle, monkeypatch):
    data = DataService(tiny_bundle[0])
    frame = data.daily_frame('2025-04-29')
    frame = frame.loc[frame[MODULE] != data.modules[-1]].copy()
    frame.loc[frame[MODULE] == data.modules[0], 'voltageR'] = [np.nan, np.inf]
    monkeypatch.setattr(data, 'daily_frame', lambda date: frame)
    result = data.get_measurements_group('2025-04-29', '전체')
    assert len(result['items']) == 13
    for reading in [result['items'][0]['measurements'][0], *result['items'][-1]['measurements']]:
        assert not reading['data_available']
        assert reading['minimum_observed'] is None and reading['below_threshold'] is None
        assert reading['coverage'] == 0
    json.dumps(result, allow_nan=False)


def test_source_timestamp_change_requires_identical_bytes(tiny_bundle):
    root, source, _, manifest = tiny_bundle
    original = source.stat()
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns + 10_000_000))
    assert _verify_source(source, manifest) == _source_meta(source)
    assert DataService(root).available_dates == ['2025-04-29']
    source.write_bytes(b'X' * source.stat().st_size)
    with pytest.raises(ValueError, match='SHA256'):
        _verify_source(source, manifest)
    assert not DataService(root).available_dates


def test_existing_cache_can_relink_identical_source_without_model_loading(tiny_bundle):
    root, source, _, manifest = tiny_bundle
    moved = root / 'shared-source.csv'
    moved.write_bytes(source.read_bytes())
    result = prepare_data_cache(root, source_path=moved)
    assert result['source_file'] == 'shared-source.csv'
    assert result['source_sha256'] == manifest['source_sha256']
    source.unlink()
    assert DataService(root).available_dates == ['2025-04-29']
    moved.write_bytes(b'Y' * moved.stat().st_size)
    with pytest.raises(ValueError, match='SHA256'):
        prepare_data_cache(root, source_path=moved)


def test_model_or_partition_identity_change_is_rejected(tiny_bundle):
    root, _, model, manifest = tiny_bundle
    original = model.read_bytes()
    model.write_bytes(b'changed model')
    assert not DataService(root).available_dates
    with pytest.raises(ValueError, match='identity'):
        prepare_data_cache(root)
    model.write_bytes(original)
    partition = root / 'outputs/agent_data' / manifest['dates']['2025-04-29']['file']
    partition.write_bytes(b'changed partition')
    with pytest.raises(ValueError, match='partition integrity'):
        prepare_data_cache(root)
