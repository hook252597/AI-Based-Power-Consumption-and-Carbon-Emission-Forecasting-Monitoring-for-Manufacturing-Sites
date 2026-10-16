"""Build a portable, verified copy of EVERY existing positive AI observation.

No inference, retraining, model flag change or notification policy is baked into
this cache. Normal source observations determine exact continuity identifiers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from monitoring.ai_notifications import (JSON_HASH_METHOD, build_detection_samples,
    evaluate_notifications, json_content_sha256)
from monitoring.alert_service import SAMPLE_SECONDS, sha256
from tools.prepare_alert_events import _check_day, _inside


def build(source_cache: Path, output_dir: Path | None = None) -> dict:
    source_cache = Path(source_cache).resolve()
    output_dir = Path(output_dir or REPO / 'ml/outputs/agent_data').resolve()
    manifest_path = source_cache / 'manifest.json'
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    signature = manifest.get('signature') or {}
    if not manifest.get('complete') or signature.get('sample_seconds') != SAMPLE_SECONDS:
        raise ValueError('완료·5초 검증이 없는 캐시는 사용할 수 없습니다.')
    project = source_cache.parents[1]
    source = (project / manifest['source_file']).resolve()
    model = (project / manifest['model_file']).resolve()
    stat = source.stat()
    source_meta = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if source_meta['size'] != signature['source_metadata']['size']:
        raise ValueError('원본 파일 크기가 검증 캐시와 다릅니다.')
    if source_meta != signature['source_metadata'] and sha256(source) != manifest['source_sha256']:
        raise ValueError('원본 파일 SHA256이 검증 캐시와 다릅니다.')
    if sha256(model) != signature['model_sha256']:
        raise ValueError('캐시 생성 당시 모델이 변경되었습니다.')
    hourly_path = output_dir / 'dashboard_hourly.parquet'
    event_path = output_dir / 'alert_events.parquet'
    hourly_hash, event_hash = sha256(hourly_path), sha256(event_path)
    hourly_metadata_hash, event_metadata_hash = json_content_sha256(hourly_path.with_suffix('.json')), json_content_sha256(event_path.with_suffix('.json'))
    hourly = pd.read_parquet(hourly_path)
    hourly['datetime'] = pd.to_datetime(hourly['datetime'])
    hourly_meta = json.loads(hourly_path.with_suffix('.json').read_text(encoding='utf-8-sig'))
    event_meta = json.loads(event_path.with_suffix('.json').read_text(encoding='utf-8-sig'))
    if (not hourly_meta.get('complete') or not event_meta.get('complete') or event_meta.get('events_sha256') != event_hash
            or event_meta.get('hourly_sha256') != hourly_hash
            or event_meta.get('source_manifest_sha256') != manifest_hash
            or manifest.get('ai_thresholds') != hourly_meta.get('ai_thresholds')
            or event_meta.get('ai_thresholds') != manifest.get('ai_thresholds')):
        raise ValueError('기존 5초 경보·시간별 요약·모델 기준의 검증 연결이 다릅니다.')
    events = pd.read_parquet(event_path)
    ai_events = events[events['kind'].eq('ai')]
    parts, dates = [], {}
    for day, info in sorted(manifest['dates'].items()):
        path = _inside(source_cache, info['file'])
        if not info.get('complete') or path.stat().st_size != info['size'] or sha256(path) != info['sha256']:
            raise ValueError(f'{day}: 원본 5초 캐시 검증 실패')
        frame = pd.read_parquet(path)
        if len(frame) != info['rows']:
            raise ValueError(f'{day}: 원본 표본 수 불일치')
        check = _check_day(frame, hourly, day)
        sample = build_detection_samples(frame, manifest['ai_thresholds'])
        if len(sample) != check['ai_flags']:
            raise ValueError(f'{day}: 원래 AI 탐지 기록이 빠지거나 추가되었습니다.')
        replay = evaluate_notifications(sample)
        original = ai_events[ai_events['date'].eq(day)]
        if len(replay['all_logs']) != len(original):
            raise ValueError(f'{day}: 원래 AI 발생 구간 수가 달라졌습니다.')
        # Exact tuple equality proves every original AI interval is unchanged.
        expected = sorted((r.equipment, r.timestamp, r.end, int(r.flagged_records), float(r.value))
                          for r in original.itertuples(index=False))
        actual = sorted((r['equipment'], r['start'], r['end'], r['ai_records'], r['score_max'])
                        for r in replay['all_logs'])
        if expected != actual:
            raise ValueError(f'{day}: 원래 AI 발생 구간/점수가 달라졌습니다.')
        parts.append(sample)
        dates[day] = {**check, 'source_partition_sha256': info['sha256'],
                      'ai_original_events': len(original), 'first_observation_at': pd.to_datetime(frame['datetime']).min().isoformat(),
                      'last_observation_at': pd.to_datetime(frame['datetime']).max().isoformat(), 'coverage_id_count_with_ai': int(sample['coverage_id'].nunique())}
        print(f'verified {day}: {len(sample):,} unchanged AI observations / {len(original):,} original intervals', flush=True)
    if sha256(manifest_path) != manifest_hash or sha256(hourly_path) != hourly_hash or sha256(event_path) != event_hash:
        raise ValueError('생성 중 원본 검증 자료가 변경되었습니다.')
    if json_content_sha256(hourly_path.with_suffix('.json')) != hourly_metadata_hash or json_content_sha256(event_path.with_suffix('.json')) != event_metadata_hash:
        raise ValueError('생성 중 상위 AI 캐시 메타데이터가 변경되었습니다.')
    if source.stat().st_size != stat.st_size or source.stat().st_mtime_ns != stat.st_mtime_ns:
        raise ValueError('생성 중 원본 파일이 변경되었습니다.')
    result = pd.concat(parts, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / 'ai_notification_samples.parquet'
    temporary = target.with_suffix('.parquet.partial')
    result.to_parquet(temporary, index=False)
    temporary.replace(target)
    metadata = {'format_version': 1, 'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'sample_seconds': SAMPLE_SECONDS, 'sample_rows': len(result), 'samples_sha256': sha256(target),
        'source_manifest_sha256': manifest_hash, 'source_sha256': manifest['source_sha256'],
        'source_model_sha256': signature['model_sha256'], 'hourly_sha256': hourly_hash,
        'events_sha256': event_hash, 'hourly_metadata_sha256': hourly_metadata_hash,
        'event_metadata_sha256': event_metadata_hash, 'json_metadata_hash_method': JSON_HASH_METHOD, 'ai_thresholds': manifest['ai_thresholds'], 'dates': dates,
        'note': 'All existing AI-positive observations retained without inference or threshold changes. Coverage IDs split at gaps and local day boundaries using all raw observations. Every original AI interval and its peak were matched exactly against verified alert_events.'}
    metadata_path = target.with_suffix('.json')
    temporary_meta = metadata_path.with_suffix('.json.partial')
    temporary_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary_meta.replace(metadata_path)
    print(f'READY: {len(dates)} days, {len(result):,} AI samples, {target.stat().st_size:,} bytes', flush=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO / 'ml/outputs/agent_data')
    args = parser.parse_args()
    build(args.source_cache, args.output_dir)
