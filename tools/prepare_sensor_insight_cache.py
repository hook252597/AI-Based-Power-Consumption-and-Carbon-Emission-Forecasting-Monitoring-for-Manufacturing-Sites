"""Export verified complete-day default-policy sensor insights for team clones.

The small files include every original AI interval and review evidence, but do
not contain the full normal 5-second stream needed to change policy or cutoff.
No inference, threshold adjustment or model training is performed.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from monitoring.sensor_insights import (DEFAULT_RAW_CACHE,PORTABLE_CACHE,PORTABLE_MODEL,
 PORTABLE_FORMAT_VERSION,MAX_SNAPSHOT_BYTES,normalize_policy,algorithm_fingerprint,_get_raw_sensor_insights)
from monitoring.alert_service import sha256


def build(source_cache=None,output_dir=None):
 source_cache=Path(source_cache or os.environ.get('MARGARET_SENSOR_CACHE') or DEFAULT_RAW_CACHE).resolve()
 output_dir=Path(output_dir or PORTABLE_CACHE).resolve()
 source_manifest=source_cache/'manifest.json';source_manifest_hash=sha256(source_manifest)
 source=json.loads(source_manifest.read_text(encoding='utf-8-sig'))
 model_hash=source['signature']['model_sha256']
 export=json.loads(PORTABLE_MODEL.with_name('export_manifest.json').read_text(encoding='utf-8-sig'))
 dashboard_model_hash=sha256(PORTABLE_MODEL)
 if (export.get('source_sha256')!=model_hash or export.get('export_sha256')!=dashboard_model_hash
    or not export.get('score_and_all_output_columns_exactly_equal')):
  raise ValueError('Git에 제공할 모델과 원본 모델의 동일 출력 검증 정보가 다릅니다.')
 export_hash=hashlib.sha256(json.dumps(export,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 events_path=REPO/'ml/outputs/agent_data/alert_events.parquet'
 event_meta=json.loads(events_path.with_suffix('.json').read_text(encoding='utf-8-sig'))
 event_hash=sha256(events_path)
 if (not event_meta.get('complete') or event_meta.get('events_sha256')!=event_hash
    or event_meta.get('source_manifest_sha256')!=source_manifest_hash
    or event_meta.get('ai_thresholds')!=source.get('ai_thresholds')):
  raise ValueError('기존 AI 원기록과 검증 원본의 출처가 다릅니다.')
 originals=pd.read_parquet(events_path);originals=originals[originals['kind'].eq('ai')]
 policy=normalize_policy();algorithm_hash=algorithm_fingerprint()
 manifest={'format_version':PORTABLE_FORMAT_VERSION,'complete':True,
  'created_utc':datetime.now(timezone.utc).isoformat(),'policy':policy,'sample_seconds':5,
  'algorithm_sha256':algorithm_hash,'model_sha256':model_hash,'dashboard_model_sha256':dashboard_model_hash,
  'model_export_metadata_sha256':export_hash,'model_file':'ml/models/anomaly_v2/selected_model.joblib',
  'source_sha256':source['source_sha256'],'source_manifest_sha256':source_manifest_hash,
  'original_ai_events_sha256':event_hash,'dates':{},
  'note':'Complete-day default-policy replay from verified five-second source partitions. Every original AI interval and peak score is preserved and matched to the existing alert_events cache. Custom policy or intraday replay requires the verified raw cache. No raw CSV or local absolute paths included.'}
 output_dir.mkdir(parents=True,exist_ok=True)
 for day,source_info in sorted(source['dates'].items()):
  cutoff=(pd.Timestamp(day)+pd.Timedelta(days=1)).isoformat()
  result=_get_raw_sensor_insights(day,policy,cutoff,source_cache)
  if result.get('status')!='ok': raise ValueError(f"{day}: {result.get('note')}")
  original=originals[originals['date'].eq(day)]
  expected=sorted((r.equipment,r.timestamp,r.end,int(r.flagged_records),float(r.value)) for r in original.itertuples(index=False))
  actual=sorted((r['equipment'],r['start'],r['end'],r['ai_records'],r['score_max']) for r in result['all_ai_logs'])
  if actual!=expected: raise ValueError(f'{day}: 기존 AI 원기록과 구간/점수가 다릅니다.')
  result['source']={key:value for key,value in result['source'].items() if key!='cache'}
  result['source'].update(mode='portable_snapshot',cache='ml/outputs/agent_data/sensor_insights',evaluated_through=cutoff)
  result['policy_editable']=False
  payload=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False,
   default=lambda x:x.item() if isinstance(x,np.generic) else str(x)).encode('utf-8')
  if len(payload)>MAX_SNAPSHOT_BYTES: raise ValueError(f'{day}: 일자별 결과 파일이 너무 큽니다.')
  compressed=gzip.compress(payload,compresslevel=9,mtime=0)
  target=output_dir/f'{day}.json.gz';temporary=target.with_suffix('.gz.partial')
  temporary.write_bytes(compressed);temporary.replace(target)
  manifest['dates'][day]={'file':target.name,'size':target.stat().st_size,'uncompressed_size':len(payload),
   'sha256':sha256(target),'source_partition_sha256':source_info['sha256'],
   'evaluated_through':cutoff,'summary':result['summary']}
  print(f"verified {day}: {len(actual)} AI intervals, {result['summary']['group_count']} review groups, {len(compressed):,} bytes",flush=True)
 if sha256(source_manifest)!=source_manifest_hash or sha256(events_path)!=event_hash or sha256(PORTABLE_MODEL)!=dashboard_model_hash or algorithm_fingerprint()!=algorithm_hash:
  raise ValueError('생성 중 원본 검증 자료 또는 계산 코드가 변경되었습니다.')
 temporary=output_dir/'manifest.json.partial'
 temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 temporary.replace(output_dir/'manifest.json')
 return manifest


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--source-cache',type=Path,default=None)
 parser.add_argument('--output-dir',type=Path,default=None)
 args=parser.parse_args()
 result=build(args.source_cache,args.output_dir)
 print(f"Complete: {len(result['dates'])} verified days / {sum(x['size'] for x in result['dates'].values()):,} compressed bytes")
