"""Default-day results work without raw data, and fail closed when unsuitable."""
from pathlib import Path
import gzip,hashlib,json,sys
import pandas as pd
import pytest
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
import monitoring.sensor_insights as service
from monitoring.alert_service import MODULE,sha256


def fixture_cache(tmp_path,monkeypatch):
 root=tmp_path/'portable';root.mkdir()
 monkeypatch.setattr(service,'REPO',tmp_path)
 events=tmp_path/'ml/outputs/agent_data/alert_events.parquet';events.parent.mkdir(parents=True);events.write_bytes(b'original-ai-events')
 model=tmp_path/'selected_model.joblib';model.write_bytes(b'export-fixture')
 export={'source_sha256':'original-model','export_sha256':sha256(model),'score_and_all_output_columns_exactly_equal':True}
 model.with_name('export_manifest.json').write_text(json.dumps(export),encoding='utf-8')
 monkeypatch.delenv('MARGARET_SENSOR_CACHE',raising=False)
 monkeypatch.setattr(service,'DEFAULT_RAW_CACHE',tmp_path/'raw-not-present')
 monkeypatch.setattr(service,'PORTABLE_CACHE',root);monkeypatch.setattr(service,'PORTABLE_MODEL',model)
 frame=pd.DataFrame({MODULE:['13(3호기)']*61,'datetime':pd.date_range('2025-04-01',periods=61,freq='5s'),
  'ai_flag':[True]+[False]*60,'ai_score':.7,
  **{sensor:197. if sensor=='voltageR' else 215. for sensor in service.VOLTAGE},
  **{sensor:95. for sensor in service.PF}})
 result=service.evaluate_sensor_insights(frame)
 result['source']={'model_sha256':'original-model','source_sha256':'source','partition_sha256':'partition'}
 payload=json.dumps(result).encode();target=root/'2025-04-01.json.gz';target.write_bytes(gzip.compress(payload))
 manifest={'format_version':1,'complete':True,'policy':service.normalize_policy(),'sample_seconds':5,
  'algorithm_sha256':service.algorithm_fingerprint(),'model_sha256':'original-model','dashboard_model_sha256':sha256(model),
  'source_sha256':'source','original_ai_events_sha256':sha256(events),'model_export_metadata_sha256':hashlib.sha256(json.dumps(export,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
  'dates':{'2025-04-01':{'file':target.name,'size':target.stat().st_size,'uncompressed_size':len(payload),
   'sha256':sha256(target),'source_partition_sha256':'partition','evaluated_through':'2025-04-02T00:00:00','summary':result['summary']}}}
 (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
 return root,model,target,result


def test_clean_clone_reads_all_records_without_raw(tmp_path,monkeypatch):
 _,_,_,expected=fixture_cache(tmp_path,monkeypatch)
 result=service.get_sensor_insights('2025-04-01',cutoff='2025-04-30')
 assert result['data_available'] and result['summary']==expected['summary']
 assert result['all_ai_logs']==expected['all_ai_logs'] and result['events']==expected['events']
 assert len(result['per_equipment'])==13 and result['policy_editable'] is False
 assert result['source']['mode']=='portable_snapshot' and str(tmp_path) not in json.dumps(result)
 result['events'].clear()
 assert service.get_sensor_insights('2025-04-01')['events'] # callers cannot alter shared cache


def test_custom_policy_never_reuses_default_snapshot(tmp_path,monkeypatch):
 fixture_cache(tmp_path,monkeypatch)
 result=service.get_sensor_insights('2025-04-01',{'voltage_severe_low':195})
 assert not result['data_available'] and result['summary']['event_count'] is None
 assert result['reason_code']=='raw_required_for_policy' and result['policy']['voltage_severe_low']==195


def test_intraday_cutoff_never_exposes_future_snapshot(tmp_path,monkeypatch):
 fixture_cache(tmp_path,monkeypatch)
 result=service.get_sensor_insights('2025-04-01',cutoff='2025-04-01T23:59:59')
 assert not result['data_available'] and result['reason_code']=='raw_required_for_cutoff'
 assert service.get_sensor_insights('2025-04-01',cutoff='2025-04-02T00:00:00')['data_available']
 assert service.get_sensor_insights('2025-04-01',cutoff='2025-04-01T15:00:00Z')['data_available']


def test_missing_day_is_unknown_not_zero(tmp_path,monkeypatch):
 fixture_cache(tmp_path,monkeypatch)
 result=service.get_sensor_insights('2025-05-31')
 assert result['summary']['group_count'] is None and result['reason_code']=='snapshot_day_missing'


@pytest.mark.parametrize('target',['file','manifest','model','export','algorithm','path','payload_counts','original_ai','evaluated_through'])
def test_portable_corruption_fails_closed(tmp_path,monkeypatch,target):
 root,model,partition,_=fixture_cache(tmp_path,monkeypatch)
 assert service.get_sensor_insights('2025-04-01')['data_available']
 manifest_path=root/'manifest.json';manifest=json.loads(manifest_path.read_text())
 if target=='original_ai':(tmp_path/'ml/outputs/agent_data/alert_events.parquet').write_bytes(b'tampered')
 if target=='evaluated_through':manifest['dates']['2025-04-01']['evaluated_through']='2025-04-01T00:01:00'
 if target=='file':partition.write_bytes(b'tampered')
 if target=='model':model.write_bytes(b'tampered')
 if target=='manifest':manifest['policy']['pf_low']=81
 if target=='algorithm':manifest['algorithm_sha256']='stale'
 if target=='path':manifest['dates']['2025-04-01']['file']='../outside.json.gz'
 if target=='export':model.with_name('export_manifest.json').write_text('{}')
 if target=='payload_counts':
  result=json.loads(gzip.decompress(partition.read_bytes()));result['all_ai_logs']=[]
  payload=json.dumps(result).encode();partition.write_bytes(gzip.compress(payload))
  manifest['dates']['2025-04-01'].update(size=partition.stat().st_size,sha256=sha256(partition),uncompressed_size=len(payload))
 manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
 result=service.get_sensor_insights('2025-04-01')
 assert not result['data_available'] and result['reason_code']=='snapshot_invalid'
 assert result['summary']['event_count'] is None


@pytest.mark.parametrize('explicit',[False,True])
def test_raw_cache_problem_cannot_silently_fallback(tmp_path,monkeypatch,explicit):
 fixture_cache(tmp_path,monkeypatch)
 raw=service.DEFAULT_RAW_CACHE
 if explicit:monkeypatch.setenv('MARGARET_SENSOR_CACHE',str(raw))
 else:
  raw.mkdir();(raw/'manifest.json').write_text('{}')
 result=service.get_sensor_insights('2025-04-01')
 assert not result['data_available'] and '실측 자료 검증 실패' in result['note']


def test_actual_bundled_day_is_portable(tmp_path,monkeypatch):
 monkeypatch.delenv('MARGARET_SENSOR_CACHE',raising=False)
 monkeypatch.setattr(service,'DEFAULT_RAW_CACHE',tmp_path/'absent')
 result=service.get_sensor_insights('2025-04-29',cutoff='2025-04-30')
 assert result['data_available'],result['note']
 assert result['summary']['original_event_count']==459 and result['summary']['ai_record_count']==461
 assert result['summary']['event_count']==439 and result['summary']['group_count']==226
 assert len(result['per_equipment'])==13 and result['policy_editable'] is False
