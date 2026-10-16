"""Causal sensor review information backed by unchanged model detections.

Experimental settings organize review information; they are not validated safety
limits. No inference, model changes, source writes or notification sends occur.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import gzip,hashlib,json,math,os
import numpy as np
import pandas as pd
from monitoring.alert_service import EQUIPMENT_LIST,MODULE,SAMPLE_SECONDS,normalize_equipment,sha256

REPO=Path(__file__).resolve().parents[1]
DEFAULT_RAW_CACHE=REPO.parent/'outputs/agent_data'
PORTABLE_CACHE=REPO/'ml/outputs/agent_data/sensor_insights'
PORTABLE_MODEL=REPO/'ml/models/anomaly_v2/selected_model.joblib'
PORTABLE_FORMAT_VERSION=1
MAX_SNAPSHOT_BYTES=16*1024*1024
DEFAULT_POLICY={'voltage_severe_low':198.0,'voltage_severe_high':242.0,
 'voltage_low':207.0,'voltage_high':233.0,'voltage_duration_seconds':60,
 'pf_sharp_low':70.0,'pf_drop_pp':20.0,'baseline_seconds':300,
 'pf_low':80.0,'pf_duration_seconds':180,'merge_minutes':10}
VOLTAGE=['voltageR','voltageS','voltageT']
PF=['powerFactorR','powerFactorS','powerFactorT']
SENSORS=VOLTAGE+PF
KINDS=['voltage_severe','voltage_sustained','pf_sharp','pf_sustained']
NOTE=('실험용 점검 정보이며 실제 고장·위험 또는 설비 허용 범위를 확정하지 않습니다. '
 '전압 기본값은 220V 가정이며 실제 정격 확인이 필요합니다. 원래 AI 탐지 기록은 그대로 보존합니다. '
 '표본 종료 후 판단하며 지속 시간은 같은 설비·같은 상의 센서 조건으로 계산합니다. '
 'AI 음성 표본도 센서 조건이 이어지면 시간에 포함하고, 판단 시점까지 AI 근거가 있어야 정보를 만듭니다. '
 '묶음 시간 범위는 연속 이상 지속 시간이 아닙니다. 알림은 발송하지 않습니다.')


def normalize_policy(policy=None):
 out=dict(DEFAULT_POLICY)
 for k,v in (policy or {}).items():
  if k not in out: raise ValueError(f'지원하지 않는 설정: {k}')
  if isinstance(v,bool): raise ValueError('설정값은 숫자여야 합니다.')
  try: n=float(v)
  except (ValueError,TypeError) as e: raise ValueError('설정값은 숫자여야 합니다.') from e
  if not math.isfinite(n): raise ValueError('설정값은 유한해야 합니다.')
  out[k]=n
 if not 0<out['voltage_severe_low']<out['voltage_low']<out['voltage_high']<out['voltage_severe_high']:
  raise ValueError('전압 기준은 0 < 큰 하락 < 지속 하한 < 지속 상한 < 큰 상승 순서여야 합니다.')
 if not 0<=out['pf_sharp_low']<out['pf_low']<=100 or not 0<out['pf_drop_pp']<=100:
  raise ValueError('역률은 0 <= 급락 하한 < 지속 하한 <= 100, 하락 폭은 0~100%p여야 합니다.')
 for k in ['voltage_duration_seconds','pf_duration_seconds','baseline_seconds','merge_minutes']:
  if out[k]!=int(out[k]): raise ValueError('시간 설정은 정수여야 합니다.')
  out[k]=int(out[k])
  if k!='merge_minutes' and (out[k]<5 or out[k]%5): raise ValueError('시간은 5초 이상의 5초 배수여야 합니다.')
 if out['merge_minutes']<0: raise ValueError('묶음 시간은 0 이상이어야 합니다.')
 return out


def _stamp(value):
 t=pd.Timestamp(value)
 if pd.isna(t): raise ValueError('날짜가 올바르지 않습니다.')
 return t.tz_convert('Asia/Seoul').tz_localize(None) if t.tzinfo else t


def _id(prefix,*parts):
 return prefix+'_'+hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:24]


def _runs(mask,coverage):
 pos=np.flatnonzero(mask)
 if not len(pos): return []
 breaks=np.r_[True,(np.diff(pos)!=1)|(coverage[pos[1:]]!=coverage[pos[:-1]])]
 return np.split(pos,np.flatnonzero(breaks)[1:])


def _empty_counts():
 return dict(original_event_count=0,ai_record_count=0,event_count=0,group_count=0,
  **{k+'_count':0 for k in KINDS})


def _unavailable(policy,day,note):
 return {'status':'unavailable','data_available':False,'policy':policy,'day':day,
  'summary':dict.fromkeys(_empty_counts()),'per_equipment':[
   {'equipment':name,'data_available':False,**dict.fromkeys(_empty_counts()),
    'voltage_min':None,'voltage_max':None,'pf_min':None,'pf_max':None,'sensor_ranges':{}}
   for _,name in EQUIPMENT_LIST], 'events':[],'groups':[],'all_ai_logs':[],'log_only':[],
  'note':note+' 자료 없음은 정상이나 0건을 뜻하지 않습니다.'}


def evaluate_sensor_insights(frame,policy=None,cutoff=None):
 """Replay only completed five-second observations, without mutating inputs.

 Event trigger fields are frozen at the first eligible decision. End, extrema
 and observed duration describe the currently available portion of that event.
 Source AI IDs cover observed event records; trigger_ai_event_ids are causal
 evidence available at trigger_at. Missing values break only that sensor.
 """
 p=normalize_policy(policy)
 required={MODULE,'datetime','ai_flag','ai_score',*SENSORS}
 if required-set(frame): raise ValueError('5초 센서 정보에 필요한 컬럼이 없습니다.')
 data=frame.copy(deep=True)
 data['datetime']=pd.to_datetime(data['datetime'])
 if data['datetime'].dt.tz is not None: data['datetime']=data['datetime'].dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
 if data['datetime'].isna().any() or data[MODULE].isna().any(): raise ValueError('설비 또는 시간이 누락되었습니다.')
 mapping={x:normalize_equipment(x) for x in data[MODULE].unique()}
 if any(v is None for v in mapping.values()): raise ValueError('설비가 올바르지 않습니다.')
 data['equipment']=data[MODULE].map(mapping)
 if data.duplicated(['equipment','datetime']).any(): raise ValueError('설비별 시각이 중복되었습니다.')
 if not data.ai_flag.isin([0,1,False,True]).all(): raise ValueError('AI 판정값이 올바르지 않습니다.')
 data['ai_flag']=data.ai_flag.astype(bool)
 for c in ['ai_score']+SENSORS: data[c]=pd.to_numeric(data[c],errors='coerce')
 if not np.isfinite(data.loc[data.ai_flag,'ai_score']).all(): raise ValueError('AI 양성 기록의 점수가 유효하지 않습니다.')
 if cutoff is not None: data=data[data.datetime+pd.Timedelta(seconds=5)<=_stamp(cutoff)].copy()
 days=sorted(data.datetime.dt.strftime('%Y-%m-%d').unique())
 day=days[0] if len(days)==1 else None
 if data.empty: return _unavailable(p,day,'관측이 완료된 센서 자료가 없습니다.')
 events=[];logs=[];cards={};baseline_missing=0
 for name,source in data.groupby('equipment',sort=False):
  g=source.sort_values('datetime').reset_index(drop=True)
  times=g.datetime;ns=times.to_numpy(dtype='datetime64[ns]').astype('int64')
  dayvals=times.dt.normalize().to_numpy()
  breaks=np.r_[True,(np.diff(ns)!=5_000_000_000)|(dayvals[1:]!=dayvals[:-1])]
  cov=np.cumsum(breaks);ai=g.ai_flag.to_numpy();scores=g.ai_score.to_numpy(dtype=float)
  ai_ids=np.full(len(g),None,dtype=object)
  for run in _runs(ai,cov):
   first,last=int(run[0]),int(run[-1]);eid=_id('ai',name,times.iloc[first].isoformat());ai_ids[run]=eid
   logs.append({'id':eid,'event_id':eid,'equipment':name,'start':times.iloc[first].isoformat(),
    'end':(times.iloc[last]+pd.Timedelta(seconds=5)).isoformat(),'score':float(np.max(scores[run])),
    'score_max':float(np.max(scores[run])),'ai_records':len(run),'duration_seconds':len(run)*5,
    'preserved':True})
  coverage_ids={int(cov[i]):_id('sensor_coverage',name,times.iloc[i].isoformat()) for i in np.flatnonzero(breaks)}
  ranges={}
  for s in SENSORS:
   finite=g[s][np.isfinite(g[s])]
   ranges[s]={'min':float(finite.min()) if len(finite) else None,'max':float(finite.max()) if len(finite) else None,
    'valid_count':len(finite),'missing_count':len(g)-len(finite)}
  def family_range(cols,key):
   values=[ranges[s][key] for s in cols if ranges[s][key] is not None]
   return (min(values) if key=='min' else max(values)) if values else None
  cards[name]={'equipment':name,'data_available':True,'observed_records':len(g),'sensor_ranges':ranges,
   'voltage_min':family_range(VOLTAGE,'min'),'voltage_max':family_range(VOLTAGE,'max'),
   'pf_min':family_range(PF,'min'),'pf_max':family_range(PF,'max')}
  def emit(run,sensor,kind,direction,required_seconds=5,baseline=None):
   evidence=run[ai[run]]
   if not len(evidence): return
   trigger=max(int(run[0])+required_seconds//5-1,int(evidence[0]))
   if trigger>int(run[-1]): return
   at_evidence=evidence[evidence<=trigger]
   values=g[sensor].to_numpy(dtype=float);start=times.iloc[int(run[0])];last=times.iloc[int(run[-1])]
   at=times.iloc[trigger]+pd.Timedelta(seconds=5)
   extrema=float(np.max(values[run]) if direction=='high' else np.min(values[run]))
   e={'id':_id('sensor',name,sensor,kind,direction,start.isoformat()),'equipment':name,'item':sensor,'kind':kind,
    'direction':direction,'start':start.isoformat(),'end':(last+pd.Timedelta(seconds=5)).isoformat(),
    'trigger_at':at.isoformat(),'value_at_trigger':float(values[trigger]),'extreme_value':extrema,
    'duration_seconds':len(run)*5,'duration_at_trigger_seconds':(trigger-int(run[0])+1)*5,
    'baseline':None if baseline is None else float(baseline),
    'drop_pp':None if baseline is None else float(baseline-values[trigger]),
    'ai_score':float(np.max(scores[at_evidence])),
    'source_ai_event_ids':sorted(set(ai_ids[evidence].tolist())),
    'trigger_ai_event_ids':sorted(set(ai_ids[at_evidence].tolist())),
    'coverage_id':coverage_ids[int(cov[trigger])],'date':times.iloc[trigger].strftime('%Y-%m-%d')}
   events.append(e)
  for sensor in VOLTAGE:
   values=g[sensor].to_numpy(dtype=float);valid=np.isfinite(values)
   for kind,low,high,duration in [('voltage_severe',p['voltage_severe_low'],p['voltage_severe_high'],5),
      ('voltage_sustained',p['voltage_low'],p['voltage_high'],p['voltage_duration_seconds'])]:
    for direction,mask in [('low',valid&(values<low)),('high',valid&(values>high))]:
     for run in _runs(mask,cov): emit(run,sensor,kind,direction,duration)
  for sensor in PF:
   values=g[sensor].to_numpy(dtype=float);valid=np.isfinite(values)
   for run in _runs(valid&(values<p['pf_low']),cov): emit(run,sensor,'pf_sustained','low',p['pf_duration_seconds'])
   baseline=np.full(len(g),np.nan);window=p['baseline_seconds']//5
   for run in _runs(valid,cov):
    baseline[run]=pd.Series(values[run]).rolling(window,min_periods=window).median().shift(1).to_numpy()
   baseline_missing+=int((~np.isfinite(baseline)).sum())
   for low_run in _runs(valid&(values<p['pf_sharp_low']),cov):
    cursor=0
    while cursor<len(low_run):
     tail=low_run[cursor:]
     eligible=np.flatnonzero(np.isfinite(baseline[tail])&(baseline[tail]-values[tail]>=p['pf_drop_pp']))
     if not len(eligible): break
     first=int(eligible[0]);fixed=float(baseline[int(tail[first])]);rest=tail[first:]
     stopped=np.flatnonzero(fixed-values[rest]<p['pf_drop_pp'])
     length=int(stopped[0]) if len(stopped) else len(rest)
     emit(rest[:length],sensor,'pf_sharp','low',5,fixed)
     cursor+=first+length
 events.sort(key=lambda e:(e['trigger_at'],e['equipment'],e['item'],e['kind']))
 groups=[];latest={}
 for e in events:
  key=(e['equipment'],e['item'],e['kind'],e['direction'],e['coverage_id'],e['date']);group=latest.get(key)
  if group is None or p['merge_minutes']==0 or _stamp(e['trigger_at'])-_stamp(group['last_trigger_at'])>=pd.Timedelta(minutes=p['merge_minutes']):
   group={'id':_id('sensor_group',*key,e['trigger_at']),'equipment':e['equipment'],'item':e['item'],
    'kind':e['kind'],'direction':e['direction'],'first_trigger_at':e['trigger_at'],'last_trigger_at':e['trigger_at'],
    'first_value_at_trigger':e['value_at_trigger'],'event_count':0,'event_ids':[],
    'extreme_value':e['extreme_value'],'max_event_duration_seconds':0,'source_ai_event_ids':[]}
   groups.append(group);latest[key]=group
  group['last_trigger_at']=e['trigger_at'];group['event_count']+=1;group['event_ids'].append(e['id'])
  group['max_event_duration_seconds']=max(group['max_event_duration_seconds'],e['duration_seconds'])
  group['extreme_value']=(max if e['direction']=='high' else min)(group['extreme_value'],e['extreme_value'])
  group['source_ai_event_ids']=sorted(set(group['source_ai_event_ids'])|set(e['source_ai_event_ids']))
 linked={eid for e in events for eid in e['source_ai_event_ids']}
 for row in logs: row['information_linked']=row['event_id'] in linked
 def summary(name=None):
  ls=[v for v in logs if name is None or v['equipment']==name];es=[v for v in events if name is None or v['equipment']==name]
  gs=[v for v in groups if name is None or v['equipment']==name]
  return dict(original_event_count=len(ls),ai_record_count=sum(v['ai_records'] for v in ls),event_count=len(es),group_count=len(gs),
   **{k+'_count':sum(v['kind']==k for v in es) for k in KINDS})
 per=[]
 for _,name in EQUIPMENT_LIST:
  if name in cards: per.append({**cards[name],**summary(name)})
  else: per.append(_unavailable(p,day,'')['per_equipment'][[n for _,n in EQUIPMENT_LIST].index(name)])
 return {'status':'ok','data_available':True,'policy':p,'day':day,'summary':summary(),'per_equipment':per,
  'events':events,'groups':groups,'all_ai_logs':sorted(logs,key=lambda v:(v['start'],v['equipment'])),
  'log_only':[v for v in logs if not v['information_linked']], 'note':NOTE,
  'coverage':{'dates':days,'sample_seconds':5,'observed_records':len(data)},
  'baseline_unavailable_sensor_records':baseline_missing}


def _file_key(path):
 s=Path(path).stat();return(str(Path(path).resolve()),s.st_size,s.st_mtime_ns)


@lru_cache(maxsize=16)
def _verified_hash(key,expected):
 if sha256(Path(key[0]))!=expected: raise ValueError('원본 캐시 또는 모델 해시가 다릅니다.')
 return True


@lru_cache(maxsize=3)
def _read_partition(key,expected,rows):
 _verified_hash(key,expected)
 frame=pd.read_parquet(key[0],columns=[MODULE,'datetime','ai_flag','ai_score',*SENSORS])
 if len(frame)!=rows: raise ValueError('원본 캐시 표본 수가 다릅니다.')
 if _file_key(Path(key[0]))!=key: raise ValueError('조회 중 원본 캐시가 변경되었습니다.')
 return frame


def _get_raw_sensor_insights(selected,p,cutoff,cache):
 try:
  manifest_path=cache/'manifest.json';manifest_key=_file_key(manifest_path)
  manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'));sig=manifest.get('signature') or {}
  if not manifest.get('complete') or sig.get('sample_seconds')!=5: raise ValueError('완료·5초 검증이 없는 원본 캐시입니다.')
  info=manifest.get('dates',{}).get(selected)
  if not info or not info.get('complete'): return _unavailable(p,selected,'선택일의 검증된 원본 실측 자료가 없습니다.')
  path=(cache/info['file']).resolve()
  if not path.is_relative_to(cache): raise ValueError('원본 캐시 경로가 허용된 폴더 밖입니다.')
  key=_file_key(path)
  if key[1]!=info['size']: raise ValueError('원본 캐시 파일 크기가 다릅니다.')
  project=cache.parents[1];model=(project/manifest['model_file']).resolve()
  model_key=_file_key(model)
  _verified_hash(model_key,sig['model_sha256'])
  frame=_read_partition(key,info['sha256'],info['rows'])
  if not pd.to_datetime(frame.datetime).dt.strftime('%Y-%m-%d').eq(selected).all(): raise ValueError('원본 캐시에 다른 날짜가 포함되어 있습니다.')
  if not frame.groupby(MODULE).size().to_dict()==info['per_equipment']: raise ValueError('설비별 원본 표본 수가 검증 정보와 다릅니다.')
  if _file_key(manifest_path)!=manifest_key or _file_key(path)!=key or _file_key(model)!=model_key:
   raise ValueError('조회 중 원본 검증 정보·캐시·모델이 변경되었습니다.')
  result=evaluate_sensor_insights(frame,p,cutoff=cutoff);result['day']=selected
  result['policy_editable']=True
  result['source']={'mode':'verified_raw','cache':str(cache),'partition_sha256':info['sha256'],'model_sha256':sig['model_sha256'],'source_sha256':manifest.get('source_sha256')}
  return result
 except (OSError,ValueError,TypeError,KeyError,IndexError) as error:
  return _unavailable(p,selected,f'실측 자료 검증 실패: {error}')


def algorithm_fingerprint():
 """Hash detection semantics independently of Windows/Git line endings."""
 core=Path(__file__).read_text(encoding='utf-8-sig').split('\ndef _file_key(',1)[0]
 return hashlib.sha256(core.replace('\r\n','\n').encode('utf-8')).hexdigest()


def _portable_unavailable(p,selected,note,reason):
 result=_unavailable(p,selected,note)
 result.update(policy_editable=False,reason_code=reason,source={'mode':'portable_snapshot'})
 return result


def _get_portable_sensor_insights(selected,p,cutoff=None):
 """Verified default-policy, complete-day replay. Never approximates custom settings."""
 if p!=normalize_policy():
  return _portable_unavailable(p,selected,'기준 변경에는 검증된 5초 원본 자료가 필요합니다. 기본값으로 복원하면 제공된 결과를 볼 수 있습니다.','raw_required_for_policy')
 if cutoff is not None and _stamp(cutoff)<_stamp(selected)+pd.Timedelta(days=1):
  return _portable_unavailable(p,selected,'일부 시간만 다시 계산하려면 검증된 5초 원본 자료가 필요합니다. 제공된 결과는 선택일 전체 기준입니다.','raw_required_for_cutoff')
 try:
  root=PORTABLE_CACHE.resolve();manifest_path=root/'manifest.json';manifest_key=_file_key(manifest_path)
  manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
  if (manifest.get('format_version')!=PORTABLE_FORMAT_VERSION or not manifest.get('complete')
      or manifest.get('policy')!=p or manifest.get('algorithm_sha256')!=algorithm_fingerprint()
      or manifest.get('sample_seconds')!=5):
   raise ValueError('제공된 결과의 버전·기준·계산 코드 검증이 일치하지 않습니다.')
  info=manifest.get('dates',{}).get(selected)
  if not info:
   return _portable_unavailable(p,selected,'선택일의 기본 기준 결과가 제공되지 않습니다.','snapshot_day_missing')
  if info.get('evaluated_through')!=(_stamp(selected)+pd.Timedelta(days=1)).isoformat():
   raise ValueError('완료일 결과의 기준 시점이 다릅니다.')
  events_path=REPO/'ml/outputs/agent_data/alert_events.parquet';events_key=_file_key(events_path)
  _verified_hash(events_key,manifest['original_ai_events_sha256'])
  model_key=_file_key(PORTABLE_MODEL)
  _verified_hash(model_key,manifest['dashboard_model_sha256'])
  export=json.loads(PORTABLE_MODEL.with_name('export_manifest.json').read_text(encoding='utf-8-sig'))
  if (export.get('source_sha256')!=manifest['model_sha256']
      or export.get('export_sha256')!=manifest['dashboard_model_sha256']
      or not export.get('score_and_all_output_columns_exactly_equal')
      or hashlib.sha256(json.dumps(export,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=manifest['model_export_metadata_sha256']):
   raise ValueError('공유 모델과 원본 모델의 동일 출력 검증 정보가 다릅니다.')
  path=(root/info['file']).resolve()
  if not path.is_relative_to(root): raise ValueError('결과 경로가 허용된 폴더 밖입니다.')
  key=_file_key(path)
  if key[1]!=info['size']: raise ValueError('결과 파일 크기가 다릅니다.')
  _verified_hash(key,info['sha256'])
  if not 0<info['uncompressed_size']<=MAX_SNAPSHOT_BYTES: raise ValueError('결과 파일의 압축 해제 크기가 올바르지 않습니다.')
  with gzip.open(path,'rb') as stream: payload=stream.read(MAX_SNAPSHOT_BYTES+1)
  if len(payload)!=info['uncompressed_size']: raise ValueError('결과 압축 해제 크기가 다릅니다.')
  result=json.loads(payload)
  if (result.get('day')!=selected or result.get('policy')!=p or result.get('status')!='ok'
      or result.get('data_available') is not True or result.get('summary')!=info['summary']):
   raise ValueError('선택일 결과의 날짜·기준·요약이 검증 정보와 다릅니다.')
  if (result.get('source',{}).get('partition_sha256')!=info['source_partition_sha256']
      or result.get('source',{}).get('model_sha256')!=manifest['model_sha256']
      or result.get('source',{}).get('source_sha256')!=manifest['source_sha256']):
   raise ValueError('결과의 원본·모델 출처 검증 정보가 다릅니다.')
  if [row.get('equipment') for row in result.get('per_equipment',[])]!=[name for _,name in EQUIPMENT_LIST]:
   raise ValueError('결과의 설비 목록이 다릅니다.')
  summary=result['summary']
  if (len(result['all_ai_logs'])!=summary['original_event_count']
      or sum(row['ai_records'] for row in result['all_ai_logs'])!=summary['ai_record_count']
      or len(result['events'])!=summary['event_count'] or len(result['groups'])!=summary['group_count']):
   raise ValueError('결과의 전체 AI 기록 또는 정보 수가 요약과 다릅니다.')
  if (_file_key(manifest_path)!=manifest_key or _file_key(path)!=key or _file_key(PORTABLE_MODEL)!=model_key or _file_key(events_path)!=events_key):
   raise ValueError('조회 중 결과 검증 자료가 변경되었습니다.')
  result.update(policy_editable=False)
  result['source'].update(mode='portable_snapshot',cache='ml/outputs/agent_data/sensor_insights',
     snapshot_sha256=info['sha256'],evaluated_through=info['evaluated_through'])
  return result
 except (OSError,ValueError,TypeError,KeyError,IndexError) as error:
  return _portable_unavailable(p,selected,f'기본 기준 결과 검증 실패: {error}','snapshot_invalid')


def get_sensor_insights(day,policy=None,cutoff=None):
 """Prefer verified raw replay; a clean clone can read shipped default-day results.

 A configured or present raw cache is authoritative: corruption/missing pieces
 fail closed and never silently switch to the portable results.
 """
 p=normalize_policy(policy);selected=_stamp(day).strftime('%Y-%m-%d')
 explicit=os.environ.get('MARGARET_SENSOR_CACHE')
 cache=Path(explicit or DEFAULT_RAW_CACHE).expanduser().resolve()
 if explicit or cache.exists():
  return _get_raw_sensor_insights(selected,p,cutoff,cache)
 return _get_portable_sensor_insights(selected,p,cutoff)
