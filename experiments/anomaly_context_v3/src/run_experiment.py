"""Fit label-free detectors, freeze thresholds on March, evaluate April as development."""
from __future__ import annotations
import json, hashlib, time, sys
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from protocol import freeze, PROTOCOL
from context_features import build_features, STATIC_FEATURES, ALL_FEATURES
from evaluation import group_alert_events, summarize_alerts, generate_synthetic_probes, evaluate_synthetic_probe

ROOT=Path(__file__).resolve().parents[1]
MODULE='module'
SENSORS=['activePower','voltageR','voltageS','voltageT','currentR','currentS','currentT','powerFactorR','powerFactorS','powerFactorT']
METHODS=['fixed_rule','static_if','robust_context','context_if']
SEED=PROTOCOL['seed']

def write_json(path, value):
 def default(x):
  if isinstance(x,np.generic): return x.item()
  if isinstance(x,(pd.Timestamp,datetime)): return x.isoformat()
  if isinstance(x,Path): return str(x)
  raise TypeError(str(type(x)))
 Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=default,allow_nan=False),encoding='utf-8')

def robust_parameters(x):
 x=np.asarray(x,dtype=float)
 center=np.median(x,axis=0)
 q25,q75=np.quantile(x,[.25,.75],axis=0)
 scale=np.maximum((q75-q25)/1.349,np.maximum(np.abs(center)*.001,1e-6))
 return center,scale

def raw_rules(frame):
 flags=np.zeros(len(frame),bool)
 for col in ['voltageR','voltageS','voltageT']:
  flags |= frame[col+'_min'].to_numpy()<210
 for col in ['voltageRS','voltageST','voltageTR']:
  flags |= frame[col+'_min'].to_numpy()<360
 for col in ['powerFactorR','powerFactorS','powerFactorT']:
  flags |= frame[col+'_min'].to_numpy()<85
 return flags

def event_count_fast(flags,timestamps,min_consecutive=3):
 flags=np.asarray(flags,bool)
 if len(flags)==0:return 0
 times=np.asarray(timestamps,dtype='datetime64[ns]').astype('int64')
 continuous=np.diff(times)==60_000_000_000
 starts=np.flatnonzero(flags & np.r_[True,~flags[:-1]|~continuous])
 ends=np.flatnonzero(flags & np.r_[~flags[1:]|~continuous,True])
 return int(np.count_nonzero(ends-starts+1>=min_consecutive))

def calibrate(scores,timestamps,budget=1.0):
 score=np.asarray(scores,float)
 if not len(score) or not np.isfinite(score).all():raise ValueError('Nonfinite or empty calibration')
 quantiles=np.unique(np.r_[np.linspace(.5,.99,100),.995,.999,.9995,.9999,.99999])
 thresholds=np.unique(np.quantile(score,quantiles))
 days=len(score)/1440
 rows=[{'threshold':float(th),'events':event_count_fast(score>=th,timestamps),'observed_days':days} for th in thresholds]
 for r in rows:r['events_per_day']=r['events']/days
 best=min(rows,key=lambda r:(abs(r['events_per_day']-budget),-r['threshold']))
 return {**best,'target_events_per_day':budget},rows

def predict(frame,features,bundle):
 valid=features['valid'].to_numpy(bool)
 arr=features[ALL_FEATURES].to_numpy(float)
 z=(arr-bundle['center'])/bundle['scale']
 scores={}
 for method in ['static_if','context_if','robust_context']:
  score=np.full(len(frame),np.nan)
  if valid.any():
   if method=='robust_context':score[valid]=np.max(np.abs(z[valid]),axis=1)
   else:
    idx=bundle['static_idx'] if method=='static_if' else np.arange(len(ALL_FEATURES))
    score[valid]=-bundle[method].score_samples(z[valid][:,idx].astype(np.float32))
  scores[method]=score
 scores['fixed_rule']=raw_rules(frame).astype(float)
 return scores,valid

def fit_one(frame,features):
 ts=frame.timestamp
 valid=features.valid.to_numpy(bool)
 train=valid & ts.ge('2024-12-01').to_numpy() & ts.lt('2025-03-01').to_numpy()
 calib=valid & ts.ge('2025-03-01').to_numpy() & ts.lt('2025-03-16').to_numpy()
 if train.sum()<5000 or calib.sum()<1000:raise ValueError('Insufficient temporal training/calibration coverage')
 arr=features[ALL_FEATURES].to_numpy(float)
 center,scale=robust_parameters(arr[train])
 rng=np.random.default_rng(SEED)
 positions=np.flatnonzero(train)
 if len(positions)>40000:positions=np.sort(rng.choice(positions,40000,replace=False))
 z=((arr[positions]-center)/scale).astype(np.float32)
 idx=[ALL_FEATURES.index(c) for c in STATIC_FEATURES]
 bundle={'center':center,'scale':scale,'feature_names':ALL_FEATURES,'static_idx':idx,
         'trained_minutes':int(train.sum()),'fitted_minutes':len(positions),'train_end_exclusive':'2025-03-01',
         'calibration_end_exclusive':'2025-03-16','protocol_sha256':freeze()['sha256']}
 for method,cols in [('static_if',idx),('context_if',list(range(len(ALL_FEATURES))))]:
  bundle[method]=IsolationForest(n_estimators=100,max_samples=2048,n_jobs=2,random_state=SEED,contamination='auto').fit(z[:,cols])
 calib_scores,_=predict(frame.loc[calib],features.loc[calib],bundle)
 policies={}; grid={}
 for method in ['static_if','context_if','robust_context']:
  policies[method]={}
  for budget in [.5,1.,2.]:
   choice,rows=calibrate(calib_scores[method],ts.to_numpy()[calib],budget)
   policies[method][str(budget)]=choice
   grid[method]=rows
 policies['fixed_rule']={str(b):{'threshold':.5,'target_events_per_day':None} for b in [.5,1.,2.]}
 bundle['policies']=policies
 bundle['frozen_utc']=datetime.now(timezone.utc).isoformat()
 scores,valid=predict(frame,features,bundle)
 sensor_center,sensor_scale=robust_parameters(frame.loc[train,[c+'_mean' for c in SENSORS]])
 bundle['sensor_scales']=dict(zip(SENSORS,sensor_scale))
 bundle['sensor_centers']=dict(zip(SENSORS,sensor_center))
 return bundle,scores,valid,grid

def probe_evaluation(frame,bundle):
 generation_audit={}
 probes=generate_synthetic_probes(frame.loc[frame.timestamp.ge('2025-04-01')].copy(),
        {str(frame[MODULE].iloc[0]):bundle['sensor_scales']},max_per_type=1,seed=SEED,generation_audit=generation_audit)
 rows=[]; metadata=[]
 for probe in probes:
  probe['probe_id']=f"{frame[MODULE].iloc[0]}:{probe['probe_id']}"
  before=probe['baseline']; after=probe['injected']
  bs,bv=predict(before,build_features(before),bundle)
  ps,pv=predict(after,build_features(after),bundle)
  for method in METHODS:
   for budget in [.5,1.,2.]:
    threshold=bundle['policies'][method][str(budget)]['threshold']
    r=evaluate_synthetic_probe(probe,(bs[method]>=threshold)&bv,(ps[method]>=threshold)&pv)
    rows.append({'method':method,'budget':budget,'module':str(frame[MODULE].iloc[0]),**r})
  metadata.append({k:v for k,v in probe.items() if k not in ('baseline','injected')})
 return rows,metadata,probes,generation_audit

def main():
 started=time.time();record=freeze()
 output=ROOT/'outputs'; model_dir=ROOT/'models'
 output.mkdir(exist_ok=True);model_dir.mkdir(exist_ok=True)
 data_manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
 if not data_manifest.get('complete'):raise ValueError('Minute aggregation is incomplete')
 frame=pd.read_parquet(ROOT/'data/minute.parquet').rename(columns={'module(equipment)':'module'})
 frame.timestamp=pd.to_datetime(frame.timestamp)
 if frame.timestamp.max()>=pd.Timestamp('2025-05-01'):raise ValueError('May data forbidden')
 modules=sorted(frame[MODULE].unique())
 metrics=[]; event_tables=[]; probe_rows=[]; probe_meta=[]; audits=[]; threshold_rows=[]; score_paths=[]; reviews=[]
 for module in modules:
  tick=time.time(); part=frame.loc[frame[MODULE].eq(module)].sort_values('timestamp').reset_index(drop=True)
  features=build_features(part)
  bundle,scores,valid,grid=fit_one(part,features)
  safe=''.join(c if c.isalnum() else '_' for c in module)
  bundle['module']=module
  joblib.dump(bundle,model_dir/f'{safe}.joblib',compress=3)
  write_json(output/f'calibration_{safe}.json',{'module':module,'policies':bundle['policies'],'grids':grid})
  scored=part[[MODULE,'timestamp','n']].copy();scored['valid']=valid
  for method in METHODS:scored[method+'_score']=scores[method]
  scored.to_parquet(output/f'scores_{safe}.parquet',index=False)
  score_paths.append(str(output/f'scores_{safe}.parquet'))
  for method in METHODS:
   for budget in [.5,1.,2.]:
    policy=bundle['policies'][method][str(budget)]
    threshold=policy['threshold']
    flags=(scores[method]>=threshold)&valid
    threshold_rows.append({'module':module,'method':method,'budget':budget,**policy})
    for label,start,end in [('march_check','2025-03-16','2025-04-01'),('april_development','2025-04-01','2025-04-30')]:
     mask=part.timestamp.ge(start)&part.timestamp.lt(end)&valid
     subset=part.loc[mask].copy()
     summary=summarize_alerts(subset,flags[mask],min_consecutive=3)
     metrics.append({'module':module,'method':method,'budget':budget,'period':label,'valid_minutes':int(mask.sum()),**summary})
     if budget==1.0:
      events=group_alert_events(subset,flags[mask],min_consecutive=3)
      if len(events):
       events=events.copy();events['method']=method;events['period']=label;events['module']=module
       event_tables.append(events)
       if method=='context_if' and label=='april_development':
        for _,event in events.head(4).iterrows():
         start_stamp=pd.Timestamp(event['start']);end_stamp=pd.Timestamp(event['end'])
         event_mask=part.timestamp.ge(start_stamp)&part.timestamp.lt(end_stamp)&valid
         pos=np.flatnonzero(event_mask)
         if not len(pos):continue
         chosen=pos[np.argmax(scores[method][pos])]
         z=np.abs((features.loc[chosen,ALL_FEATURES].to_numpy(float)-bundle['center'])/bundle['scale'])
         top=np.argsort(z)[-4:][::-1]
         reviews.append({'module':module,'start':start_stamp.isoformat(),'end':end_stamp.isoformat(),
            'decision_time':pd.Timestamp(event['decision_time']).isoformat(),'duration_minutes':event['duration_minutes'],
            'peak_score':float(scores[method][chosen]),'threshold':threshold,
            'rule_flagged_minutes':int(raw_rules(part.loc[event_mask]).sum()),
            'related_large_deviations':'; '.join(f'{ALL_FEATURES[i]}={features.loc[chosen,ALL_FEATURES[i]]:.4g} (train median {bundle["center"][i]:.4g})' for i in top),
            'interpretation':'Unreviewed pattern deviation; not a confirmed fault or causal explanation.'})
  train=part.timestamp.lt('2025-03-01')&valid
  cols=['activePower_mean','currentR_mean','currentS_mean','currentT_mean','powerFactorR_mean','voltageR_mean']
  corr_frame=part.loc[train,cols].corr().round(4)
  corr=corr_frame.astype(object).where(corr_frame.notna(),None).to_dict()
  audits.append({'module':module,'minutes':len(part),'valid_minutes':int(valid.sum()),'fitted_minutes':bundle['fitted_minutes'],
    'train_correlations':corr,'operation_unique':sorted(part.operation_mean.dropna().unique().tolist()),
    'power_mean_cv':float(part.loc[train,'activePower_mean'].std()/part.loc[train,'activePower_mean'].mean()),
    'sensor_centers':bundle['sensor_centers'],'sensor_scales':bundle['sensor_scales']})
  p_rows,p_meta,probes,generation_audit=probe_evaluation(part,bundle)
  write_json(output/f'probe_generation_{safe}.json',generation_audit)
  probe_rows.extend(p_rows);probe_meta.extend(p_meta)
  # Save one transparent paired example per type for the local review report.
  for probe in probes:
   example=probe['baseline'].copy();after=probe['injected']
   for col in ['voltageR_mean','voltageR_std','currentR_mean','activePower_mean','powerFactorR_mean']:
    example[col+'_injected']=after[col].to_numpy()
   example.to_parquet(output/f'probe_{safe}_{probe["kind"]}.parquet',index=False)
  print(f'{module}: {valid.sum():,} valid minutes, {len(probes)} probes, {time.time()-tick:.1f}s',flush=True)
  # Persist incremental results so a failed later equipment can be diagnosed.
  pd.DataFrame(metrics).to_csv(output/'real_alert_metrics.csv',index=False,encoding='utf-8-sig')
  pd.DataFrame(probe_rows).to_csv(output/'synthetic_results.csv',index=False,encoding='utf-8-sig')
  write_json(output/'training_audit.json',audits)
  write_json(output/'synthetic_manifest.json',probe_meta)
 if event_tables:pd.concat(event_tables,ignore_index=True).to_csv(output/'events.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(reviews).to_csv(output/'review_queue.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(threshold_rows).to_csv(output/'thresholds.csv',index=False,encoding='utf-8-sig')
 write_json(output/'run_manifest.json',{'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':record['sha256'],
     'rows':len(frame),'modules':modules,'feature_count':len(ALL_FEATURES),'static_feature_count':len(STATIC_FEATURES),
     'score_files':score_paths,'elapsed_seconds':time.time()-started,'api_calls':0,'dashboard_changed':False,
     'evaluation_notice':'April development re-evaluation and minute-aggregate synthetic function tests only. No real fault accuracy claim.'})
 print(f'COMPLETE {time.time()-started:.1f}s',flush=True)

if __name__=='__main__':main()
