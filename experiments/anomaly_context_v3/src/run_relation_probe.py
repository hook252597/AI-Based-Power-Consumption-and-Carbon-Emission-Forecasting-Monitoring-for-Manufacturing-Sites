"""Exploratory sensor-relation residual detector, separate from the four-method run.

Fits contemporaneous observed power against sum of contemporaneous voltage-mean
multiplied by current-mean. This is a learned empirical relation, not future power
forecasting, a validated electrical formula, or a confirmed fault detector.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from context_features import build_features
from evaluation import (SENSORS, group_alert_events, summarize_alerts, generate_synthetic_probes,
                        evaluate_synthetic_probe, summarize_synthetic_results)
from run_experiment import calibrate, robust_parameters

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/relation_experiment'
MODELS = ROOT / 'models/relation'
METHOD = 'relation_residual'
BUDGETS = (0.5,1.0,2.0)
SEED = 20261016
PROTOCOL = {
    'version':1,'experiment':METHOD,'seed':SEED,
    'status':'Exploratory follow-up after a source sensor-relation audit; not an untouched final holdout.',
    'hypothesis_source':'outputs/sensor_relation_audit.json; no results from the four-detector comparison used to choose parameters.',
    'formula':'activePower_mean = intercept + coefficient * sum(voltage_phase_mean * current_phase_mean), phases R/S/T.',
    'regression':'Per-equipment ordinary least squares, LinearRegression(fit_intercept=True,n_jobs=1).',
    'train':['2024-12-01','2025-03-01'],'calibration':['2025-03-01','2025-03-16'],
    'march_check':['2025-03-16','2025-04-01'],'april_development':['2025-04-01','2025-04-30'],
    'score':'Absolute contemporaneous observed-power minus learned relation-predicted-power residual.',
    'score_unit':'W under source activePower unit assumption; source specification confirmation remains pending.',
    'comparison_minutes':'Exactly build_features(frame).valid, including five past complete minutes and midnight/gap resets.',
    'labels_used':False,'may_used':False,
    'calibration':'Reuse run_experiment.calibrate with March1-15 absolute residual scores only; budget 0.5/1/2 events per observed equipment day.',
    'primary_budget':1.0,'persistence_minutes':3,'decision_time':'End of third consecutive flagged complete minute.',
    'synthetic':'Reuse the same generator, training-only sensor scales, seed, rule-safe paired controls, amended directions and max_per_type=1 per equipment.',
    'results_interpretation':'Report event counts and flagged-time burden, paired synthetic recall/delay and baseline pre-existing alerts. No real-fault precision/recall claim.',
    'limitations':['Contemporaneous cross-sensor consistency is not future forecasting.',
        'Product of minute means differs from mean of products; aggregation covariance contributes to residual.',
        'A learned coefficient absorbs source conventions; it is not a verified electrical conversion factor.',
        'Coherent current and power changes can stay on the relation and be missed.',
        'Changes in within-minute variability without changed means are intentionally invisible to this relation score.',
        'Sensor inconsistencies can produce a residual without equipment damage.',
        'April and source audit are development data; new real data and human-confirmed events remain necessary.'],
    'original_experiment_modified':False,'dashboard_modified':False,'git_upload':False,'api_calls':0
}


def write_json(path, value):
    def encode(item):
        if isinstance(item,np.generic):return item.item()
        if isinstance(item,(pd.Timestamp,datetime)):return item.isoformat()
        if isinstance(item,Path):return str(item)
        raise TypeError(str(type(item)))
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=encode,allow_nan=False),encoding='utf-8')


def freeze_protocol():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    canonical=json.dumps(PROTOCOL,sort_keys=True,ensure_ascii=False).encode('utf-8')
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(canonical).hexdigest(),'protocol':PROTOCOL}
    path=OUTPUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['sha256']!=result['sha256']:raise ValueError('Do not change a frozen relation protocol in place.')
        return old
    write_json(path,result)
    return result


def predictor(frame):
    total=np.zeros(len(frame),dtype=np.float64)
    for phase in 'RST':
        total += frame[f'voltage{phase}_mean'].to_numpy(dtype=np.float64) * frame[f'current{phase}_mean'].to_numpy(dtype=np.float64)
    return total.reshape(-1,1)


def predict_relation(frame,bundle):
    values=predictor(frame)
    expected=np.full(len(frame),np.nan,dtype=np.float64)
    finite=np.isfinite(values).all(axis=1)
    if finite.any():expected[finite]=bundle['model'].predict(values[finite])
    residual=frame.activePower_mean.to_numpy(dtype=np.float64)-expected
    return np.abs(residual),expected,residual


def fit_and_calibrate(frame,valid):
    valid=np.asarray(valid,dtype=bool)
    ts=frame.timestamp
    train=valid & ts.ge('2024-12-01').to_numpy() & ts.lt('2025-03-01').to_numpy()
    calibration=valid & ts.ge('2025-03-01').to_numpy() & ts.lt('2025-03-16').to_numpy()
    if train.sum()<5000 or calibration.sum()<1000:raise ValueError('Insufficient training/calibration coverage')
    model=LinearRegression(fit_intercept=True,n_jobs=1)
    model.fit(predictor(frame.loc[train]),frame.loc[train,'activePower_mean'].to_numpy(dtype=np.float64))
    bundle={'model':model,'method':METHOD,'coefficient':float(model.coef_[0]),'intercept':float(model.intercept_),
            'fit_start':'2024-12-01','fit_end_exclusive':'2025-03-01','training_minutes':int(train.sum()),
            'calibration_start':'2025-03-01','calibration_end_exclusive':'2025-03-16','calibration_minutes':int(calibration.sum())}
    calibration_scores,_,_=predict_relation(frame.loc[calibration],bundle)
    policies={};grids={}
    for budget in BUDGETS:
        selected,grid=calibrate(calibration_scores,ts.to_numpy()[calibration],budget=budget)
        policies[str(budget)]=selected
        grids[str(budget)]=grid
    bundle['policies']=policies
    bundle['frozen_utc']=datetime.now(timezone.utc).isoformat()
    _,scales=robust_parameters(frame.loc[train,[sensor+'_mean' for sensor in SENSORS]])
    bundle['sensor_scales']=dict(zip(SENSORS,scales))
    return bundle,grids


def regression_metrics(observed,predicted):
    observed=np.asarray(observed,dtype=float);predicted=np.asarray(predicted,dtype=float)
    residual=observed-predicted
    denominator=float(np.sum((observed-observed.mean())**2))
    return {'n':len(observed),'residual_mae_W':float(np.mean(np.abs(residual))),
            'relation_r2':1-float(np.sum(residual**2))/denominator if denominator else None,
            'residual_p95_W':float(np.quantile(np.abs(residual),.95)),
            'residual_max_W':float(np.max(np.abs(residual)))}


def self_test():
    rng=np.random.default_rng(123)
    times=pd.date_range('2024-12-01',periods=6000,freq='min').append(
        pd.date_range('2025-03-01',periods=1440,freq='min')).append(pd.date_range('2025-04-01',periods=60,freq='min'))
    frame=pd.DataFrame({'timestamp':times})
    for sensor in SENSORS:
        frame[sensor+'_mean']=220.0 if sensor.startswith('voltage') else 95.0 if sensor.startswith('powerFactor') else rng.uniform(1,8,len(frame))
    frame['activePower_mean']=3.0+.27*predictor(frame)[:,0]+rng.normal(0,1,len(frame))
    valid=np.ones(len(frame),dtype=bool)
    first,_=fit_and_calibrate(frame,valid)
    changed=frame.copy(deep=True)
    april=changed.timestamp.ge('2025-04-01')
    changed.loc[april,'activePower_mean']*=100
    changed.loc[april,'currentR_mean']*=30
    second,_=fit_and_calibrate(changed,valid)
    assert first['coefficient']==second['coefficient']
    assert first['intercept']==second['intercept']
    assert first['policies']==second['policies']
    assert first['sensor_scales']==second['sensor_scales']
    original_score,_,_=predict_relation(frame,first)
    changed_score,_,_=predict_relation(changed,first)
    assert np.array_equal(original_score[~april],changed_score[~april])
    frozen=joblib.hash(first)
    predict_relation(changed,first)
    assert frozen==joblib.hash(first)
    return {'passed':True,'checks':['April values cannot change train coefficients or March thresholds.',
            'April changes cannot alter scores of earlier rows.','Scoring does not mutate the frozen model.']}


def compile_summary(real_rows,synthetic_rows):
    real=pd.DataFrame(real_rows)
    periods={}
    for period,rows in real.loc[real.budget.eq(1.0)].groupby('period'):
        hours=float(rows.observed_equipment_hours.sum());flags=int(rows.flagged_minutes.sum());minutes=int(rows.complete_minutes.sum())
        periods[period]={'equipment_count':int(rows.module.nunique()),'alerts':int(rows.alert_count.sum()),
                         'observed_equipment_hours':hours,'alerts_per_24_equipment_hours':float(rows.alert_count.sum())/hours*24 if hours else None,
                         'flagged_minutes':flags,'flagged_fraction':flags/minutes if minutes else None,
                         'longest_flagged_run_minutes':int(rows.longest_flagged_run_minutes.max()),
                         'relation_r2_min':float(rows.relation_r2.min()),'relation_r2_median':float(rows.relation_r2.median()),
                         'residual_mae_W_min':float(rows.residual_mae_W.min()),'residual_mae_W_max':float(rows.residual_mae_W.max())}
    primary=[row for row in synthetic_rows if row['budget']==1.0]
    return {'method':METHOD,'primary_budget':1.0,'real':periods,
            'synthetic':summarize_synthetic_results(primary),'limitations':PROTOCOL['limitations'],
            'separate_followup_experiment':True,'original_four_methods_not_changed':True}


def main():
    start=time.perf_counter()
    protocol=freeze_protocol()
    tests=self_test()
    write_json(OUTPUT/'self_test.json',tests)
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):raise ValueError('Minute source incomplete')
    MODELS.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/minute.parquet'
    source_modules=pq.read_table(source,columns=['module(equipment)'])
    modules=sorted(pc.unique(source_modules['module(equipment)']).to_pylist())
    del source_modules
    real_rows=[];synthetic_rows=[];review_rows=[];threshold_rows=[];training_audit=[];all_events=[];probe_manifest=[]
    for module in modules:
        tick=time.perf_counter()
        frame=pd.read_parquet(source,filters=[('module(equipment)','==',module)]).rename(columns={'module(equipment)':'module'})
        frame=frame.sort_values('timestamp').reset_index(drop=True)
        frame['timestamp']=pd.to_datetime(frame.timestamp)
        if frame.timestamp.max()>=pd.Timestamp('2025-05-01'):raise ValueError('May forbidden')
        features=build_features(frame)
        valid=features.valid.to_numpy(dtype=bool)
        del features
        bundle,grids=fit_and_calibrate(frame,valid)
        bundle.update({'module':module,'protocol_sha256':protocol['sha256']})
        safe=''.join(c if c.isalnum() else '_' for c in module)
        joblib.dump(bundle,MODELS/f'{safe}.joblib',compress=3)
        write_json(OUTPUT/f'calibration_{safe}.json',{'module':module,'policies':bundle['policies'],'grids':grids})
        scores,predicted,residual=predict_relation(frame,bundle)
        fit_audit={'module':module,'coefficient':bundle['coefficient'],'intercept_W':bundle['intercept'],
                   'training_minutes':bundle['training_minutes'],'calibration_minutes':bundle['calibration_minutes'],
                   'fit_period':[bundle['fit_start'],bundle['fit_end_exclusive']],
                   'thresholds_frozen_before_evaluation':True,'periods':{}}
        for period,begin,end in (('train_dec_feb','2024-12-01','2025-03-01'),('march_calibration','2025-03-01','2025-03-16'),
                                 ('march_check','2025-03-16','2025-04-01'),('april_development','2025-04-01','2025-04-30')):
            mask=frame.timestamp.ge(begin)&frame.timestamp.lt(end)&valid
            fit_audit['periods'][period]=regression_metrics(frame.loc[mask,'activePower_mean'],predicted[mask])
        training_audit.append(fit_audit)
        for budget in BUDGETS:
            policy=bundle['policies'][str(budget)];threshold=policy['threshold']
            threshold_rows.append({'module':module,'method':METHOD,'budget':budget,**policy})
            flags=(scores>=threshold)&valid
            for period,begin,end in (('march_check','2025-03-16','2025-04-01'),('april_development','2025-04-01','2025-04-30')):
                mask=frame.timestamp.ge(begin)&frame.timestamp.lt(end)&valid
                subset=frame.loc[mask]
                summary=summarize_alerts(subset,flags[mask],min_consecutive=3)
                real_rows.append({'module':module,'method':METHOD,'budget':budget,'period':period,'valid_minutes':int(mask.sum()),
                    **summary,'flagged_fraction':summary['flagged_minutes']/summary['complete_minutes'] if summary['complete_minutes'] else None,
                    **regression_metrics(frame.loc[mask,'activePower_mean'],predicted[mask])})
                if budget==1.0:
                    events=group_alert_events(subset,flags[mask],min_consecutive=3)
                    if len(events):
                        events['method']=METHOD;events['period']=period
                        all_events.append(events)
                    if period=='april_development':
                        for _,event in events.head(4).iterrows():
                            event_mask=frame.timestamp.ge(event.start)&frame.timestamp.lt(event.end)&valid
                            positions=np.flatnonzero(event_mask)
                            pos=positions[np.argmax(scores[positions])]
                            review_rows.append({'module':module,'method':METHOD,'start':event.start.isoformat(),'end':event.end.isoformat(),
                                'decision_time':event.decision_time.isoformat(),'duration_minutes':int(event.duration_minutes),
                                'peak_timestamp':frame.timestamp.iloc[pos].isoformat(),'observed_power_W':float(frame.activePower_mean.iloc[pos]),
                                'relationship_expected_power_W':float(predicted[pos]),'signed_residual_W':float(residual[pos]),
                                'absolute_residual_W':float(scores[pos]),'threshold_W':threshold,
                                'interpretation':'Unreviewed observed sensor-relation deviation; no confirmed fault or diagnosis.'})
        april=frame.loc[frame.timestamp.ge('2025-04-01')].copy()
        generation_audit={}
        probes=generate_synthetic_probes(april,{str(module):bundle['sensor_scales']},max_per_type=1,seed=SEED,generation_audit=generation_audit)
        write_json(OUTPUT/f'probe_generation_{safe}.json',{'module':module,**generation_audit})
        for probe in probes:
            probe['probe_id']=f"{module}:{probe['probe_id']}"
            baseline,injected=probe['baseline'],probe['injected']
            baseline_valid=build_features(baseline).valid.to_numpy(dtype=bool)
            injected_valid=build_features(injected).valid.to_numpy(dtype=bool)
            baseline_scores,_,_=predict_relation(baseline,bundle)
            injected_scores,_,_=predict_relation(injected,bundle)
            for budget in BUDGETS:
                threshold=bundle['policies'][str(budget)]['threshold']
                result=evaluate_synthetic_probe(probe,(baseline_scores>=threshold)&baseline_valid,(injected_scores>=threshold)&injected_valid)
                synthetic_rows.append({'method':METHOD,'budget':budget,**result})
            probe_manifest.append({key:value for key,value in probe.items() if key not in ('baseline','injected')})
        pd.DataFrame(real_rows).to_csv(OUTPUT/'real_alert_metrics.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(synthetic_rows).to_csv(OUTPUT/'synthetic_results.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(threshold_rows).to_csv(OUTPUT/'thresholds.csv',index=False,encoding='utf-8-sig')
        write_json(OUTPUT/'training_audit.json',training_audit)
        write_json(OUTPUT/'synthetic_manifest.json',probe_manifest)
        print(f'{module}: {int(valid.sum()):,} common valid minutes, {len(probes)} probes, {time.perf_counter()-tick:.2f}s',flush=True)
        del frame,april,probes,scores,predicted,residual,valid,bundle
        gc.collect()
    pd.DataFrame(review_rows).to_csv(OUTPUT/'review_queue.csv',index=False,encoding='utf-8-sig')
    if all_events:pd.concat(all_events,ignore_index=True).to_csv(OUTPUT/'events.csv',index=False,encoding='utf-8-sig')
    summary=compile_summary(real_rows,synthetic_rows)
    write_json(OUTPUT/'summary.json',summary)
    source_hashes={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in ('run_relation_probe.py','context_features.py','evaluation.py','run_experiment.py')}
    write_json(OUTPUT/'run_manifest.json',{'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':protocol['sha256'],'method':METHOD,'equipment_count':len(modules),'modules':modules,
        'source_code_hashes':source_hashes,'self_test':tests,'elapsed_seconds':time.perf_counter()-start,
        'memory_strategy':'One equipment per load; only valid bit retained from shared feature matrix. Single BLAS thread.',
        'original_experiment_modified':False,'dashboard_modified':False,'api_calls':0,
        'notice':'Exploratory follow-up on observed sensor relationships; not final holdout or real fault accuracy.'})
    print(json.dumps({'complete':True,'elapsed_seconds':round(time.perf_counter()-start,2),'primary_results':summary},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        if '--self-test' in sys.argv:
            print(json.dumps(self_test(),ensure_ascii=False))
        else:
            main()
