"""Calendar-only multi-week forecasts; no future sensor or ground-truth inputs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PROJECT = Path(__file__).resolve().parents[2]
MODULE = 'module(equipment)'
RATE = 180.0
CARBON = 0.424
SEED = 42
MODULE_FEATURE_PREFIX = 'equipment_'

def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def physical_energy(power_sum):
    """Sum of 5-second mean watts -> kWh. Complete hours are checked separately."""
    return np.asarray(power_sum, dtype=float) / 720_000.0

def load_hourly(path):
    d = pd.read_csv(path, parse_dates=['datetime'])
    required={MODULE, 'datetime', 'n', 'power_count', 'power_sum'}
    if not required.issubset(d): raise ValueError(f'Missing hourly columns: {required-set(d)}')
    if d.duplicated([MODULE,'datetime']).any(): raise ValueError('Duplicate module/localtime hour')
    modules=sorted(d[MODULE].unique())
    if d.empty:raise ValueError('No hourly data')
    complete=(d.n==720)&(d.power_count==720)
    if ((~complete)&d.datetime.ne(d.datetime.max())).any():
        raise ValueError('Incomplete interior hour: resolve missing data before fitting')
    if (d.n>720).any():raise ValueError('Too many measurements in an hour')
    d=d.loc[complete].copy()
    if d.empty:raise ValueError('No complete hourly data')
    if not np.isfinite(d.power_sum).all(): raise ValueError('Nonfinite power totals')
    if (d.power_sum<0).any(): raise ValueError('Negative energy requires explicit handling')
    d['energy_kWh']=physical_energy(d.power_sum)
    if not d.groupby('datetime')[MODULE].nunique().eq(len(modules)).all():
        raise ValueError('Missing equipment hour: do not silently produce factory totals')
    expected=pd.date_range(d.datetime.min(),d.datetime.max(),freq='h')
    actual=pd.DatetimeIndex(d.datetime.drop_duplicates().sort_values())
    if not actual.equals(expected):raise ValueError('Missing entire factory hour')
    return d.sort_values(['datetime',MODULE]).reset_index(drop=True)

def calendar_features(frame, modules):
    """Features are known at forecast origin, regardless of the forecast horizon."""
    t=pd.to_datetime(frame['datetime'])
    unknown=set(frame[MODULE])-set(modules)
    if unknown: raise ValueError(f'Unknown equipment: {unknown}')
    x=pd.DataFrame({'hour':t.dt.hour,'weekday':t.dt.dayofweek,
                    'weekend':(t.dt.dayofweek>=5).astype(int)},index=frame.index)
    for i,mod in enumerate(modules): x[f'{MODULE_FEATURE_PREFIX}{i}']=(frame[MODULE]==mod).astype(int)
    return x.astype(float)

def grid_for_dates(start, end, modules):
    dates=pd.date_range(start,end,freq='h',inclusive='both')
    return pd.MultiIndex.from_product([dates,modules],names=['datetime',MODULE]).to_frame(index=False)

@dataclass
class ForecastModel:
    kind: str
    modules: list
    last_training_time: str
    values: dict | None = None
    estimator: object | None = None
    parameters: dict | None = None

    def predict(self, frame):
        if set(frame[MODULE])-set(self.modules): raise ValueError('Unknown equipment')
        if self.kind in ['mean','median']:
            result=frame[MODULE].map(self.values).to_numpy(dtype=float)
        else:
            result=np.asarray(self.estimator.predict(calendar_features(frame,self.modules)),dtype=float)
        if not np.isfinite(result).all():raise ValueError('Nonfinite model output')
        # Physical constraint is fixed before any test evaluation.
        return np.maximum(result,0.0)

def fit_model(train, kind, params=None):
    modules=sorted(train[MODULE].unique())
    model=ForecastModel(kind,modules,str(train.datetime.max()),parameters=params)
    if kind in ['mean','median']:
        model.values=getattr(train.groupby(MODULE).energy_kWh,kind)().to_dict()
    else:
        options=dict(objective='reg:absoluteerror',eval_metric='mae',tree_method='hist',
                     random_state=SEED,n_jobs=2,subsample=1.,colsample_bytree=1.)
        options.update(params or {})
        model.estimator=XGBRegressor(**options)
        model.estimator.fit(calendar_features(train,modules),train.energy_kWh)
    return model

def score_predictions(frame, prediction):
    d=frame[['datetime',MODULE,'energy_kWh']].copy()
    d['prediction_kWh']=prediction
    totals=d.groupby('datetime')[['energy_kWh','prediction_kWh']].sum()
    error=(totals.prediction_kWh-totals.energy_kWh).abs()
    module_mae=d.assign(error=(d.prediction_kWh-d.energy_kWh).abs()).groupby(MODULE).error.mean()
    metrics={'factory_mae_kWh':float(error.mean()),
             'normalized_mae':float(error.mean()/totals.energy_kWh.mean()),
             'factory_bias_kWh':float((totals.prediction_kWh-totals.energy_kWh).mean()),
             'module_mae_mean_kWh':float(module_mae.mean()),'hours':len(totals),
             'module_mae_kWh':module_mae.to_dict()}
    return metrics,d,totals

def evaluate(model, data, focus_start=None, focus_end=None):
    if len(data)==0: raise ValueError('Empty evaluation window')
    if pd.Timestamp(model.last_training_time)>=data.datetime.min():
        raise ValueError('Training and evaluation overlap')
    pred=model.predict(data)
    all_metrics,detail,totals=score_predictions(data,pred)
    mask=np.ones(len(data),dtype=bool)
    if focus_start is not None: mask&=data.datetime.ge(focus_start).to_numpy()
    if focus_end is not None: mask&=data.datetime.le(focus_end).to_numpy()
    focus,_,_=score_predictions(data.loc[mask],pred[mask])
    return {'all':all_metrics,'focus':focus},detail,totals

def export_prediction(model,start,end,out):
    grid=grid_for_dates(start,end,model.modules)
    grid['prediction_kWh']=model.predict(grid)
    physical=grid.groupby('datetime').prediction_kWh.sum().to_frame('energy_kWh')
    physical['cost_KRW_estimate']=physical.energy_kWh*RATE
    physical['carbon_kgCO2_estimate']=physical.energy_kWh*CARBON
    native=physical[['energy_kWh']].rename(columns={'energy_kWh':'hourly_pow'})*1000
    native['may_bill']=native.hourly_pow*RATE
    native['may_carbon']=native.hourly_pow*CARBON
    # agg_pow is omitted: challenge metadata does not need to be invented for scoring.
    native.index.name='id'
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    grid.to_csv(out/'may_equipment_physical.csv',index=False,encoding='utf-8-sig')
    physical.to_csv(out/'may_physical.csv',encoding='utf-8-sig')
    native.to_csv(out/'may_reference_convention.csv',encoding='utf-8-sig')
    return physical,native

def validate_reference(reference):
    required={'id','hourly_pow','may_bill','may_carbon'}
    if not required.issubset(reference): raise ValueError('Reference schema mismatch')
    reference=reference.copy();reference['id']=pd.to_datetime(reference.id)
    expected=pd.date_range('2025-05-10 00:00','2025-05-31 00:00',freq='h')
    if reference.id.duplicated().any() or not pd.DatetimeIndex(reference.id).equals(expected):
        raise ValueError('Reference must have the fixed 505 hourly timestamps, ordered and unique')
    if not np.isfinite(reference[['hourly_pow','may_bill','may_carbon']]).all().all():
        raise ValueError('Reference contains nonfinite values')
    if not np.allclose(reference.may_bill,reference.hourly_pow*RATE) or not np.allclose(reference.may_carbon,reference.hourly_pow*CARBON):
        raise ValueError('Reference coefficient identity mismatch')
    return reference.set_index('id')

def select_final_method(march_winner, best_baseline, april_ml_gate_passed):
    """Predeclared policy: April can reject March's ML winner, not promote a loser."""
    if march_winner['kind']=='xgboost' and not april_ml_gate_passed:
        return best_baseline
    return march_winner

def run():
    start=time.time()
    out=PROJECT/'outputs/forecast';out.mkdir(parents=True,exist_ok=True)
    models=PROJECT/'models/forecast';models.mkdir(parents=True,exist_ok=True)
    d=load_hourly(PROJECT/'reports/eda/hourly_observed.csv')
    train=d[d.datetime<'2025-03-01'];march=d[(d.datetime>='2025-03-01')&(d.datetime<'2025-04-01')]
    april=d[d.datetime>='2025-04-01']
    policy={'seed':SEED,'selection_metric':'March 10 00:00 through March 31 00:00 factory MAE',
      'april_focus':'April 10 00:00 through April 29 23:00',
      'physical_unit_assumption':'activePower is W; evidence: course data-rules and accumulator increments',
      'reference_convention':'physical kWh multiplied by 1000 for supplied hourly_pow; multiply native number by180/.424 for supplied reference columns. This is NOT a physical bill.',
      'unit_contract_confirmed_by_provider':False,
      'minimum_normalized_mae':0.01,'model_adoption':'XGBoost must not exceed best simple baseline MAE on April focus; otherwise predeclared fallback to March-selected best baseline',
      'stretch_improvement':0.05,'future_sensor_inputs':False,'random_split':False,
      'test_note':'April was included in prior EDA; internal test is not a fully unseen dataset. May values are read only after model freeze.'}
    save_json(out/'experiment_policy.json',policy)
    configurations=[
      {'max_depth':2,'n_estimators':100,'learning_rate':.03,'min_child_weight':100,'reg_lambda':20},
      {'max_depth':2,'n_estimators':300,'learning_rate':.03,'min_child_weight':100,'reg_lambda':20},
      {'max_depth':3,'n_estimators':150,'learning_rate':.03,'min_child_weight':100,'reg_lambda':20},
      {'max_depth':3,'n_estimators':300,'learning_rate':.03,'min_child_weight':200,'reg_lambda':50},
      {'max_depth':4,'n_estimators':150,'learning_rate':.03,'min_child_weight':200,'reg_lambda':50},
    ]
    experiments=[]
    for name,kind,params in [('mean','mean',None),('median','median',None)]+[(f'xgb_{i+1}','xgboost',p) for i,p in enumerate(configurations)]:
        model=fit_model(train,kind,params)
        met,_,_=evaluate(model,march,'2025-03-10','2025-03-31')
        item={'name':name,'kind':kind,'parameters':params,'march':met}
        experiments.append(item)
        print(f"March {name}: factory MAE focus={met['focus']['factory_mae_kWh']:.6f} kWh",flush=True)
    metric=lambda x:x['march']['focus']['factory_mae_kWh']
    best_base=min([x for x in experiments if x['kind']!='xgboost'],key=metric)
    best_ml=min([x for x in experiments if x['kind']=='xgboost'],key=metric)
    march_winner=min([best_base,best_ml],key=metric)
    save_json(out/'march_selection.json',{'candidates':experiments,'best_baseline':best_base['name'],
      'best_xgboost':best_ml['name'],'march_winner':march_winner['name']})
    train_april=d[d.datetime<'2025-04-01']
    april_results={}
    for spec in [next(x for x in experiments if x['name']=='mean'),next(x for x in experiments if x['name']=='median'),best_ml]:
        model=fit_model(train_april,spec['kind'],spec['parameters'])
        met,detail,totals=evaluate(model,april,'2025-04-10','2025-04-29 23:00')
        april_results[spec['name']]=met
        totals.to_csv(out/f"april_{spec['name']}.csv",encoding='utf-8-sig')
        pd.Series(met['all']['module_mae_kWh'],name='MAE_kWh').to_csv(out/f"april_equipment_mae_{spec['name']}.csv",encoding='utf-8-sig')
        print(f"April {spec['name']}: factory MAE focus={met['focus']['factory_mae_kWh']:.6f} kWh",flush=True)
    bmae=min(april_results[n]['focus']['factory_mae_kWh'] for n in ['mean','median'])
    xmae=april_results[best_ml['name']]['focus']['factory_mae_kWh']
    adoption=xmae<=bmae
    selected=select_final_method(march_winner,best_base,adoption)
    final_model=fit_model(d,selected['kind'],selected['parameters'])
    final_ml=fit_model(d,best_ml['kind'],best_ml['parameters'])
    joblib.dump(final_model,models/'selected_model.joblib')
    joblib.dump(final_ml,models/'xgboost_model.joblib')
    final_ml.estimator.save_model(models/'xgboost_native.json')
    final_ml_features=calendar_features(d.iloc[:1],final_ml.modules).columns
    pd.Series(final_ml.estimator.feature_importances_,index=final_ml_features,name='importance').to_csv(out/'xgboost_feature_importance.csv',encoding='utf-8-sig')
    selection={'selected_method':selected['name'],'selected_kind':selected['kind'],'selected_params':selected['parameters'],
      'ml_candidate':best_ml['name'],'ml_adoption_gate_passed':adoption,
      'ml_improvement_over_best_april_baseline_pct':float((bmae-xmae)/bmae*100),
      'april_results':april_results,'final_training_start':str(d.datetime.min()),'final_training_end':str(d.datetime.max()),
      'final_training_rows':len(d),'module_count':len(final_model.modules),
      'selection_rule':'March selection, followed only by predeclared April non-inferiority fallback; no May tuning.',
      'frozen_before_reference_read':True,
      'model_sha256':hashlib.sha256((models/'selected_model.joblib').read_bytes()).hexdigest()}
    # Persist the fixed decision and forecasts before opening the May answer file.
    save_json(out/'model_freeze.json',selection)
    phys,native=export_prediction(final_model,'2025-05-10','2025-05-31',out/'selected')
    _,ml_native=export_prediction(final_ml,'2025-05-10','2025-05-31',out/'xgboost')
    reference=validate_reference(pd.read_csv(PROJECT.parent/'프로젝트/data/rtu_ground_truth_may.csv'))
    evaluations={}
    for label,pred in [('selected',native),('xgboost',ml_native)]:
        joined=reference[['hourly_pow','may_bill','may_carbon']].join(pred,lsuffix='_actual',rsuffix='_predicted',validate='one_to_one')
        if joined.isna().any().any():raise ValueError('Unmatched forecast/reference rows')
        vals={c:float((joined[c+'_actual']-joined[c+'_predicted']).abs().mean()) for c in ['hourly_pow','may_bill','may_carbon']}
        nmae=vals['hourly_pow']/float(reference.hourly_pow.mean())
        evaluations[label]={'rows':len(joined),'mae_reference_units':vals,'normalized_mae':nmae,
           'mae_kWh_if_reference_hourly_pow_is_Wh':vals['hourly_pow']/1000,
           'basic_error_target_passed':bool(nmae<=.01),'unit_confirmation_pending':True}
        joined.to_csv(out/f'may_comparison_{label}.csv',encoding='utf-8-sig')
    summary={'policy':policy,'selection':selection,'may':evaluations,
      'success':{'functional':len(native)==505 and bool(np.isfinite(native).all().all()),
        'ml_non_inferiority_on_april':adoption,'ml_5pct_improvement':bool(xmae<=bmae*.95),
        'selected_basic_error_goal':evaluations['selected']['basic_error_target_passed'],
        'official_physical_unit_acceptance':'pending provider confirmation'},'elapsed_seconds':time.time()-start}
    save_json(out/'results.json',summary)
    print(json.dumps({'selected':selected['name'],'may':evaluations,'success':summary['success']},ensure_ascii=False,indent=2),flush=True)
    # Validate the serialized artifact using the same prediction grid.
    loaded=joblib.load(models/'selected_model.joblib')
    probe=grid_for_dates('2025-05-10','2025-05-10 03:00',loaded.modules)
    if not np.array_equal(loaded.predict(probe),final_model.predict(probe)):raise AssertionError('Saved-model inference mismatch')
    return summary

if __name__=='__main__':run()
