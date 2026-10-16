"""Causal sensor review, independent phases and immutable model evidence."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import pytest
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from monitoring.sensor_insights import DEFAULT_POLICY,normalize_policy,evaluate_sensor_insights,get_sensor_insights
from monitoring.alert_service import MODULE,sha256


def raw(n,flags=None,start='2025-04-01',seconds=None):
 t=pd.date_range(start,periods=n,freq='5s') if seconds is None else pd.Timestamp(start)+pd.to_timedelta(seconds,unit='s')
 f=pd.DataFrame({MODULE:'13(3호기)','datetime':t,'ai_flag':[True]*n if flags is None else flags,'ai_score':.7,
  **{s:215. for s in ['voltageR','voltageS','voltageT']},**{s:95. for s in ['powerFactorR','powerFactorS','powerFactorT']}})
 return f


def kind(r,k): return [e for e in r['events'] if e['kind']==k]


def test_input_policy_and_ai_logs_preserved():
 f=raw(13,[True]*2+[False]+[True]*10);f['voltageR']=197.;before=f.copy(deep=True);p=dict(DEFAULT_POLICY)
 r=evaluate_sensor_insights(f,p);pd.testing.assert_frame_equal(f,before);assert p==DEFAULT_POLICY
 assert r['summary']['ai_record_count']==12 and r['summary']['original_event_count']==2
 assert all(x['preserved'] for x in r['all_ai_logs'])
 assert sum(x['ai_records'] for x in r['all_ai_logs'])==12


@pytest.mark.parametrize('v,expected',[(198,0),(197.99,1),(242,0),(242.01,1)])
def test_voltage_severe_strict_limits(v,expected):
 f=raw(1);f['voltageR']=v;r=evaluate_sensor_insights(f)
 assert len(kind(r,'voltage_severe'))==expected


def test_severe_requires_ai_at_selected_sample_and_one_event_per_run():
 f=raw(20,[False]*4+[True]+[False]*15);f['voltageR']=197
 r=evaluate_sensor_insights(f);e=kind(r,'voltage_severe')
 assert len(e)==1 and e[0]['trigger_at'].endswith('00:00:25')
 assert e[0]['duration_seconds']==100
 f['ai_flag']=False;assert not evaluate_sensor_insights(f)['events']


def test_sustained_uses_sensor_continuity_through_ai_negative():
 f=raw(13,[True]+[False]*12);f['voltageR']=205
 assert not evaluate_sensor_insights(f,cutoff='2025-04-01 00:00:59')['events']
 e=kind(evaluate_sensor_insights(f,cutoff='2025-04-01 00:01:00'),'voltage_sustained')[0]
 assert e['trigger_at'].endswith('00:01:00') and e['duration_seconds']==60
 assert len(e['trigger_ai_event_ids'])==1


def test_late_ai_after_duration_triggers_when_evidence_arrives():
 f=raw(20,[False]*15+[True]+[False]*4);f['voltageR']=205
 e=kind(evaluate_sensor_insights(f),'voltage_sustained');assert len(e)==1
 assert e[0]['trigger_at'].endswith('00:01:20')


def test_phases_cannot_accumulate_each_others_duration():
 f=raw(24);f.loc[0:10,'voltageR']=205;f.loc[11:21,'voltageS']=205
 assert not kind(evaluate_sensor_insights(f),'voltage_sustained')


@pytest.mark.parametrize('change',['gap','normal','nan','day'])
def test_sensor_continuity_breaks(change):
 f=raw(24);f['voltageR']=205
 if change=='gap': f.loc[11:,'datetime']+=pd.Timedelta(seconds=5)
 if change=='normal': f.loc[11,'voltageR']=215
 if change=='nan': f.loc[11,'voltageR']=np.nan
 if change=='day': f['datetime']=pd.date_range('2025-04-01 23:59:05',periods=24,freq='5s')
 # At most 12 readings after the break in 24 rows; limit to 22 makes both pieces shorter than60s.
 f=f.iloc[:22]
 assert not kind(evaluate_sensor_insights(f),'voltage_sustained')


def test_pf_drop_needs_sixty_previous_samples_current_excluded():
 f=raw(61);f['powerFactorR']=90.;f.loc[60,'powerFactorR']=69.
 assert not kind(evaluate_sensor_insights(f.iloc[:60]),'pf_sharp')
 e=kind(evaluate_sensor_insights(f),'pf_sharp')[0]
 assert e['baseline']==90 and e['drop_pp']==21 and e['trigger_at'].endswith('00:05:05')
 f.loc[59,'powerFactorR']=69
 assert not kind(evaluate_sensor_insights(f.iloc[:60]),'pf_sharp')


@pytest.mark.parametrize('baseline,current,expected',[(89,69,1),(88.99,69,0),(95,70,0)])
def test_pf_drop_percentage_points_and_strict_low(baseline,current,expected):
 f=raw(61);f['powerFactorR']=baseline;f.loc[60,'powerFactorR']=current
 assert len(kind(evaluate_sensor_insights(f),'pf_sharp'))==expected


def test_pf_baseline_same_phase_valid_contiguous_and_not_future():
 f=raw(61);f['powerFactorR']=85;f['powerFactorS']=100;f.loc[60,'powerFactorR']=69
 assert not kind(evaluate_sensor_insights(f),'pf_sharp')
 f['powerFactorR']=95;f.loc[20,'powerFactorR']=np.nan;f.loc[60,'powerFactorR']=69
 assert not kind(evaluate_sensor_insights(f),'pf_sharp')
 f=raw(61);f.loc[20:,'datetime']+=pd.Timedelta(seconds=5);f.loc[60,'powerFactorR']=69
 assert not kind(evaluate_sensor_insights(f),'pf_sharp')


def test_pf_baseline_fixed_through_long_drop_and_single_emission():
 f=raw(200);f.loc[60:,'powerFactorR']=69
 e=kind(evaluate_sensor_insights(f),'pf_sharp')
 assert len(e)==1 and e[0]['baseline']==95 and e[0]['duration_seconds']==700
 assert len(kind(evaluate_sensor_insights(f),'pf_sustained'))==1


def test_pf_baseline_resets_at_day_boundary():
 f=raw(61,start='2025-04-01 23:55:00');f.loc[60,'powerFactorR']=69
 assert not kind(evaluate_sensor_insights(f),'pf_sharp')


def test_pf_sustained_one_evidence_and_three_minutes():
 f=raw(36,[True]+[False]*35);f['powerFactorT']=79
 e=kind(evaluate_sensor_insights(f),'pf_sustained')[0]
 assert e['item']=='powerFactorT' and e['trigger_at'].endswith('00:03:00')
 assert not kind(evaluate_sensor_insights(f,cutoff='2025-04-01 00:02:59'),'pf_sustained')
 f['powerFactorT']=80;assert not kind(evaluate_sensor_insights(f),'pf_sustained')


def test_cutoff_invariance_and_prefix_no_future_leakage():
 f=raw(200);f.loc[60:,'powerFactorR']=69
 stop='2025-04-01 00:06:00';a=evaluate_sensor_insights(f,cutoff=stop)
 future=f.copy();future.loc[72:,'powerFactorR']=10;future.loc[72:,'ai_flag']=False
 b=evaluate_sensor_insights(future,cutoff=stop)
 assert a==b
 prefix=evaluate_sensor_insights(f.iloc[:72]);assert a==prefix


def test_missing_day_and_empty_frame_unknown():
 r=evaluate_sensor_insights(raw(0));assert not r['data_available']
 assert r['summary']['event_count'] is None and len(r['per_equipment'])==13
 r=evaluate_sensor_insights(raw(1),cutoff='2025-04-01 00:00:02');assert not r['data_available']


def test_group_silence_exact_boundary_direction_and_phase_separate():
 f=raw(241);f.loc[[0,60,180],'voltageR']=197 #0,5,15min; exact10 gap splits
 f.loc[60,'voltageS']=197;f.loc[120,'voltageR']=243 #10min high separate
 r=evaluate_sensor_insights(f);groups=[g for g in r['groups'] if g['kind']=='voltage_severe']
 assert len(groups)==4
 low=[g for g in groups if g['item']=='voltageR' and g['direction']=='low']
 assert [g['event_count'] for g in low]==[2,1]


@pytest.mark.parametrize('policy',[{'voltage_low':198},{'voltage_high':242},{'pf_low':101},{'pf_sharp_low':80},
 {'pf_drop_pp':0},{'baseline_seconds':0},{'voltage_duration_seconds':61},{'merge_minutes':-1},{'pf_low':True},{'pf_low':float('nan')},{'unknown':1}])
def test_policy_validation(policy):
 with pytest.raises(ValueError):normalize_policy(policy)


def make_cache(tmp_path,monkeypatch):
 project=tmp_path/'project';cache=project/'outputs/agent_data';(cache/'daily').mkdir(parents=True)
 model=project/'model.joblib';model.write_bytes(b'fixture-model-not-deserialized')
 f=raw(61);f['voltageR']=197;partition=cache/'daily/2025-04-01.parquet';f.to_parquet(partition,index=False)
 manifest={'complete':True,'signature':{'sample_seconds':5,'model_sha256':sha256(model)},'model_file':'model.joblib','source_sha256':'fixture',
  'dates':{'2025-04-01':{'complete':True,'file':'daily/2025-04-01.parquet','rows':len(f),'size':partition.stat().st_size,'sha256':sha256(partition),'per_equipment':{'13(3호기)':len(f)}}}}
 (cache/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8');monkeypatch.setenv('MARGARET_SENSOR_CACHE',str(cache))
 return cache,model,partition


def test_loader_good_missing_day_and_before_completed_sample(tmp_path,monkeypatch):
 make_cache(tmp_path,monkeypatch)
 r=get_sensor_insights('2025-04-01');assert r['status']=='ok' and r['summary']['ai_record_count']==61
 assert not get_sensor_insights('2025-05-31')['data_available']
 assert not get_sensor_insights('2025-04-01',cutoff='2025-04-01 00:00:02')['data_available']
 assert len(r['per_equipment'])==13 and sum(x['data_available'] for x in r['per_equipment'])==1


@pytest.mark.parametrize('target',['model','partition','manifest'])
def test_loader_corruption_fail_closed(tmp_path,monkeypatch,target):
 cache,model,partition=make_cache(tmp_path,monkeypatch)
 assert get_sensor_insights('2025-04-01')['data_available']
 if target=='model':model.write_bytes(b'changed-model')
 if target=='partition':partition.write_bytes(b'changed-partition')
 if target=='manifest':(cache/'manifest.json').write_text('{}',encoding='utf-8')
 r=get_sensor_insights('2025-04-01');assert not r['data_available'] and r['summary']['event_count'] is None
