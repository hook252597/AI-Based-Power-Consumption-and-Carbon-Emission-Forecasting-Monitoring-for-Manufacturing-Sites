from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_experiment import calibrate,event_count_fast,robust_parameters
from evaluation import group_alert_events

def test_fast_calibration_counter_matches_causal_grouping_with_gaps():
    rng=np.random.default_rng(31)
    t=pd.date_range('2025-03-02',periods=300,freq='min').delete([20,21,140])
    flags=rng.random(len(t))>.4
    frame=pd.DataFrame({'module':'x','timestamp':t,'n':12})
    assert event_count_fast(flags,t)==len(group_alert_events(frame,flags))

def test_three_minute_alert_not_a_single_spike():
    t=pd.date_range('2025-03-02',periods=8,freq='min')
    assert event_count_fast([0,1,0,0,1,1,0,0],t)==0
    assert event_count_fast([0,1,1,1,0,1,1,1],t)==2

def test_calibration_is_label_free_deterministic_and_budget_reported():
    t=pd.date_range('2025-03-01',periods=3000,freq='min')
    x=np.random.default_rng(19).normal(size=len(t))
    a,grid=calibrate(x,t,1.)
    b,_=calibrate(x,t,1.)
    assert a==b
    assert a['target_events_per_day']==1.
    assert a['events']==event_count_fast(x>=a['threshold'],t)
    assert a['events_per_day']==a['events']/(len(t)/1440)

def test_robust_scale_cannot_divide_by_zero():
    center,scale=robust_parameters(np.zeros((10,3)))
    assert np.all(center==0) and np.all(scale>0)
