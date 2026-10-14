import numpy as np
import pandas as pd
import pytest
from manufacturing_ai.forecast import (MODULE, physical_energy, calendar_features, grid_for_dates,
                                       fit_model,evaluate,load_hourly,validate_reference,
                                       export_prediction,select_final_method)

def test_watts_energy_conversion():
    # One hour continuously at 1kW -> 1kWh, not 1,000kWh.
    assert physical_energy([1000*720])[0]==pytest.approx(1.)

def test_features_do_not_consume_future_measurements():
    d=grid_for_dates('2025-05-10','2025-05-10 02:00',['A','B'])
    expected=calendar_features(d,['A','B'])
    for col in ['energy_kWh','activePower','currentR','is_outlier','outlier_reason']:
        d[col]=999999
    pd.testing.assert_frame_equal(expected,calendar_features(d,['A','B']))

def test_may_grid_end_inclusive():
    d=grid_for_dates('2025-05-10','2025-05-31',['A','B'])
    assert len(d)==505*2
    assert not d.duplicated(['datetime',MODULE]).any()

def test_overlap_is_rejected():
    d=grid_for_dates('2025-01-01','2025-01-02',['A'])
    d['energy_kWh']=3.
    model=fit_model(d,'mean')
    with pytest.raises(ValueError,match='overlap'):evaluate(model,d)

def test_partial_hours_excluded_and_missing_equipment_rejected(tmp_path):
    d=pd.DataFrame({MODULE:['A','B','A','B'],'datetime':['2025-01-01','2025-01-01','2025-01-01 01:00','2025-01-01 01:00'],
        'n':[720,720,1,1],'power_count':[720,720,1,1],'power_sum':[720000,1440000,1000,2000]})
    p=tmp_path/'hours.csv';d.to_csv(p,index=False)
    # Consistent timestamp strings for pandas' strict parsing.
    d['datetime']=pd.to_datetime(d.datetime,format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S');d.to_csv(p,index=False)
    out=load_hourly(p)
    assert len(out)==2 and out.energy_kWh.sum()==pytest.approx(3.)
    d.loc[2,['n','power_count']]=720;d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Missing equipment'):load_hourly(p)

def test_reference_does_not_accept_duplicates_or_wrong_units():
    times=pd.date_range('2025-05-10','2025-05-31',freq='h')
    d=pd.DataFrame({'id':times,'hourly_pow':39000.,'may_bill':39000.*180,'may_carbon':39000.*.424})
    assert len(validate_reference(d))==505
    bad=d.copy();bad.loc[1,'id']=bad.loc[0,'id']
    with pytest.raises(ValueError):validate_reference(bad)
    bad=d.copy();bad['may_bill']/=1000
    with pytest.raises(ValueError,match='coefficient'):validate_reference(bad)

def test_missing_whole_hour_cannot_disappear_silently(tmp_path):
    d=grid_for_dates('2025-01-01','2025-01-01 02:00',['A','B'])
    d['n']=720;d['power_count']=720;d['power_sum']=720000.
    p=tmp_path/'hours.csv'
    d.loc[d.datetime.dt.hour==1,'n']=1;d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='interior'):load_hourly(p)
    d=d[d.datetime.dt.hour!=1];d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='entire factory'):load_hourly(p)

def test_selection_gate_cannot_promote_march_loser():
    baseline={'kind':'mean','name':'mean'};ml={'kind':'xgboost','name':'xgb_1'}
    assert select_final_method(baseline,baseline,True)==baseline
    assert select_final_method(ml,baseline,True)==ml
    assert select_final_method(ml,baseline,False)==baseline

def test_export_keeps_reference_and_physical_conventions_separate(tmp_path):
    d=grid_for_dates('2025-01-01','2025-01-01',['A']);d['energy_kWh']=1.
    m=fit_model(d,'mean')
    physical,native=export_prediction(m,'2025-05-10','2025-05-10',tmp_path)
    assert physical.energy_kWh.iloc[0]==1.
    assert physical.cost_KRW_estimate.iloc[0]==180.
    assert physical.carbon_kgCO2_estimate.iloc[0]==pytest.approx(.424)
    assert native.hourly_pow.iloc[0]==1000.
    assert native.may_bill.iloc[0]==180000.
    assert native.may_carbon.iloc[0]==pytest.approx(424.)
