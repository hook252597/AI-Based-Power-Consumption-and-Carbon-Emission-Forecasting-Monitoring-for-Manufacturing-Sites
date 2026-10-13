"""Predict arbitrary future calendar hours with the saved model."""
from pathlib import Path
import argparse,sys
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from manufacturing_ai.forecast import grid_for_dates
import pandas as pd
parser=argparse.ArgumentParser()
parser.add_argument('--start',required=True)
parser.add_argument('--end',required=True)
parser.add_argument('--model',default=str(ROOT/'models/forecast/selected_model.joblib'))
parser.add_argument('--output',default=str(ROOT/'outputs/forecast/custom_prediction.csv'))
args=parser.parse_args()
model=joblib.load(args.model)
if pd.Timestamp(args.start)<=pd.Timestamp(model.last_training_time):raise ValueError('Start must be after training')
if pd.Timestamp(args.start)>pd.Timestamp(args.end):raise ValueError('End must not precede start')
out=Path(args.output).resolve()
if out==Path(args.model).resolve() or (ROOT/'models').resolve() in out.parents:raise ValueError('Prediction output cannot overwrite saved models')
if ROOT not in out.parents:raise ValueError('Output must remain under this ml directory')
d=grid_for_dates(args.start,args.end,model.modules)
d['energy_kWh']=model.predict(d)
d['cost_KRW_estimate']=d.energy_kWh*180
d['carbon_kgCO2_estimate']=d.energy_kWh*.424
out.parent.mkdir(parents=True,exist_ok=True)
d.to_csv(out,index=False,encoding='utf-8-sig')
print(f'Saved {len(d)} equipment-hour forecasts to {out}')
