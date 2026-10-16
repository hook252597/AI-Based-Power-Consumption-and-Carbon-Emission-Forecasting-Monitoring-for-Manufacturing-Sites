"""Predeclared contextual anomaly experiment; never updates production files."""
from pathlib import Path
import json, hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL={
 'version':1,'seed':20261016,'purpose':'Detect deviations from equipment history, not reproduce voltage/PF labels.',
 'source':'localtime from rtu_data_full.csv; original files are read-only',
 'resolution':'Complete 1-minute aggregates of 12 five-second samples; score available at minute end.',
 'train':['2024-12-01','2025-03-01'],'calibration':['2025-03-01','2025-03-16'],
 'march_check':['2025-03-16','2025-04-01'],'april_development':['2025-04-01','2025-04-30'],
 'labels_used':False,'may_data_used':False,
 'input':'Current mean/std, phases and observed sensor relationships, previous-minute delta, prior-five-minute mean difference/std. Reset history at gaps/day changes.',
 'methods':['fixed_rule','static_if','robust_context','context_if'],
 'forest':{'n_estimators':100,'max_samples':2048,'n_jobs':2,'train_sample_max_per_equipment':40000},
 'scaling':'Equipment median and IQR/1.349 from training only; scale floor max(abs(median)*0.001, 1e-6).',
 'persistence_minutes':3,'alert_decision':'Third consecutive flagged complete minute end. Never backdate the notification.',
 'primary_budget_per_equipment_day':1.0,'sensitivity_budgets':[0.5,2.0],
 'threshold_selection':'March1-15 only. Empirical quantile grid 0.5..0.99999; minimize absolute distance to target events/equipment-observed-day, tie higher threshold. No injected or April labels used.',
 'fixed_rule':'Any phase voltage min<210V, line voltage min<360V, or PF min<85%. Report raw burden and same three-minute persistence reference.',
 'comparison':'Compare March check and April observed alert burdens; matched calibration budget is not a promise of matched test budgets.',
 'synthetic_probe':'Five predefined patterns in same 120-minute baseline/injected windows. Minute-aggregate perturbation, not reconstructed raw sensor dynamics. Separate baseline pre-existing alerts from newly detected injections.',
 'synthetic_patterns':['level_shift','volatility','phase_imbalance','relationship_break','gradual_drift'],
 'synthetic_target_per_kind':13,'probe_seed':20261016,
 'success_interpretation':'Do not call a model better based on alert suppression. Report per-kind incremental detections/delay, real alert burden, and examples. Even improved synthetic results do not prove real fault precision or recall.',
 'deployment_gate':'No dashboard/model replacement or Git upload. April already inspected; future actual data and human-reviewed events are needed for real usefulness.',
 'references':['https://scikit-learn.org/stable/modules/outlier_detection.html','https://scikit-learn.org/stable/common_pitfalls.html#data-leakage']
}
def freeze():
 p=ROOT/'outputs/protocol.json'
 canonical=json.dumps(PROTOCOL,sort_keys=True,ensure_ascii=False).encode()
 result={'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(canonical).hexdigest(),'protocol':PROTOCOL}
 if p.exists():
  old=json.loads(p.read_text(encoding='utf-8'))
  if old['sha256']!=result['sha256']: raise ValueError('Protocol changed; use a new experiment directory.')
  return old
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 return result
if __name__=='__main__': print(freeze()['sha256'])
