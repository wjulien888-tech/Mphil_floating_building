"""Causal, simulation-only repairs. Never alter frozen training/scoring data."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'output/prepared/annual_18_2_12_20261006'
OUTPUT = ROOT/'output/prepared/annual_simulation_imputed_20261006'
START, END = '2023-02-28', '2024-03-02'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    f = pd.read_csv(path, parse_dates=['datetime'], float_precision='round_trip').set_index('datetime')
    if f.index.has_duplicates or not f.index.is_monotonic_increasing:
        raise ValueError(f'Invalid time axis: {path}')
    return f


def donors(frame, valid, t, days=28, same_weekday=False):
    candidates = [t-pd.Timedelta(days=d) for d in range(1, days+1)]
    candidates = [s for s in candidates if s in frame.index and valid.loc[s]]
    if same_weekday:
        matched = [s for s in candidates if s.dayofweek == t.dayofweek]
        candidates = matched or candidates
    if not candidates:
        raise ValueError(f'No past valid same-clock donor for {t}')
    return candidates


def prepare(data=DATA, output=OUTPUT):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    sources = [data/'forecast/lookback_96/pv_timeseries.csv', data/'dispatch/dispatch.csv', data/'dispatch/load_tiers.csv']
    pv, dispatch, tiers = [read(p) for p in sources]
    axis = pd.date_range(START, END, freq='15min', inclusive='left')
    p, d, l = [f.reindex(axis).copy() for f in (pv, dispatch, tiers)]
    pv_valid = pv.history_valid.eq(1) & pv.pv_target_valid.eq(1) & np.isfinite(pv.pv_kW) & pv.pv_kW.ge(0)
    p['pv_reference_valid'] = pv_valid.reindex(axis).astype(int)
    p['pv_imputed'] = 1-p.pv_reference_valid
    p['original_pv_kw'] = p.pv_kW
    records = []
    for t in axis[p.pv_imputed.eq(1)]:
        chosen = donors(pv, pv_valid, t)[:7]
        p.loc[t, 'pv_kW'] = float(pv.loc[chosen, 'pv_kW'].median())
        records.append(dict(kind='PV', timestamp=str(t), method='median_of_up_to_7_past_valid_same_clock_days_within_28d',
                            donors=[str(s) for s in chosen], old_value=float(pv.loc[t,'pv_kW']), new_value=float(p.loc[t,'pv_kW'])))
    # A whole load/tier vector is copied from one past valid slot so that
    # components and all service-tier partitions remain internally consistent.
    load_columns = ['base_kW','plug_kW','lighting_kW','hvac_kW','aux_kW','occupancy_factor','operation_factor','load_kW']
    tier_columns = list(tiers.columns)
    valid = dispatch.load_scenario_valid.eq(1) & np.isfinite(dispatch[load_columns]).all(axis=1)
    valid &= np.isfinite(tiers.reindex(dispatch.index)[tier_columns]).all(axis=1)
    d['load_imputed'] = (~valid.reindex(axis)).astype(int)
    for t in axis[d.load_imputed.eq(1)]:
        donor = donors(dispatch, valid, t, same_weekday=True)[0]
        d.loc[t,load_columns] = dispatch.loc[donor,load_columns].to_numpy()
        l.loc[t,tier_columns] = tiers.loc[donor,tier_columns].to_numpy()
        records.append(dict(kind='load_and_tiers',timestamp=str(t),method='past_valid_same_weekday_and_clock_else_same_clock_within_28d',
                            donors=[str(donor)],old_value=float(dispatch.loc[t,'load_kW']),new_value=float(d.loc[t,'load_kW'])))
    d['load_reference_valid'] = 1-d.load_imputed
    d['load_scenario_valid'] = 1
    d['pv_kW'] = p.pv_kW
    d['pv_imputed'] = p.pv_imputed
    d['pv_reference_valid'] = p.pv_reference_valid
    # These original flags are deliberately retained, not relabelled truth.
    d['pv_history_valid'] = p.history_valid
    d['pv_target_valid'] = p.pv_target_valid
    l['load_imputed'] = d.load_imputed
    required = ['load_kW','import_price','export_price','grid_available']
    if not np.isfinite(d[required]).all().all() or not np.isfinite(p.pv_kW).all():
        raise ValueError('Unrepaired simulation inputs')
    for mode in ('emergency','minimum_operation','extended_operation'):
        if not np.allclose(l[[f'{mode}_CL{i}_kW' for i in (1,2,3)]].sum(axis=1),d.load_kW,atol=1e-6,rtol=0):
            raise ValueError('Tier balance differs from load')
    if not np.array_equal(p.loc[p.pv_imputed.eq(0),'pv_kW'],pv.loc[axis[p.pv_imputed.eq(0)],'pv_kW']):
        raise ValueError('Valid PV modified')
    output.mkdir(parents=True)
    for name,f in [('pv.csv',p),('dispatch.csv',d),('load_tiers.csv',l)]:
        f.rename_axis('datetime').to_csv(output/name)
    (output/'imputation_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    report = dict(schema='annual-simulation-imputation-v1',start=START,end_exclusive=END,
        source_hashes={str(s):digest(s) for s in sources},
        output_hashes={name:digest(output/name) for name in ['pv.csv','dispatch.csv','load_tiers.csv','imputation_records.json']},
        pv_imputed_points=int(p.pv_imputed.sum()),load_imputed_points=int(d.load_imputed.sum()),
        policy='Past original valid donors only; no future or recursive imputed donors. Simulation only; frozen training/validation/scoring unchanged.',
        monthly=p.assign(load_imputed=d.load_imputed).groupby(p.index.strftime('%Y-%m'))[['pv_imputed','load_imputed']].sum().to_dict('index'))
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def verify(output, data=DATA):
    output, data = Path(output).resolve(), Path(data).resolve()
    m = json.loads((output/'manifest.json').read_text())
    expected = {str(data/p) for p in ['forecast/lookback_96/pv_timeseries.csv','dispatch/dispatch.csv','dispatch/load_tiers.csv']}
    if set(m['source_hashes']) != expected:
        raise ValueError('Simulation overlay belongs to different source data')
    for path, value in m['source_hashes'].items():
        if digest(path) != value: raise ValueError(f'Source changed: {path}')
    for name,value in m['output_hashes'].items():
        if digest(output/name) != value: raise ValueError(f'Overlay changed: {name}')
    return m


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    print(json.dumps(prepare(args.data,args.output),ensure_ascii=False,indent=2))
