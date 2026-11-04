"""Compare completed, time-aligned initial-SOC trajectories; no model execution."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def analyze(suite, output):
    manifest = json.loads((suite / 'manifest.json').read_text())
    state = json.loads((suite / 'suite_status.json').read_text())
    capacity = manifest['identity']['hardware']['battery_capacity_kWh']
    trajectories, metrics = {}, []
    for name in ('S0_initial_soc20', 'S0_initial_soc50', 'S0_initial_soc80'):
        record = state['cases'][name]
        summary = json.loads((Path(record['path']) / 'summary.json').read_text())
        if summary['status'] != 'completed':
            raise ValueError(f'{name} is not a completed trajectory')
        frame = pd.read_csv(Path(record['path']) / 'actual.csv', parse_dates=['timestamp'])
        if len(frame) != summary['requested_steps']:
            raise ValueError('Incomplete actual ledger')
        if not np.allclose(frame.soc_before_kwh.iloc[1:], frame.soc_after_kwh.iloc[:-1], atol=1e-6):
            raise ValueError('Discontinuous SOC inside a trajectory')
        trajectories[name] = frame.set_index('timestamp')
        metrics.append(dict(case=name, initial_SOC_percent=100*summary['initial_soc_kwh']/capacity,
            final_SOC_percent=100*summary['final_soc_kwh']/capacity,
            grid_cost_CNY=summary['grid_cost'], critical_ENS_kWh=summary['cl1_ens_kwh']+summary['cl2_ens_kwh'],
            elapsed_seconds=summary['elapsed_seconds'], completed_steps=summary['completed_steps']))
    reference = next(iter(trajectories.values())).index
    if not all(f.index.equals(reference) for f in trajectories.values()):
        raise ValueError('Trajectories are not time aligned')
    values = pd.DataFrame({n: f.soc_after_kwh for n, f in trajectories.items()})
    values['spread_kWh'] = values.max(axis=1)-values.min(axis=1)
    values['spread_percentage_points'] = 100*values.spread_kWh/capacity
    convergence = []
    for tolerance in (1., .1, .01):
        good = values.spread_percentage_points.to_numpy() <= tolerance
        sustained = np.logical_and.accumulate(good[::-1])[::-1]
        indices = np.flatnonzero(sustained)
        i = int(indices[0]) if len(indices) else None
        convergence.append(dict(tolerance_percentage_points=tolerance,
            first_sustained_through_end_timestamp=str(reference[i]+pd.Timedelta(minutes=15)) if i is not None else None,
            elapsed_hours=(i+1)*.25 if i is not None else None))
    output.mkdir(parents=True, exist_ok=True)
    values.to_csv(output / 'soc_comparison.csv')
    pd.DataFrame(metrics).to_csv(output / 'trajectory_metrics.csv', index=False)
    result = dict(suite=str(suite.resolve()), battery_kWh=capacity, steps_per_case=len(reference),
        final_spread_kWh=float(values.spread_kWh.iloc[-1]), convergence=convergence,
        note='Convergence applies only to this checked grid-connected interval and fixed controller; no annual/outage guarantee.')
    (output/'convergence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('suite',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.suite,a.output)
