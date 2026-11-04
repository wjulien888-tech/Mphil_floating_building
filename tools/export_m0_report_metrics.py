#!/usr/bin/env python3
"""Export the first M0 report's metric contract from saved scenario results only.

No model, solver, or simulation imports. The source suite is never modified.
Counts/dates/hardware come from its manifest, not the November 2023 report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DT, H = 0.25, 96
SCHEMA = 'm0-report-metrics-v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(frame):
    valid = frame.loc[frame.actual_count.eq(H)]
    error = valid.forecast_kwh - valid.actual_kwh
    variance = ((valid.actual_kwh - valid.actual_kwh.mean()) ** 2).sum()
    return dict(complete_windows=len(valid), unit='kWh',
                mae=float(error.abs().mean()) if len(valid) else None,
                rmse=float(np.sqrt((error ** 2).mean())) if len(valid) else None,
                mbe=float(error.mean()) if len(valid) else None,
                r2=float(1 - (error ** 2).sum() / variance) if variance > 0 else None,
                overprediction_pct=float(100 * error.gt(0).mean()) if len(valid) else None)


def export_report_metrics(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / 'report_data'
    manifest = json.loads((run / 'manifest.json').read_text())
    state = json.loads((run / 'suite_status.json').read_text())
    s = pd.read_csv(run / 'summary.csv')
    identity = manifest['identity']
    cases = {c['name']: c for c in manifest['all_cases']}
    steps = int(identity['config']['days'] * 96)
    if (not state.get('full_matrix_completed') or s.case.duplicated().any()
            or set(s.case) != set(cases) or not s.status.eq('completed').all()
            or not s.completed_steps.eq(steps).all() or not s.requested_steps.eq(steps).all()):
        raise ValueError('Report metrics require a complete, non-smoke scenario matrix')
    start = pd.Timestamp(identity['config']['start'])
    clock = pd.date_range(start, periods=steps, freq='15min')
    hardware = identity['hardware']
    capacity = hardware['battery_capacity_kWh']
    floor = capacity * hardware['min_soc_fraction']
    eta_d = np.sqrt(hardware['roundtrip_efficiency'])
    for col in ['event_hour', 'outage_hours', 'initial_soc', 'rho', 'pv_regime']:
        s[col] = s.case.map(lambda name: cases[name][col])
    s['critical_ens_kwh'] = s.outage_cl1_ens_kwh + s.outage_cl2_ens_kwh
    s['critical_load_kwh'] = s.outage_cl1_load_kwh + s.outage_cl2_load_kwh
    s['critical_supply_pct'] = 100 * (1 - s.critical_ens_kwh / s.critical_load_kwh.where(s.critical_load_kwh > 0))
    s['outage_soc_pct'] = 100 * s.outage_start_soc_kwh / capacity
    # Keep the legacy event_timestamp even for a false warning / grid control.
    s['event_timestamp'] = start + pd.to_timedelta(s.event_hour, unit='h')
    s['event_end_exclusive'] = s.event_timestamp + pd.to_timedelta(s.outage_hours, unit='h')
    s['month'] = start.strftime('%Y-%m')
    s['checkpoint_sha256'] = identity['checkpoint_sha256']
    if manifest.get('monthly_models'):
        s['initial_checkpoint_sha256'] = identity['checkpoint_sha256']
        s['checkpoint_sha256'] = None  # No single checkpoint controlled the year.
        s['monthly_models_sha256'] = manifest['monthly_models_sha256']
        s['month'] = start.strftime('%Y-%m')+'..'+(start+pd.Timedelta(minutes=15*(steps-1))).strftime('%Y-%m')
    s['metric_schema'] = SCHEMA
    actuals = {}
    for row in s.itertuples():
        keep = {'timestamp','dt_hours','actual_pv_kw','forecast_pv_kw','grid_connected',
                'soc_before_kwh','pv_available_kwh','pv_reference_valid','pv_imputed','load_imputed',
                'cl1_scheduled_shed_kw','cl2_scheduled_shed_kw'}
        keep.update(f'cl{i}_{m}_kwh' for i in (1,2,3) for m in ('ens','load'))
        a = pd.read_csv(Path(row.result_path) / 'actual.csv', usecols=lambda c:c in keep, parse_dates=['timestamp'])
        if len(a) != steps or not pd.DatetimeIndex(a.timestamp).equals(clock) or not np.allclose(a.dt_hours, DT):
            raise ValueError(f'Incomplete/irregular actual records: {row.case}')
        actuals[row.case] = a
    # The v2 scenarios share unmodified PV and a single frozen model per suite.
    reference = 'S0_grid' if 'S0_grid' in actuals else s.case.iloc[0]
    a0 = actuals[reference]
    for name, a in actuals.items():
        if not np.array_equal(a.actual_pv_kw, a0.actual_pv_kw) or not np.array_equal(a.forecast_pv_kw, a0.forecast_pv_kw):
            raise ValueError(f'Cannot share forecast statistics: differing PV/forecast in {name}')
    p = pd.read_csv(Path(s.set_index('case').loc[reference, 'result_path']) / 'plans.csv.gz',
                    usecols=['decision_step', 'decision_timestamp', 'horizon_timestamp', 'pv_forecast_kw'])
    p['horizon_timestamp'] = pd.to_datetime(p.horizon_timestamp)
    p['decision_timestamp'] = pd.to_datetime(p.decision_timestamp)
    g = p.groupby('decision_step', sort=True)
    if len(p) != steps * H or not g.size().eq(H).all() or list(g.groups) != list(range(steps)):
        raise ValueError('Missing or duplicate forecast windows')
    expected_targets = np.repeat(clock.to_numpy(), H) + np.tile(np.arange(H) * np.timedelta64(15, 'm'), steps)
    if (not np.array_equal(p.horizon_timestamp.to_numpy(), expected_targets)
            or not np.array_equal(p.decision_timestamp.to_numpy(), np.repeat(clock.to_numpy(), H))
            or not np.isfinite(p.pv_forecast_kw).all()):
        raise ValueError('Forecast timestamps/values do not match the 96-step contract')
    p['actual_pv_kw'] = p.horizon_timestamp.map(a0.set_index('timestamp').actual_pv_kw)
    if 'pv_reference_valid' in a0:
        valid = p.horizon_timestamp.map(a0.set_index('timestamp').pv_reference_valid).eq(1)
        p.loc[~valid,'actual_pv_kw'] = np.nan
    g = p.groupby('decision_step', sort=True)
    rolling = pd.DataFrame(dict(timestamp=g.decision_timestamp.first(),
        forecast_kwh=g.pv_forecast_kw.sum() * DT,
        actual_kwh=g.actual_pv_kw.sum(min_count=H) * DT,
        actual_count=g.actual_pv_kw.count())).reset_index()
    event_steps = sorted(set((s.loc[s.outage_hours.gt(0), 'event_hour'] / DT).astype(int)))
    events = rolling.set_index('decision_step').loc[event_steps].reset_index()
    events['ratio'] = events.forecast_kwh / events.actual_kwh.where(events.actual_kwh > 0)
    events['error_kwh'] = events.forecast_kwh - events.actual_kwh
    chains = []
    for row in s.itertuples():
        a = actuals[row.case]
        o = a.loc[a.grid_connected.eq(False)]
        expected = a.timestamp.ge(row.event_timestamp) & a.timestamp.lt(row.event_end_exclusive)
        if not np.array_equal(a.grid_connected.eq(False), expected):
            raise ValueError(f'Actual outage differs from configured event: {row.case}')
        # Audit saved aggregates before allowing a report to consume them.
        for tier in ('cl1', 'cl2', 'cl3'):
            for metric in ('ens', 'load'):
                key = f'outage_{tier}_{metric}_kwh'
                if not np.isclose(o[f'{tier}_{metric}_kwh'].sum(), getattr(row, key), atol=1e-7, rtol=0):
                    raise ValueError(f'Saved summary mismatch: {row.case} {key}')
        if o.empty:
            continue
        step = int(row.event_hour / DT)
        f = rolling.iloc[step]
        usable = max(0., a.iloc[step].soc_before_kwh - floor) * eta_d
        planned = ((o.cl1_scheduled_shed_kw + o.cl2_scheduled_shed_kw) * o.dt_hours).sum()
        chains.append(dict(case=row.case, step=step, event_timestamp=str(row.event_timestamp),
            first24_forecast_kwh=f.forecast_kwh, first24_actual_kwh=f.actual_kwh,
            start_soc_pct=row.outage_soc_pct, start_usable_battery_kwh=usable,
            critical_load_kwh=row.critical_load_kwh, actual_pv_kwh=o.pv_available_kwh.sum(),
            critical_ens_kwh=row.critical_ens_kwh, planned_critical_shed_kwh=planned,
            additional_critical_ens_kwh=row.critical_ens_kwh-planned,
            supply_pct=row.critical_supply_pct,
            energy_gap_lower_bound_kwh=max(0., row.critical_load_kwh-o.pv_available_kwh.sum()-usable)))
    chain_columns = ['case', 'step', 'event_timestamp', 'first24_forecast_kwh', 'first24_actual_kwh',
        'start_soc_pct', 'start_usable_battery_kwh', 'critical_load_kwh', 'actual_pv_kwh',
        'critical_ens_kwh', 'planned_critical_shed_kwh', 'additional_critical_ens_kwh',
        'supply_pct', 'energy_gap_lower_bound_kwh']
    evidence = pd.DataFrame(chains, columns=chain_columns)
    s = s.merge(evidence[['case', 'actual_pv_kwh', 'start_usable_battery_kwh',
        'planned_critical_shed_kwh', 'additional_critical_ens_kwh', 'energy_gap_lower_bound_kwh']], on='case', how='left')
    for flag in ('pv_imputed','load_imputed'):
        s[flag+'_points'] = s.case.map(lambda name:int(actuals[name][flag].sum()) if flag in actuals[name] else 0)
    s['simulation_uses_imputation'] = s.pv_imputed_points.gt(0) | s.load_imputed_points.gt(0)
    selections = []
    for e in json.loads((run / 'event_selection.json').read_text()):
        if cases[e['case']]['group'] != 'S4' or cases[e['case']]['outage_hours'] <= 0:
            continue
        n = int(e['hours'] / DT)
        candidates = [(float(a0.actual_pv_kw.iloc[i:i+n].sum() * DT), t)
                      for i, t in enumerate(clock)
                      if t.hour + t.minute/60 == e['start_clock']
                      and i*DT >= max(24, cases[e['case']]['warning_hours']) and i+n+H <= steps]
        best, chosen = min(candidates)
        if chosen != pd.Timestamp(e['start']) or len(candidates) != e['candidate_count'] or not np.isclose(best, e['pv_kwh'], atol=1e-8, rtol=0):
            raise ValueError(f'Low-PV selection mismatch: {e["case"]}')
        segment = a0.loc[a0.timestamp.ge(e['start']) & a0.timestamp.lt(e['end_exclusive'])]
        selections.append(dict(case=e['case'], start=e['start'], end_exclusive=e['end_exclusive'],
            hours=e['hours'], pv_kwh=best, candidate_count=len(candidates),
            candidate_median_kwh=float(np.median([v for v, _ in candidates])), rank=1,
            peak_pv_kw=float(segment.actual_pv_kw.max()), repaired_pv_points=e['repaired_pv_points']))
    pair_columns = ['case_low', 'case_high', 'start', 'hours', 'rho05_critical_ens_kwh',
                    'rho1_critical_ens_kwh', 'reduction_pct', 'initial_soc']
    pairs = []
    low = s.loc[s.group.eq('S5') & s.pv_regime.eq('observed_low')]
    for _, group in low.groupby(['event_timestamp', 'outage_hours', 'initial_soc']):
        lo, hi = group.loc[group.rho.eq(.5)], group.loc[group.rho.eq(1.)]
        if len(lo) != 1 or len(hi) != 1:
            continue
        lo, hi = lo.iloc[0], hi.iloc[0]
        pairs.append(dict(case_low=lo['case'], case_high=hi['case'], start=str(lo.event_timestamp),
            hours=lo.outage_hours, rho05_critical_ens_kwh=lo.critical_ens_kwh,
            rho1_critical_ens_kwh=hi.critical_ens_kwh,
            reduction_pct=100*(1-hi.critical_ens_kwh/lo.critical_ens_kwh) if lo.critical_ens_kwh > 0 else np.nan,
            initial_soc=lo.initial_soc))
    # Write only after all source checks pass. Preserve legacy column names and
    # numeric precision. Count-bearing aliases match old filenames when 47/10.
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    def csv(frame, name):
        frame.to_csv(output / name, index=False)
        files[name] = dict(rows=len(frame), sha256=sha256(output / name))
    csv(s, '全部案例绘图指标.csv')
    csv(s, f'全部{len(s)}案例绘图指标.csv')
    csv(rolling, '整月滚动24h预测电量.csv')
    csv(events, '全部停电起点预测.csv')
    csv(events, f'全部{len(events)}个停电起点预测.csv')
    csv(evidence, '重点案例证据链.csv')  # Now every outage case is available.
    csv(pd.DataFrame(selections, columns=['case', 'start', 'end_exclusive', 'hours', 'pv_kwh',
        'candidate_count', 'candidate_median_kwh', 'rank', 'peak_pv_kw', 'repaired_pv_points']), 'S4事件选取与候选核对.csv')
    csv(pd.DataFrame(pairs, columns=pair_columns), 'S5低光伏配对对照.csv')
    csv(s.loc[s.group.eq('S4') & s.outage_hours.gt(0)], 'S4报告表2.csv')
    provenance = dict(schema=SCHEMA, source_run=str(run),
        source_manifest_sha256=sha256(run/'manifest.json'), source_summary_sha256=sha256(run/'summary.csv'),
        exporter_sha256=sha256(__file__), checkpoint_sha256=identity['checkpoint_sha256'],
        checkpoint_scope='initial checkpoint; see monthly_models for actual issuing-month weights' if manifest.get('monthly_models') else 'single checkpoint',
        monthly_models=manifest.get('monthly_models',[]),
        month=start.strftime('%Y-%m'), start=str(start), end_exclusive=str(start+steps*pd.Timedelta(minutes=15)),
        case_count=len(s), steps_per_case=steps, invalid_pv_policy=identity['invalid_pv'],
        critical_definition='CL1 + CL2; outage interval only; zero-demand SR is undefined',
        forecast_reference_case=reference, rolling_24h_scores=scores(rolling), outage_start_24h_scores=scores(events),
        forecast_scope='Only complete 96-point original-valid truth windows within this suite; imputed PV and last 95 origins unscored. Overlapping windows are not independent.',
        imputation_scope='Imputed values advance physical simulation and affect SOC/ENS; they are not measured truth. Refer to *_imputed_points in case table.',
        evidence_scope='All outage cases; select case by name, never by row position.',
        pair_scope='Both initial SOC levels kept separately; equality is not assumed.',
        cost_scope='grid_cost is net electricity bill; component_cost_sum includes penalties, not battery lifetime cost.',
        files=files)
    (output/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = export_report_metrics(args.run, args.output)
    print(json.dumps(dict(case_count=result['case_count'], files=list(result['files'])), ensure_ascii=False))
