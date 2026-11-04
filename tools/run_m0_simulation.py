#!/usr/bin/env python3
"""Continuous S0 simulation for frozen PV-only H96 checkpoints. No training."""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime
import csv
import gzip
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'model_variants'), str(ROOT / 'external/ol-ems-hkust')]
import numpy as np
import pandas as pd
import torch
import cvxpy as cp
from configurations.hkust_15min import DEFAULT_CONFIG as BASE_HARDWARE, SplitWindow
from _pv_only_dataset import DEFAULT_CONFIG as PV_CONFIG, PVOnlyDataset
from _m0_variant_runner import _build_controller
from gru_2layer_m0 import PVGRUController
from lstm_2layer_m0 import PVLSTMController
from src.models.controller import OptimisationLayer
from src.training_and_evaluation.m0_training import load_checkpoint, sha256_file
from main import _build_adapter
from _hardware_profile import load_hardware_profile

# Runtime dispatch policy for these frozen-PV simulation runners. Keep the
# historical training configuration/checkpoint contract unchanged.
HARDWARE = load_hardware_profile(
    replace(BASE_HARDWARE, allow_grid_charging=True),
    os.environ.get("MPC_HARDWARE_PROFILE"),
)

DEFAULT_CHECKPOINT = ROOT / 'experiments/pv_multihorizon96_20260921_182457_137681/gru_L96_H96_seed42/best.pt'
FUTURE = ('load', 'import_price', 'export_price', 'grid_available', 'critical_load')


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')


def resolve_frozen_pv_path(saved_path):
    """Relocate a frozen dataset with the project; callers still verify both hashes."""
    original = Path(saved_path)
    if original.is_file():
        return original
    parts = original.parts
    for index in range(len(parts) - 1):
        if parts[index:index + 2] == ('output', 'frozen'):
            candidate = ROOT.joinpath(*parts[index:]).resolve()
            if candidate.is_relative_to((ROOT / 'output/frozen').resolve()) and candidate.is_file():
                return candidate
    raise FileNotFoundError(f'Frozen PV dataset not found: {original}')


def load_model(path, *, evaluation_start=None, evaluation_steps=None):
    checkpoint = load_checkpoint(path)
    if 'annual_fold' in checkpoint:
        fold = checkpoint['annual_fold']
        if evaluation_start is None or evaluation_steps is None or evaluation_steps < 1:
            raise ValueError('Monthly checkpoints require an explicit month-scoped evaluation interval.')
        start = pd.Timestamp(evaluation_start)
        end = start + pd.Timedelta(minutes=15 * evaluation_steps)
        if (start < pd.Timestamp(fold['test_origin_start'])
                or end > pd.Timestamp(fold['test_origin_end_exclusive'])):
            raise ValueError('Monthly checkpoint cannot control times outside its issuing month.')
    cfg, arch = checkpoint['config'], checkpoint['model_architecture']
    if cfg['prediction_horizon'] != 96:
        raise ValueError('This simulation requires a 96-step forecast checkpoint.')
    config = replace(PV_CONFIG, data_path=resolve_frozen_pv_path(cfg['data_path']),
                     lookback_steps=cfg['lookback_steps'], prediction_horizon=96,
                     anchor_lookback_steps=cfg.get('anchor_lookback_steps', 96),
                     split_windows={key: SplitWindow(**value) for key, value in cfg['split_windows'].items()},
                     pv_input_provenance=cfg.get('pv_input_provenance', PV_CONFIG.pv_input_provenance))
    identity = checkpoint['dataset_identity']
    if sha256_file(config.data_path) != identity['pv_timeseries_sha256']:
        raise ValueError('Frozen PV data hash differs from checkpoint.')
    manifest = config.data_path.parent / 'manifest.json'
    if sha256_file(manifest) != identity['manifest_sha256']:
        raise ValueError('Frozen manifest hash differs from checkpoint.')
    dataset = PVOnlyDataset(config)
    args = argparse.Namespace(load_model=path, mode='evaluate',
                              recurrent_type=arch['recurrent_type'], hidden_dim=arch['hidden_dim'],
                              num_layers=arch['num_layers'], dropout=arch['dropout'],
                              fc_hidden_size=arch['fc_hidden_size'])
    cls = {'GRU': PVGRUController, 'LSTM': PVLSTMController}[arch['recurrent_type']]
    model = _build_controller(config, dataset, args, cls).eval()
    return model, dataset, checkpoint


def prepare_inputs(dataset, start, steps, invalid_policy, dispatch_path, tiers_path, simulation_inputs=None):
    """Timestamp join; no skipped time, no future fill, no edits to frozen data."""
    start = pd.Timestamp(start)
    if start.tz is not None or start != start.floor('15min'):
        raise ValueError('Start must be local Asia/Shanghai time on a 15-minute boundary.')
    index = pd.date_range(start, periods=steps + 95, freq='15min')
    test = dataset.config.split_windows['test']
    planning_end = pd.Timestamp(test.end)
    manifest_path = dataset.config.data_path.parent / 'manifest.json'
    if manifest_path.is_file():
        declared_tail = json.loads(manifest_path.read_text()).get('planning_tail_end_exclusive')
        if declared_tail:
            planning_end = pd.Timestamp(declared_tail)
            if planning_end < pd.Timestamp(test.end):
                raise ValueError('Planning tail cannot end before the test interval.')
    if (index[0] < pd.Timestamp(test.start) or index[steps - 1] >= pd.Timestamp(test.end)
            or index[-1] >= planning_end):
        raise ValueError('Execution must stay in the test split; final planning window must stay in the test split or its declared planning tail.')
    pv = dataset.source_frame.set_index('datetime')
    if simulation_inputs is not None:
        from prepare_annual_simulation_inputs import verify, read
        verify(simulation_inputs, dataset.config.data_path.parents[2])
        pv = read(Path(simulation_inputs)/'pv.csv').rename(columns={'pv_kW':'pv'})
        # Local execution view only. Original reference validity is carried to
        # output; dataset and supervised-learning masks remain untouched.
        pv['history_valid'] = True
        pv['pv_target_valid'] = True
    needed = pd.date_range(start - pd.Timedelta(minutes=15 * dataset.lookback_window),
                           index[-1], freq='15min')
    section = pv.reindex(needed).copy()
    if section[['history_valid', 'pv_target_valid']].isna().any().any():
        raise ValueError('Missing PV timestamps.')
    bad = (~section.history_valid) | (~section.pv_target_valid) | (~np.isfinite(section.pv))
    repairs = section.loc[bad, ['pv']].copy()
    if bad.any():
        if invalid_policy == 'error':
            raise ValueError(f'{bad.sum()} invalid PV points in required interval: '
                             f'{list(section.index[bad].astype(str))}. '
                             'Use --invalid-pv hold-last for explicitly logged causal replacements.')
        section.loc[bad, 'pv'] = np.nan
        section['pv'] = section.pv.ffill()
        if section.pv.isna().any():
            raise ValueError('No preceding valid PV in the history window; cannot fill causally.')
    repairs['replacement_pv_kw'] = section.loc[bad, 'pv']
    if simulation_inputs is not None:
        repaired = section.pv_imputed.eq(1)
        repairs = section.loc[repaired,['original_pv_kw','pv']].rename(
            columns={'original_pv_kw':'pv','pv':'replacement_pv_kw'})
    repairs.index.name = 'timestamp'
    dispatch = pd.read_csv(dispatch_path, parse_dates=['datetime']).set_index('datetime')
    tiers = pd.read_csv(tiers_path, parse_dates=['datetime']).set_index('datetime')
    for frame in (dispatch, tiers):
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError('Dispatch/tier timestamps must be unique and ordered.')
    frame = dispatch.reindex(index)[['load_kW', 'import_price', 'export_price', 'grid_available', 'load_scenario_valid']].rename(columns={'load_kW': 'load'})
    aligned_tiers = tiers.reindex(index)
    if not np.allclose(frame.load, aligned_tiers.load_kW, rtol=0, atol=1e-6):
        raise ValueError('Tier and dispatch loads differ at matched timestamps.')
    frame['critical_load'] = aligned_tiers[list(HARDWARE.critical_load_columns)].sum(axis=1, min_count=2)
    frame['pv'] = section.pv.reindex(index)
    if not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy() < 0).any():
        raise ValueError('Invalid/missing dispatch inputs.')
    if not (frame.load_scenario_valid == 1).all():
        raise ValueError('Invalid load scenario rows.')
    if not (frame.grid_available == 1).all():
        raise ValueError('This runner supports S0 (normal grid) only; outage execution needs separate validation.')
    if (frame.critical_load > frame.load + 1e-6).any():
        raise ValueError('Critical load exceeds total load.')
    if simulation_inputs is not None:
        frame['pv_reference_valid'] = section.pv_reference_valid.reindex(index)
        frame['pv_imputed'] = section.pv_imputed.reindex(index)
        frame['load_imputed'] = dispatch.load_imputed.reindex(index)
    raw = torch.tensor(section.pv.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    history = (raw - dataset.pv_min) / max(dataset.pv_max - dataset.pv_min, 1e-12)
    return frame, history, repairs


def make_mpc():
    c = HARDWARE
    return OptimisationLayer(96, charge_min_max=c.soc_min_max_kWh,
        storage_power_min_max=c.storage_power_min_max_kW, grid_power_min_max=c.grid_power_min_max_kW,
        sample_time=c.dt_hours, charge_efficiency=c.charge_efficiency,
        discharge_efficiency=c.discharge_efficiency, self_discharge=c.self_discharge_per_step,
        allow_grid_charging=c.allow_grid_charging,
        no_arbitrage_margin=c.no_arbitrage_margin_CNY_per_kWh, objective_weights=c.objective_weights)


def solve_mpc(mpc, forecast, future, initial_soc):
    """Solve the existing MPC problem directly: M0 evaluation needs no gradients.

    Same objective/constraints and tariff cap as OptimisationLayer.forward.
    ECOS avoids the differentiable solver's extra work and loose default tolerance.
    """
    values = {key: value.detach().cpu().numpy().reshape(-1) for key, value in future.items()}
    values['pv'] = forecast.detach().cpu().numpy().reshape(-1)
    values['initial_soc'] = np.array([initial_soc])
    values['export_price'] = np.minimum(values['export_price'],
        np.maximum(values['import_price'] - mpc.no_arbitrage_margin, 0))
    for parameter in mpc._problem.parameters():
        parameter.value = values[parameter.name()]
    # ECOS occasionally returns optimal_inaccurate for an otherwise feasible LP.
    # Re-solve the identical problem with HiGHS rather than accepting that status
    # or skipping the interval. No battery state advances until a solution passes.
    attempts = []
    for solver, options in (
        ('ECOS', dict(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, feastol=1e-8,
                      max_iters=1000, warm_start=True)),
        ('SCIPY/HiGHS', dict(solver=cp.SCIPY, scipy_options={'method': 'highs'})),
    ):
        try:
            mpc._problem.solve(**options)
            if mpc._problem.status != cp.OPTIMAL:
                raise RuntimeError(f'status={mpc._problem.status}')
            if not all(v.value is not None and np.isfinite(v.value).all()
                       for v in mpc._problem.variables()):
                raise RuntimeError('nonfinite solution')
            violation = max(float(np.max(c.violation())) for c in mpc._problem.constraints)
            if not np.isfinite(violation) or violation > 1e-4:
                raise RuntimeError(f'constraint violation={violation}')
        except (cp.error.SolverError, RuntimeError) as exc:
            attempts.append(f'{solver}: {exc}')
            logging.warning('MPC solver attempt failed: %s', attempts[-1])
            continue
        mpc.last_solver = solver
        mpc.last_solver_attempts = len(attempts) + 1
        break
    else:
        raise RuntimeError('MPC failed all solvers: ' + '; '.join(attempts))
    arrays = {v.name(): np.asarray(v.value).reshape(-1).copy() for v in mpc._problem.variables()}
    for key in arrays:
        arrays[key] = np.maximum(arrays[key], 0.)
    net = arrays['grid_import'] - arrays['grid_export']
    arrays['grid_import'], arrays['grid_export'] = np.maximum(net, 0), np.maximum(-net, 0)
    return arrays, violation


def spark(values, width):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return ''
    selected = values[np.linspace(0, len(values)-1, min(width, len(values))).astype(int)]
    low, high = float(values.min()), float(values.max())
    levels = np.clip(np.rint((selected-low)/max(high-low, 1e-12)*7), 0, 7).astype(int)
    return ''.join('▁▂▃▄▅▆▇█'[i] for i in levels)


class TerminalView:
    """One latest snapshot; timer rendering never queues frames or blocks solves."""
    def __init__(self, enabled=True):
        self.enabled = enabled and sys.stdout.isatty()
        self.snapshot = None
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = None
    def __enter__(self):
        if self.enabled:
            sys.stdout.write('\033[?1049h\033[?25l'); sys.stdout.flush()
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()
        return self
    def publish(self, snapshot):
        self.snapshot = snapshot
        self.ready.set()
    def loop(self):
        while not self.stop.is_set() and not self.ready.wait(.1):
            pass
        while not self.stop.is_set():
            self.render()
            if self.stop.wait(1):
                break
    def render(self):
        s = self.snapshot
        if s is None:
            return
        size = shutil.get_terminal_size((100, 28))
        width = max(8, min(96, size.columns - 4))
        lines = [f"M0 / {s['method']} | {s.get('scenario', 'S0')} | {s['step']}/{s['total']} | {s['timestamp']}",
                 f"Elapsed {s['elapsed']:.1f}s | ETA {s['eta']/60:.1f} min | last loop {s['seconds']:.2f}s",
                 f"Actual SOC {s['soc']:.1f}% | cumulative purchase {s['purchase']:.2f} kWh | grid bill {s['bill']:.2f} CNY",
                 f"Latest plan: next 24h purchase {s['plan_purchase']:.2f} kWh (not actual)",
                 'NEXT 24h: left=now, right=+24h; each chart has its own scale']
        if s.get('event_status'):
            lines.insert(4, s['event_status'])
        for label, values, unit in s['charts']:
            lines.extend([f'{label} [{min(values):.2f}, {max(values):.2f}] {unit}', spark(values, width)])
        lines += [f"RECENT actual SOC: last {len(s['history'])} steps [{min(s['history']):.1f}, {max(s['history']):.1f}] %",
                  spark(s['history'], width),
                  'Refresh <= 1 Hz; intermediate frames skipped; Ctrl-C saves partial results.']
        # ASCII labels make display width predictable; never scroll beyond the screen.
        rendered = '\n'.join(line[:max(1,size.columns-1)] for line in lines[:max(1,size.lines-1)])
        sys.stdout.write('\033[H' + rendered + '\033[J'); sys.stdout.flush()
    def __exit__(self, *exc):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.enabled:
            sys.stdout.write('\033[?25h\033[?1049l'); sys.stdout.flush()


def verify_execution(row):
    residual = row['pv_used_kwh'] + row['battery_discharge_kwh'] + row['grid_import_kwh'] - row['load_served_kwh'] - row['battery_charge_kwh'] - row['grid_export_kwh'] - row['overgeneration_kwh']
    expected_soc = row['soc_before_kwh'] + HARDWARE.charge_efficiency * row['battery_charge_kwh'] - row['battery_discharge_kwh'] / HARDWARE.discharge_efficiency
    soc_error = row['soc_after_kwh'] - expected_soc
    if abs(residual) > 1e-5 or abs(soc_error) > 1e-5:
        raise RuntimeError(f'Physical energy/SOC check failed: {residual}, {soc_error}')
    if not HARDWARE.min_soc_kWh-1e-5 <= row['soc_after_kwh'] <= HARDWARE.max_soc_kWh+1e-5:
        raise RuntimeError('SOC outside battery limits')
    return residual, soc_error


def run(args):
    if args.days <= 0 or not float(args.days*96).is_integer() or args.history_steps < 1:
        raise ValueError('days must be a positive multiple of 1/96; history-steps must be positive.')
    steps = args.steps if args.steps is not None else int(args.days * 96)
    if steps < 1:
        raise ValueError('steps must be positive')
    torch.set_num_threads(1)
    model, dataset, checkpoint = load_model(args.checkpoint, evaluation_start=args.start, evaluation_steps=steps)
    frame, histories, repairs = prepare_inputs(dataset, args.start, steps, args.invalid_pv,
                                               args.dispatch, args.load_tiers)
    print(f'Preflight OK: {steps} steps, {frame.index[0]} to {frame.index[steps-1] + pd.Timedelta(minutes=15)}; '
          f'{len(repairs)} PV replacements. CPU / {checkpoint["model_architecture"]["recurrent_type"]}.', flush=True)
    if args.check_only:
        return
    output = args.output or ROOT / 'experiments' / ('m0_simulation_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=output/'run.log', level=logging.INFO, force=True)
    logging.captureWarnings(True)
    print(f'Results: {output}', flush=True)
    repairs.to_csv(output/'pv_replacements.csv')
    metadata = dict(arguments=vars(args), checkpoint_sha256=sha256_file(args.checkpoint),
        model_architecture=checkpoint['model_architecture'], hardware=asdict(HARDWARE),
        dispatch_sha256=sha256_file(args.dispatch), tiers_sha256=sha256_file(args.load_tiers),
        pv_identity=checkpoint['dataset_identity'], requested_steps=steps,
        information='S0 only; future scenario load/tariffs are assumed known; PV predicted from past only. Perfect mode is an oracle benchmark.',
        missing_policy=args.invalid_pv, replacement_count=len(repairs),
        provenance='Constructed PV and synthetic load, not measured floating-building operation.',
        execution='Actual PV and current SOC drive python-microgrid; no daily reset. Export tariff capped below import only in MPC, original tariff used for actual bill.')
    metadata['solver'] = 'ECOS with SCIPY/HiGHS fallback; identical MPC problem; optimal status and constraint checks required'
    metadata['runner_sha256'] = sha256_file(Path(__file__))
    metadata['python'] = sys.version
    metadata['versions'] = dict(torch=torch.__version__, cvxpy=cp.__version__, numpy=np.__version__)
    write_json(output/'run_metadata.json', metadata)
    totals = dict(grid_import_kwh=0., grid_export_kwh=0., grid_cost=0., component_cost_sum=0.,
                  pv_curtailed_kwh=0., battery_charge_kwh=0., battery_discharge_kwh=0., loss_load_kwh=0.)
    completed, status, error = 0, 'running', None
    recent = deque(maxlen=args.history_steps)
    began = time.perf_counter()
    adapter = None
    interrupted = threading.Event()
    old_handler = signal.signal(signal.SIGINT, lambda *_: interrupted.set())
    try:
        mpc, adapter = make_mpc(), _build_adapter(HARDWARE)
        future_tensors = {key: torch.tensor(frame[key].to_numpy(), dtype=torch.float64) for key in FUTURE}
        actual_writer = plan_writer = None
        with (output/'actual.csv').open('w', newline='') as af, gzip.open(output/'plans.csv.gz', 'wt', newline='') as pf, TerminalView(not args.no_live) as view:
            for i in range(steps):
                if interrupted.is_set():
                    break
                tick = time.perf_counter()
                with torch.no_grad():
                    if args.method == 'model':
                        forecast = model.forecast_pv(histories[i:i+dataset.lookback_window].unsqueeze(0))
                    elif args.method == 'persistence':
                        raw = histories[i+dataset.lookback_window-96:i+dataset.lookback_window] * (dataset.pv_max-dataset.pv_min) + dataset.pv_min
                        forecast = raw.unsqueeze(0)
                    else:
                        forecast = torch.tensor(frame.pv.iloc[i:i+96].to_numpy(), dtype=torch.float32).reshape(1,96,1)
                    prediction_seconds = time.perf_counter()-tick
                    arrays, violation = solve_mpc(mpc, forecast,
                        {key: values[i:i+96] for key, values in future_tensors.items()}, adapter.get_soc_kwh())
                solved = time.perf_counter()
                if not all(np.isfinite(v).all() for v in arrays.values()):
                    raise RuntimeError('Nonfinite MPC solution')
                plan = {key: max(0., float(arrays[source][0])) for key,source in
                        dict(battery_charge='charge',battery_discharge='discharge',grid_import='grid_import',grid_export='grid_export').items()}
                row = adapter.step(plan, {key: float(frame.iloc[i][key]) for key in ('pv',)+FUTURE})
                balance, soc_error = verify_execution(row)
                row.update(timestamp=str(frame.index[i]), decision_step=i, forecast_pv_kw=float(forecast[0,0,0]),
                           actual_pv_kw=float(frame.pv.iloc[i]), prediction_seconds=prediction_seconds,
                           mpc_max_constraint_violation=violation,
                           mpc_solver=mpc.last_solver, mpc_solver_attempts=mpc.last_solver_attempts,
                           mpc_seconds=solved-tick-prediction_seconds, balance_residual_kwh=balance, soc_error_kwh=soc_error,
                           step_seconds=time.perf_counter()-tick)
                if actual_writer is None:
                    actual_writer = csv.DictWriter(af, fieldnames=list(row)); actual_writer.writeheader()
                actual_writer.writerow(row); af.flush()
                plan_frame = pd.DataFrame({k:v[:96] for k,v in arrays.items()})
                plan_frame['soc_end'] = arrays['soc'][1:97]
                plan_frame['pv_forecast_kw'] = forecast.numpy().reshape(-1)
                plan_frame['horizon_timestamp'] = frame.index[i:i+96].astype(str)
                plan_frame['decision_timestamp'] = str(frame.index[i])
                plan_frame['decision_step'] = i
                for key in FUTURE:
                    plan_frame[key] = frame[key].iloc[i:i+96].to_numpy()
                if plan_writer is None:
                    plan_writer = csv.writer(pf); plan_writer.writerow(plan_frame.columns)
                plan_writer.writerows(plan_frame.itertuples(index=False, name=None)); pf.flush()
                completed = i+1
                for key in totals:
                    totals[key] += row[key]
                recent.append(100*row['soc_after_fraction'])
                elapsed = time.perf_counter()-began
                view.publish(dict(method=args.method,step=completed,total=steps,timestamp=str(frame.index[i]),
                    elapsed=elapsed,eta=elapsed/completed*(steps-completed),seconds=time.perf_counter()-tick,
                    soc=recent[-1],purchase=totals['grid_import_kwh'],bill=totals['grid_cost'],
                    plan_purchase=float(np.maximum(arrays['grid_import'],0).sum()*.25),history=list(recent),
                    charts=[('Charge',arrays['charge'].tolist(),'kW'),('Discharge',arrays['discharge'].tolist(),'kW'),
                            ('SOC', (arrays['soc']/HARDWARE.battery_capacity_kWh*100).tolist(),'%'),
                            ('Purchase', (np.maximum(arrays['grid_import'],0)*.25).tolist(),'kWh / 15min')]))
                if not view.enabled and (completed == 1 or completed % 96 == 0):
                    print(f'{completed}/{steps} | {frame.index[i]} | {elapsed:.1f}s | SOC {recent[-1]:.1f}%', flush=True)
                if completed % 96 == 0:
                    write_json(output/'status.json', dict(status='running',completed_steps=completed,elapsed_seconds=elapsed,**totals))
            status = 'completed' if completed == steps else 'interrupted'
    except KeyboardInterrupt:
        status = 'interrupted'
    except Exception:
        status, error = 'failed', traceback.format_exc()
        logging.exception('Simulation failed')
    finally:
        signal.signal(signal.SIGINT, old_handler)
        summary = dict(status=status, completed_steps=completed, requested_steps=steps,
                       elapsed_seconds=time.perf_counter()-began, **totals,
                       final_soc_kwh=adapter.get_soc_kwh() if adapter else None, error=error)
        write_json(output/'summary.json', summary)
        write_json(output/'status.json', summary)
        print(f'{status}: {completed}/{steps} steps; results: {output}', flush=True)
    if error:
        raise RuntimeError(error)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--start', default='2023-11-01 00:00:00')
    p.add_argument('--days', type=float, default=30)
    p.add_argument('--steps', type=int, help='Short validation override; otherwise days * 96')
    p.add_argument('--invalid-pv', choices=['error','hold-last'], default='error')
    p.add_argument('--method', choices=['model','persistence','perfect'], default='model')
    p.add_argument('--history-steps', type=int, default=96)
    p.add_argument('--dispatch', type=Path, default=ROOT/'output/frozen/hkust_m0_v1_20260909/dispatch.csv')
    p.add_argument('--load-tiers', type=Path, default=ROOT/'output/frozen/hkust_m0_v1_20260909/load_tiers.csv')
    p.add_argument('--output', type=Path)
    p.add_argument('--no-live', action='store_true')
    p.add_argument('--check-only', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
