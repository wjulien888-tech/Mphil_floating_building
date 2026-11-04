#!/usr/bin/env python3
"""M2: optional prediction supervision + economics + tiered actual ENS penalty.

Independent output namespace: experiments/M2/<run>/ (isolated from existing M1 jobs).
No test observations enter training, normalisation, validation or selection.
Full episode state gradients use activation recomputation, not truncated SOC.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'model_variants'),
               str(ROOT / 'external/ol-ems-hkust')]
import numpy as np
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint as recompute

import run_m0_simulation as common
from m0_scenarios import C, DT, H, TIERS, ScenarioMPC, execute, reserve_profile
from _differentiable_scenario_mpc import DifferentiableScenarioMPC, reserve_profile_tensor
from _differentiable_execution import execute_tensor
from _task_training_data import prepare_training_data, validate_checkpoint_splits, write_event_tables
from src.training_and_evaluation.m0_training import (
    atomic_save, capture_rng, restore_rng, json_write, load_checkpoint,
    runtime_versions, seed_run, sha256_file,
)

DEFAULT_EVENTS = ROOT / 'configs/m1_s1_s4_soc50_v1_20261025.json'
DISPATCH = ROOT / 'output/frozen/task_dispatch_2021_2023_v1_20261003/dispatch.csv'
TIERS_PATH = DISPATCH.parent / 'load_tiers.csv'
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True
    print('\nStopping at the next checkpoint-safe boundary; current episode is retried on resume.', flush=True)


def check_stop():
    if STOP:
        raise KeyboardInterrupt


def canonical(value):
    return json.loads(json.dumps(value, default=str, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True).encode()).hexdigest()


def code_identity():
    paths = [Path(__file__), ROOT / 'tools/_hardware_profile.py', ROOT / 'model_variants/_task_training_data.py',
             ROOT / 'tools/prepare_task_dispatch_multiyear.py',
             ROOT / 'tools/prepare_task_training_data.py',
             ROOT / 'model_variants/_differentiable_scenario_mpc.py',
             ROOT / 'model_variants/_differentiable_execution.py',
             ROOT / 'model_variants/_m0_variant_runner.py',
             ROOT / 'model_variants/_pv_only_dataset.py',
             ROOT / 'model_variants/gru_2layer_m0.py',
             ROOT / 'model_variants/lstm_2layer_m0.py',
             ROOT / 'tools/run_m0_simulation.py', ROOT / 'tools/m0_scenarios.py',
             ROOT / 'external/ol-ems-hkust/main.py',
             ROOT / 'external/ol-ems-hkust/configurations/hkust_15min.py',
             ROOT / 'external/ol-ems-hkust/src/plants/pymgrid_adapter.py',
             ROOT / 'external/ol-ems-hkust/src/training_and_evaluation/m0_training.py']
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def training_runtime():
    runtime = runtime_versions()
    for package in ('cvxpy', 'cvxpylayers', 'diffcp', 'ecos', 'scipy', 'pandas'):
        runtime[package] = importlib.metadata.version(package)
    return runtime


def loss_scales(episodes):
    # Every training timestamp counts once, even when episodes overlap.
    frame = pd.concat([e.frame.iloc[:e.steps] for e in episodes])
    frame = frame.loc[~frame.index.duplicated()].sort_index()
    mse = float(np.var(frame.pv.to_numpy()))
    money = float((frame.load * frame.import_price * DT).mean())
    if mse <= 0 or money <= 0:
        raise ValueError('Training-only loss normalisation must be positive')
    return dict(mse_kW2=mse, economic_CNY_per_step=money,
                unique_training_steps=len(frame),
                policy='PV population variance; mean all-load import cost per 15 min; train timestamps only')


def episode_tensors(episode):
    keys = ['pv', 'load', 'import_price', 'export_price', *TIERS]
    return {k: torch.as_tensor(episode.frame[k].to_numpy(copy=True), dtype=torch.float64)
            for k in keys}


def differentiable_rollout(model, mpc, episode, *, chunk_steps=8, limit_steps=None,
                           on_progress=None):
    """Recompute chunks in backward; retain the SOC derivative across chunks.

    Only the first control is executed. Histories contain actual *past* PV;
    actual future PV is used exclusively as the supervised target/execution.
    No optimiser step is taken until the whole episode has been differentiated.
    """
    n = min(episode.steps, limit_steps) if limit_steps else episode.steps
    data = episode_tensors(episode)
    soc = torch.tensor(episode.case.initial_soc * C.battery_capacity_kWh,
                       dtype=torch.float64, requires_grad=True)
    # MSE sum, economics, critical ENS, grid bill, three tier ENS, three tier demands
    totals = torch.zeros(10, dtype=torch.float64)
    start_time = time.perf_counter()

    def block(initial, first, stop):
        state = initial
        values = initial.new_zeros(10)
        for i in range(first, stop):
            check_stop()
            # Histories have one full input window before frame[0].
            lookback = episode.histories.shape[0] - len(episode.frame)
            history = episode.histories[i:i + lookback].unsqueeze(0)
            forecast = model.forecast_pv(history).reshape(-1).double()
            future = {k: data[k][i:i + H] for k in (*TIERS, 'import_price', 'export_price')}
            future['grid'] = episode.case.information(i * DT)[0]
            reserve = reserve_profile_tensor(episode.case, i * DT, forecast, data['load'][i:i + H])
            try:
                plan = mpc(forecast, future, state, reserve)
            except Exception as exc:
                raise RuntimeError(f'{episode.name}: MPC failed at step {i}, '
                                   f'{episode.frame.index[i]}') from exc
            actual = {k: data[k][i] for k in ('pv', 'load', 'import_price', 'export_price')}
            actual['grid_available'] = float(episode.case.actual_grid(np.array([i * DT]))[0])
            actual['critical_load'] = data['cl1'][i] + data['cl2'][i]
            row = execute_tensor(plan, actual, {k: data[k][i] for k in TIERS}, state)
            mse = (forecast - data['pv'][i:i + H]).square().mean()
            values = values + torch.stack((mse, row['economic_loss'],
                                          row['cl1_ens_kwh'] + row['cl2_ens_kwh'], row['grid_cost'],
                                          *(row[k + '_ens_kwh'] for k in TIERS),
                                          *(data[k][i] * DT for k in TIERS)))
            state = row['soc_after_kwh']
        return state, values

    for first in range(0, n, chunk_steps or n):
        stop = min(n, first + (chunk_steps or n))
        # Bind indices now: checkpoint backward happens after the loop advances.
        def function(state, first=first, stop=stop):
            return block(state, first, stop)
        if chunk_steps:
            soc, values = recompute(function, soc, use_reentrant=True, preserve_rng_state=True)
        else:
            soc, values = function(soc)
        totals = totals + values
        if on_progress is not None and (stop % 96 == 0 or stop == n):
            on_progress(stop, n, time.perf_counter() - start_time)
    return totals, soc, n


def evaluate_episode(model, mpc, episode, *, limit_steps=None):
    """Validation uses ECOS/HiGHS defaults + python-microgrid execution."""
    n = min(episode.steps, limit_steps) if limit_steps else episode.steps
    adapter = common._build_adapter(common.HARDWARE)
    adapter.reset(soc_kwh=episode.case.initial_soc * C.battery_capacity_kWh)
    lookback = episode.histories.shape[0] - len(episode.frame)
    sums = np.zeros(10)
    with torch.no_grad():
        for i in range(n):
            check_stop()
            forecast = model.forecast_pv(episode.histories[i:i + lookback].unsqueeze(0)).numpy().reshape(-1)
            future = {k: episode.frame[k].iloc[i:i + H].to_numpy() for k in (*TIERS, 'import_price', 'export_price')}
            future['grid'] = episode.case.information(i * DT)[0]
            reserve = reserve_profile(episode.case, i * DT, forecast, episode.frame.load.iloc[i:i + H].to_numpy())
            plan, _, _, _ = mpc.solve(forecast, future, adapter.get_soc_kwh(), reserve)
            actual = {k: float(episode.frame.iloc[i][k]) for k in ('pv', 'load', 'import_price', 'export_price')}
            actual['grid_available'] = float(episode.case.actual_grid(np.array([i * DT]))[0])
            actual['critical_load'] = float(episode.frame.iloc[i].cl1 + episode.frame.iloc[i].cl2)
            row = execute(adapter, plan, actual, {k: float(episode.frame.iloc[i][k]) for k in TIERS})
            common.verify_execution(row)
            sums += [np.mean((forecast - episode.frame.pv.iloc[i:i + H].to_numpy()) ** 2),
                     row['grid_cost'] + row['battery_cost'] + row['pv_curtailment_cost'],
                     row['cl1_ens_kwh'] + row['cl2_ens_kwh'], row['grid_cost'],
                     *(row[k + '_ens_kwh'] for k in TIERS),
                     *(float(episode.frame.iloc[i][k]) * DT for k in TIERS)]
    return sums, adapter.get_soc_kwh(), n


def penalty(ens, args):
    return sum(ens[i] * getattr(args, 'voll_' + k) for i, k in enumerate(TIERS))


def objective(mse, economic_per_step, scales, args, ens_per_step=None):
    reliability = 0. if ens_per_step is None else penalty(ens_per_step, args)
    return (args.prediction_weight * mse / scales['mse_kW2']
            + (args.economic_weight * economic_per_step + args.ens_weight * reliability)
            / scales['economic_CNY_per_step'])


def metric_row(sums, n, scales, args):
    a = np.asarray(sums, dtype=float)
    if a.shape != (10,) or not np.isfinite(a).all():
        raise FloatingPointError('Expected ten finite rollout totals')
    j = float(penalty(a[4:7], args))
    row = dict(steps=n, mse_kW2=a[0]/n, economic_CNY=a[1],
               economic_CNY_per_step=a[1]/n, critical_ENS_kWh=a[2], grid_bill_CNY=a[3],
               shedding_penalty_CNY=j, operational_CNY=a[1]+j,
               operational_CNY_per_step=(a[1]+j)/n,
               loss=float(objective(a[0]/n, a[1]/n, scales, args, a[4:7]/n)))
    for i, k in enumerate(TIERS):
        row[k + '_ENS_kWh'] = a[4+i]
        row[k + '_demand_kWh'] = a[7+i]
        row[k + '_supply_rate'] = 1-a[4+i]/a[7+i] if a[7+i] > 0 else None
    demand = a[7] + a[8]
    row['critical_demand_kWh'] = demand
    row['critical_supply_rate'] = 1-a[2]/demand if demand > 0 else None
    return row


def append_csv(path, row):
    # Idempotent on resume: an interrupted write/validation cannot duplicate
    # an epoch or episode. last.pt remains the authoritative optimiser state.
    keys = ('epoch', 'position') if 'position' in row else (('epoch', 'episode') if 'episode' in row else ('epoch',))
    existing = []
    if path.exists():
        with path.open(newline='') as stream:
            existing = list(csv.DictReader(stream))
        existing = [r for r in existing if not all(str(r[k]) == str(row[k]) for k in keys)]
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)
    temporary.replace(path)


def write_solver_diagnostics(folder, mpc, phase):
    """Persist numerical observations without adding an accuracy cutoff."""
    path = folder / 'solver_diagnostics.json'
    saved = json.loads(path.read_text()) if path.exists() else {}
    saved[phase] = dict(
        controller=mpc.version, accuracy_policy='library_defaults',
        solver_calls=mpc.solver_statistics, warnings=mpc.warning_statistics,
        max_constraint_violation=mpc.maximum_observed_constraint_violation,
        counts_include_backward_recomputation=phase == 'training')
    json_write(path, saved)


def validation(model, episodes, scales, args, folder, epoch, limit_steps=None):
    model.eval()
    mpc = ScenarioMPC(solver_defaults=True, battery_exclusive=False)
    records = []
    began = time.perf_counter()
    for j, episode in enumerate(episodes):
        try:
            sums, soc, n = evaluate_episode(model, mpc, episode, limit_steps=limit_steps)
        finally:
            write_solver_diagnostics(folder, mpc, f'validation_epoch_{epoch}')
        row = metric_row(sums, n, scales, args)
        records.append(row)
        append_csv(folder / 'validation_episodes.csv', dict(epoch=epoch, episode=episode.name, **row))
        print(f'VAL epoch {epoch} | {j+1}/{len(episodes)} | {episode.name} | loss {row["loss"]:.5f}', flush=True)
    # Every event has equal weight; this is a designed sample average, not population EENS.
    average = {k: (float(np.mean([r[k] for r in records if r[k] is not None]))
                   if any(r[k] is not None for r in records) else None) for k in records[0]}
    demand = sum(r['critical_demand_kWh'] for r in records)
    average['critical_supply_rate_pooled'] = (1-sum(r['critical_ENS_kWh'] for r in records)/demand) if demand > 0 else None
    average['elapsed_seconds'] = time.perf_counter() - began
    return average


def checkpoint_payload(base, model, optimizer, state, generator, identity):
    result = {k: copy.deepcopy(base[k]) for k in
              ('format_version', 'config', 'dataset_scaler', 'dataset_identity', 'model_architecture')}
    result.update(model_kind='M2', training_method='M2',
                  baseline_initialization=state['epoch'] == 0 and state['next_episode'] == 0,
                  cumulative_epoch=state['epoch'],
                  controller_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  optimizer_state_dict=optimizer.state_dict(), state=state,
                  rng_state=capture_rng(generator), training_identity=identity,
                  epoch=state['epoch'], best_epoch=state.get('best_epoch', 0),
                  selection_metric=identity['training'].get('selection_metric', 'loss'),
                  best_selection_value=state['best_loss'],
                  best_val_loss=state.get('best_mixed_loss', state['best_loss']), seed=identity['training']['seed'])
    return result


def output_directory(args):
    if args.resume:
        path = args.resume.resolve()
        if path.name != 'last.pt' or path.parent.parent != ROOT / 'experiments/M2':
            raise ValueError('Resume requires experiments/M2/<run>/last.pt')
        return path.parent
    if args.mode == 'train':
        for method in ('M2',):
            (ROOT / 'experiments' / method).mkdir(parents=True, exist_ok=True)
        root = ROOT / 'experiments/M2'
    else:
        root = ROOT / 'tmp' / ('m2_' + args.mode)
    name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('run-name must contain only letters, digits, underscores and hyphens')
    output = root / name
    output.mkdir(parents=True, exist_ok=False)
    return output



def main(args):
    for key in ('max_epochs', 'patience', 'checkpoint_steps', 'threads', 'smoke_steps'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('learning_rate', 'economic_weight', 'ens_weight', 'gradient_clip', 'voll_cl1', 'voll_cl2', 'voll_cl3'):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if args.resume and args.mode != 'train':
        raise ValueError('resume applies only to train')
    if not np.isfinite(args.prediction_weight) or args.prediction_weight < 0:
        raise ValueError('prediction_weight must be finite and nonnegative')
    if not np.isfinite(args.min_delta) or args.min_delta < 0:
        raise ValueError('min_delta must be finite and nonnegative')
    seed_run(args.seed, args.threads)
    generator = torch.Generator().manual_seed(args.seed)
    model, dataset, base = common.load_model(args.checkpoint)
    if base.get('training_method') in ('M1', 'M2'):
        raise ValueError('Initial checkpoint must be M0; use --resume for M2 continuation')
    validate_checkpoint_splits(base['config'])
    bundle = prepare_training_data(dataset, args.scenario_config, args.dispatch, args.load_tiers)
    scales = loss_scales(bundle.episodes['train'])
    training = {k: getattr(args, k) for k in ('seed', 'threads', 'learning_rate', 'prediction_weight',
                'economic_weight', 'ens_weight', 'voll_cl1', 'voll_cl2', 'voll_cl3', 'min_delta', 'gradient_clip', 'checkpoint_steps', 'patience', 'selection_metric')}
    training.update(method='M2', battery_exclusive=False,
                    validation_controller='continuous_relaxation_matched_to_training',
                    optimizer='Adam', weight_decay=1e-6, hardware=asdict(C),
                    differentiable_controller=DifferentiableScenarioMPC.version,
                    solver_accuracy='library_defaults_for_training_and_validation',
                    solver_warning_policy='record library warnings; do not promote inaccurate to exception',
                    constraint_residual_policy='record only; reject nonfinite outputs',
                    event_weighting='equal weight per episode; no event-probability claim',
                    state_gradient='full episode; activation recomputation; no SOC detach',
                    loss='lambda_p*M + lambda_e*C + lambda_j*J; J=VOLL dot actual tier ENS; C,J share train money scale',
                    economic='realised grid bill + battery cycle cost + PV curtailment cost; excludes ENS penalties',
                    initial_hyperparameters='Starting configuration, not validation-tuned optimal weights')
    identity = canonical(dict(training=training, scales=scales, data=bundle.metadata,
        initial_checkpoint_sha256=sha256_file(args.checkpoint), code_sha256=code_identity(),
        runtime=training_runtime()))
    folder = output_directory(args)
    logging.basicConfig(filename=folder / 'run.log', level=logging.INFO, force=True)
    logging.captureWarnings(True)
    print(f'M2 | CPU | GRU {dataset.lookback_window} -> {H}', flush=True)
    print('M2 training/selection: continuous relaxation; use MILP evaluation for final exclusive-dispatch comparisons.', flush=True)
    print('Solver accuracy: library defaults | numerical warnings recorded in run.log and solver_diagnostics.json', flush=True)
    for split in ('train', 'val'):
        lower, upper = bundle.metadata['split_intervals'][split]
        requested = bundle.metadata['requested_episode_counts'][split]
        excluded = bundle.metadata['excluded_counts'][split]
        print(f'{split.upper()} dates: [{lower}, {upper}) | requested {requested} events, '
              f'excluded by quality {excluded}', flush=True)
    missing_months = [row['month'] for row in bundle.metadata['monthly_coverage'] if not row['included']]
    if missing_months:
        print(f'WARNING: no quality-valid events in these months: {", ".join(missing_months)}; '
              'see monthly_coverage.csv and event_manifest.json', flush=True)
    print(f'Train {len(bundle.episodes["train"])} episodes / {sum(e.steps for e in bundle.episodes["train"])} steps; '
          f'validation {len(bundle.episodes["val"])} episodes / {sum(e.steps for e in bundle.episodes["val"])} steps', flush=True)
    print(f'Results: {folder}', flush=True)
    manifest = dict(method='M2', mode=args.mode, identity=identity, source_checkpoint=str(args.checkpoint.resolve()),
                    output=str(folder), max_epochs=args.max_epochs, test_used_for_training_or_selection=False,
                    M2_output_root=str(ROOT / 'experiments/M2'))
    if not args.resume:
        json_write(folder / 'manifest.json', manifest)
        write_event_tables(bundle, folder)
        for relative in identity['code_sha256']:
            destination = folder / 'source_snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        shutil.copy2(args.scenario_config, folder / 'source_scenario_config.json')
    if args.mode == 'preflight':
        json_write(folder / 'status.json', dict(status='preflight_passed', formal_training=False))
        return folder

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    state = dict(epoch=0, next_episode=0, order=[], best_loss=None, best_epoch=0, bad_epochs=0)
    if args.resume:
        saved = load_checkpoint(args.resume)
        if saved.get('training_method') != 'M2' or saved['training_identity'] != identity:
            raise ValueError('Resume rejected: method, data, code, runtime or training arguments changed. '
                             'An old eight-month run cannot resume as the multiyear experiment; '
                             'start a new run from M0 without --resume.')
        model.load_state_dict(saved['controller_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        state = saved['state']
        restore_rng(saved['rng_state'], generator)
        if state['bad_epochs'] >= args.patience:
            raise ValueError('This run has already early-stopped')

    def save_last():
        atomic_save(folder / 'last.pt', checkpoint_payload(base, model, optimizer, state, generator, identity))

    limit = args.smoke_steps if args.mode == 'smoke' else None
    if limit:
        train_episodes = [e for e in bundle.episodes['train'] if args.smoke_case in e.name][:1]
        val_episodes = [e for e in bundle.episodes['val'] if args.smoke_case in e.name][:1]
        if not train_episodes or not val_episodes:
            raise ValueError('Requested smoke-case must exist in train and val')
    else:
        train_episodes, val_episodes = bundle.episodes['train'], bundle.episodes['val']
    began = time.perf_counter()
    mpc = None
    try:
        if state['best_loss'] is None:
            save_last()
            json_write(folder / 'status.json', dict(status='validating_baseline', epoch=0,
                                                   resume_from=str(folder / 'last.pt')))
            baseline = validation(model, val_episodes, scales, args, folder, 0, limit)
            state['best_loss'] = baseline[args.selection_metric]
            state['best_mixed_loss'] = baseline['loss']
            append_csv(folder / 'metrics.csv', dict(epoch=0, **baseline))
            # A worsened fine-tune must not silently replace a better M0 baseline.
            atomic_save(folder / 'best.pt', checkpoint_payload(base, model, optimizer, state, generator, identity))
            save_last()

        mpc = DifferentiableScenarioMPC()
        max_epochs = 1 if limit else args.max_epochs
        while state['epoch'] < max_epochs and state['bad_epochs'] < args.patience:
            epoch = state['epoch'] + 1
            if not state['order']:
                state['order'] = torch.randperm(len(train_episodes), generator=generator).tolist()
                state['next_episode'] = 0
                save_last()
            model.train()
            for position in range(state['next_episode'], len(state['order'])):
                check_stop()
                episode = train_episodes[state['order'][position]]
                optimizer.zero_grad(set_to_none=True)
                # Saved RNG is before this episode. An interruption replays it exactly.
                epoch_tick = time.perf_counter()
                def progress(done, total, seconds):
                    print(f'TRAIN epoch {epoch}/{max_epochs} | event {position+1}/{len(train_episodes)} '
                          f'| {episode.name} | forward {done}/{total} | {seconds:.1f}s', flush=True)
                sums, soc, n = differentiable_rollout(model, mpc, episode,
                    chunk_steps=args.checkpoint_steps, limit_steps=limit, on_progress=progress)
                loss = objective(sums[0] / n, sums[1] / n, scales, args, sums[4:7] / n)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite M2 loss')
                print('  backward: recomputing chunks and updating GRU', flush=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
                if float(grad) <= 0:
                    raise RuntimeError('Zero total gradient: refusing a no-update run')
                optimizer.step()
                row = metric_row(sums.detach().numpy(), n, scales, args)
                state['next_episode'] = position + 1
                elapsed = time.perf_counter() - epoch_tick
                append_csv(folder / 'training_episodes.csv', dict(epoch=epoch, position=position,
                    episode=episode.name, **row, gradient_norm=float(grad), seconds=elapsed))
                save_last()
                write_solver_diagnostics(folder, mpc, 'training')
                json_write(folder / 'status.json', dict(status='training', epoch=epoch,
                    completed_episodes=position+1, total_episodes=len(train_episodes), latest=row,
                    latest_episode_seconds=elapsed, elapsed_seconds=time.perf_counter()-began))
                print(f'  saved | loss {row["loss"]:.5f} | grid bill {row["grid_bill_CNY"]:.3f} CNY '
                      f'| grad {float(grad):.4g} | {elapsed:.1f}s', flush=True)
                del sums, loss, soc
            json_write(folder / 'status.json', dict(status='validating', epoch=epoch,
                                                   resume_from=str(folder / 'last.pt')))
            val = validation(model, val_episodes, scales, args, folder, epoch, limit)
            append_csv(folder / 'metrics.csv', dict(epoch=epoch, **val))
            improved = val[args.selection_metric] < state['best_loss'] - args.min_delta
            state.update(epoch=epoch, order=[], next_episode=0,
                         bad_epochs=0 if improved else state['bad_epochs']+1)
            if improved:
                state.update(best_loss=val[args.selection_metric], best_mixed_loss=val['loss'], best_epoch=epoch)
                atomic_save(folder / 'best.pt', checkpoint_payload(base, model, optimizer, state, generator, identity))
            save_last()
            print(f'Epoch {epoch}: validation loss {val["loss"]:.6f}; '
                  f'selection {args.selection_metric}={val[args.selection_metric]:.6f}; '
                  f'best epoch {state["best_epoch"]}', flush=True)
        status = 'smoke_completed' if limit else 'completed'
        json_write(folder / 'status.json', dict(status=status, formal_training=not bool(limit),
            epoch=state['epoch'], best_epoch=state['best_epoch'], best_loss=state['best_mixed_loss'],
            selection_metric=args.selection_metric, best_selection_value=state['best_loss'],
            early_stopped=state['bad_epochs'] >= args.patience,
            baseline_retained=state['best_epoch']==0, elapsed_seconds=time.perf_counter()-began))
    except BaseException as exc:
        if mpc is not None:
            write_solver_diagnostics(folder, mpc, 'training')
            if mpc.last_failure is not None:
                json_write(folder / 'failure_mpc.json', mpc.last_failure)
        # last.pt already contains the last completely updated episode; never
        # replace it with parameters/RNG halfway through an interrupted episode.
        json_write(folder / 'status.json', dict(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            epoch=state['epoch'], resume_from=str(folder / 'last.pt'), error=traceback.format_exc()))
        raise
    print(f'{status}: {folder}', flush=True)
    return folder


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('preflight', 'smoke', 'train'), default='train')
    p.add_argument('--checkpoint', type=Path, default=common.DEFAULT_CHECKPOINT)
    p.add_argument('--scenario-config', type=Path, default=DEFAULT_EVENTS)
    p.add_argument('--dispatch', type=Path, default=DISPATCH)
    p.add_argument('--load-tiers', type=Path, default=TIERS_PATH)
    p.add_argument('--run-name')
    p.add_argument('--resume', type=Path)
    p.add_argument('--max-epochs', type=int, default=7)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--prediction-weight', type=float, default=1.0)
    p.add_argument('--economic-weight', type=float, default=5.0)
    p.add_argument('--ens-weight', type=float, default=5.0)
    p.add_argument('--voll-cl1', type=float, default=50.0)
    p.add_argument('--voll-cl2', type=float, default=10.0)
    p.add_argument('--voll-cl3', type=float, default=2.0)
    p.add_argument('--min-delta', type=float, default=1e-4)
    p.add_argument('--selection-metric', choices=('operational_CNY_per_step',), default='operational_CNY_per_step',
                   help='Validation episode-mean metric minimized for best checkpoint and early stopping')
    p.add_argument('--gradient-clip', type=float, default=1.0)
    p.add_argument('--checkpoint-steps', type=int, default=8,
                   help='Recompute this many steps at once in backward; full cross-chunk SOC gradients remain')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--smoke-steps', type=int, default=4)
    p.add_argument('--smoke-case', default='S4_observed_low_outage_24h',
                   help='Case-name substring for the one train/validation smoke episode')
    return p


if __name__ == '__main__':
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    main(parser().parse_args())
