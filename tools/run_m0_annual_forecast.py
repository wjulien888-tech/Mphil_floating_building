"""Monthly walk-forward M0 forecasting; default is preparation only, no training.

A single initial-training scaler is shared by all folds. The monthly models
inherit the preceding selected weights; Adam moments are reset each month.
No missing daylight is filled. Training, review and coupled MPC/environment
testing are separate explicit stages; none automatically starts the next stage.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "model_variants"), str(ROOT / "external/ol-ems-hkust")]
import numpy as np
import pandas as pd
import torch
from _pv_only_dataset import PVOnlyDataset
from _m0_variant_runner import (build_parser, _runtime_config, _dataset_identity, _build_controller,
                               _model_architecture, _variant_code_fingerprint, _configure_logging)
from gru_2layer_m0 import PVGRUController
from _m0_variant_evaluation import ErrorAccumulator, evaluate_m0
from src.training_and_evaluation.m0_training import (train_m0, seed_run, load_checkpoint, json_write,
                                                     sha256_file, runtime_versions)
from preflight_annual_experiment import DEFAULT_DATA, initial_arguments

TRAIN_START = pd.Timestamp("2021-07-01")
TEST_START = pd.Timestamp("2023-03-01")
TEST_END = pd.Timestamp("2024-03-01")


def month_folds():
    rows = []
    for month in pd.date_range(TEST_START, TEST_END, freq="MS", inclusive="left"):
        first = month == TEST_START
        val_start = pd.Timestamp("2023-01-01") if first else month - pd.Timedelta(days=14)
        rows.append(dict(month=month.strftime("%Y-%m"), training_start=str(TRAIN_START),
            training_end_exclusive=str(val_start), validation_start=str(val_start),
            labels_available_before=str(month), test_origin_start=str(month),
            test_origin_end_exclusive=str(month + pd.offsets.MonthBegin(1)),
            test_target_end_exclusive=str(TEST_END), update="initial" if first else "warm_start"))
    return rows


def window_anchors(frame, start, end, *, origin_end=None, history=96, horizon=96):
    """Require the entire history and target valid; target labels end before end.

    For tests, origin_end partitions issuing months, not target months. Thus
    forecasts issued near month-end retain their full horizons when possible.
    No score crosses the final annual boundary; invalid intervals remain gaps.
    """
    stamps = pd.DatetimeIndex(frame.datetime)
    candidates = np.arange(history, len(frame) - horizon + 1, dtype=np.int64)
    history_bad = np.r_[0, np.cumsum(~frame.history_valid.to_numpy(dtype=bool))]
    target_bad = np.r_[0, np.cumsum(~frame.pv_target_valid.to_numpy(dtype=bool))]
    within = (stamps[candidates] >= pd.Timestamp(start)) & (stamps[candidates + horizon - 1] < pd.Timestamp(end))
    if origin_end is not None:
        within &= stamps[candidates] < pd.Timestamp(origin_end)
    valid = (history_bad[candidates] == history_bad[candidates-history]) & (target_bad[candidates+horizon] == target_bad[candidates])
    return candidates[within & valid], int(within.sum()), int((within & ~valid).sum())


def fold_views(base, fold):
    views, counts = {}, {}
    frame = base.source_frame
    for name, start, end, origin_end in (
        ("train", fold["training_start"], fold["training_end_exclusive"], None),
        ("val", fold["validation_start"], fold["labels_available_before"], None),
        ("test", fold["test_origin_start"], fold["test_target_end_exclusive"], fold["test_origin_end_exclusive"]),
    ):
        anchors, eligible, rejected = window_anchors(frame, start, end, origin_end=origin_end)
        if not len(anchors):
            raise ValueError(f"No valid {name} samples for {fold['month']}")
        view = copy.copy(base)
        view._anchors, view.split = anchors, name
        # The underlying frozen CSV and scaler stay unchanged. Actual sample
        # boundaries are recorded in fold metadata, not relabelled in the CSV.
        views[name] = view
        counts[name] = dict(valid_origins=len(anchors), possible_origins=eligible,
            quality_rejected_origins=rejected, coverage_fraction=len(anchors)/eligible,
            first_origin=str(frame.datetime.iloc[anchors[0]]),
            last_origin=str(frame.datetime.iloc[anchors[-1]]),
            last_target=str(frame.datetime.iloc[anchors[-1]+95]))
    return views, counts


def inputs(data, output):
    args = build_parser("GRU").parse_args(initial_arguments(data, output))
    identity = _dataset_identity(args)
    config = _runtime_config(args)
    base = PVOnlyDataset(config)
    if config.split_dates() != {"train": ("2021-07-01", "2023-01-01"),
                                "val": ("2023-01-01", "2023-03-01"),
                                "test": ("2023-03-01", "2024-03-01")}:
        raise ValueError("Annual runner requires the agreed 18/2/12 split")
    return args, identity, config, base


def prepare_protocol(data, output, *, update_lr=1e-4, update_epochs=20):
    if not np.isfinite(update_lr) or update_lr <= 0 or update_epochs < 1:
        raise ValueError("Invalid update learning rate/epoch limit")
    args, identity, config, base = inputs(data, output)
    rows = []
    for fold in month_folds():
        _, counts = fold_views(base, fold)
        rows.append({**fold, "sample_counts": counts})
    protocol = dict(schema_version=1, status="prepared_not_run", model="M0_GRU_96_to_96",
        dataset_identity=identity, initial_split_dates=config.split_dates(),
        fixed_scaler=base.state_dict(), folds=rows,
        monthly_update=dict(weights="previous month's selected model", history="expanding from 2021-07-01",
            validation="last 14 days before next month; all target labels must precede cutoff",
            learning_rate=update_lr, max_epochs=update_epochs, patience=8, min_delta=0.01,
            optimizer="fresh Adam each month, inherit model weights only",
            selection="strict minimum validation MSE including incoming weights as epoch 0"),
        initial_training=dict(learning_rate=args.learning_rate, max_epochs=args.max_epochs,
                              patience=args.patience, seed=args.seed),
        missing_policy="No daytime imputation; all 96 history AND 96 target points must be valid",
        boundary_policy="Issue month selects weights; full horizon may cross month end; no targets from March 2024",
        fixed_model_control=True, persistence_controls=["same slot previous day", "last observation"],
        source_report_groups=["2023-03_to_2023-12_constructed", "2024-01_to_2024-02_PVGIS6_only"],
        elapsed_time_assumption="Offline monthly-cutoff backtest; training/deployment latency is not simulated",
        execution="Separate manual stages: train-month, review, test-month with MPC/environment",
        test_gate="Training returns before any test evaluation. Review training/validation first, then explicitly invoke test-month.",
        closed_loop_missing_policy="Error on invalid PV or load; no imputation, no skipped intervals or automatic SOC reset across gaps",
        initial_old_checkpoint_reuse=False,
        evaluation="All valid issued forecast/horizon pairs, overlapping targets included; report sample coverage",
        experimental_status="Retrospective chronological backtest; historical 2023 data previously used in project development")
    protocol["fixed_scaler"] = {key: value.tolist() if isinstance(value, torch.Tensor) else value
                                for key, value in protocol["fixed_scaler"].items()}
    protocol["code_sha256"] = {**_variant_code_fingerprint(), "tools/run_m0_annual_forecast.py": sha256_file(__file__),
                             "tools/preflight_annual_experiment.py": sha256_file(ROOT / "tools/preflight_annual_experiment.py")}
    for name in ('run_m0_simulation.py','run_m0_scenarios.py','m0_scenarios.py'):
        protocol['code_sha256']['tools/'+name] = sha256_file(ROOT/'tools'/name)
    json_write(Path(data) / "annual_forecast_protocol.json", protocol)
    pd.DataFrame([{**{k:v for k,v in row.items() if k != "sample_counts"},
                   **{f"{split}_{key}": value for split, counts in row["sample_counts"].items()
                      for key,value in counts.items()}} for row in rows]).to_csv(Path(data) / "monthly_forecast_folds.csv", index=False)
    return protocol


def verify_parent(checkpoint, shared, previous_fold):
    if checkpoint.get("dataset_identity") != shared["dataset_identity"] or checkpoint.get("model_architecture") != shared["model_architecture"]:
        raise ValueError("Warm-start data/model mismatch")
    if checkpoint.get("annual_fold") != previous_fold:
        raise ValueError("Warm-start checkpoint does not come from the preceding month")
    for key in ("scaling_factors_min", "scaling_factors_max"):
        if not torch.equal(checkpoint["dataset_scaler"][key], shared["dataset_scaler"][key]):
            raise ValueError("Warm-start scaler changed")


def pool_results(output, months):
    """Recompute pooled sufficient statistics; never average monthly RMSE/R2."""
    groups = {"annual": months, "2023_constructed": [x for x in months if x.startswith("2023")],
              "2024_PVGIS6_only": [x for x in months if x.startswith("2024")]}
    pooled = []
    for group, members in groups.items():
        if not members:
            continue
        for arm in ("updated", "fixed_initial"):
            acc = {method: ErrorAccumulator(1) for method in ("M0", "persistence_24h", "persistence_last")}
            for month in members:
                path = output / month / arm / "predictions.csv.gz"
                for chunk in pd.read_csv(path, chunksize=100000):
                    target = chunk.pv_actual_kw.to_numpy()[:,None]
                    for method, column in (("M0", "pv_forecast_kw"), ("persistence_24h", "persistence_24h_kw"),
                                           ("persistence_last", "persistence_last_kw")):
                        acc[method].update(chunk[column].to_numpy()[:,None], target, np.ones_like(target, dtype=bool))
            pooled.extend(dict(source_group=group, arm=arm, method=method, **meter.metrics()) for method,meter in acc.items())
    pd.DataFrame(pooled).to_csv(output / "pooled_metrics.csv", index=False)


def run(data, output, protocol):
    raise RuntimeError("Combined train/test workflow retired. Use separate train-month, review and test-month stages.")


def select_fold(month):
    matches = [fold for fold in month_folds() if fold['month'] == month]
    if len(matches) != 1:
        raise ValueError('Month must be between 2023-03 and 2024-02')
    return matches[0]


def review_month(output, month):
    """Inspect saved train/validation records only; no inference or test data."""
    folder = Path(output) / month / 'training'
    metadata = json.loads((folder / 'run_metadata.json').read_text())
    if metadata.get('status') != 'completed' or metadata.get('model_status') != 'trained_M0':
        raise ValueError('Review requires completed full training, not a smoke or interrupted run')
    metrics = pd.read_csv(folder / 'metrics.csv')
    if metrics.empty or not np.isfinite(metrics[['train_loss_mse', 'val_loss_mse']]).all().all():
        raise ValueError('Training metrics missing or nonfinite')
    checkpoint = load_checkpoint(folder / 'best.pt')
    if checkpoint.get('annual_fold') != select_fold(month):
        raise ValueError('Selected checkpoint belongs to a different month')
    if checkpoint['epoch'] != metadata['best_epoch']:
        raise ValueError('Selected checkpoint and completed training disagree')
    review = dict(status='awaiting_user_review_before_test', month=month,
        checkpoint_sha256=sha256_file(folder / 'best.pt'),
        metrics_sha256=sha256_file(folder / 'metrics.csv'),
        epochs_completed=int(metadata['epochs_completed']), best_epoch=int(checkpoint['epoch']),
        best_validation_mse=float(checkpoint['val_loss_mse']),
        best_validation_rmse=float(checkpoint['val_loss_mse'])**.5,
        latest_train_mse=float(metrics.iloc[-1].train_loss_mse),
        latest_validation_mse=float(metrics.iloc[-1].val_loss_mse),
        incoming_validation_mse=metadata.get('incoming_validation_mse'),
        stop_reason=metadata['stop_reason'], test_evaluated=False,
        review_points=['loss convergence and train/validation gap', 'best epoch and early stopping',
                       'finite gradients/metrics and saved checkpoint', 'no test-driven parameter selection'])
    json_write(folder.parent / 'training_review.json', review)
    return review


def train_month(data, output, protocol, month, resume=False):
    """One model update only. Always return before invoking any test or MPC."""
    fold = select_fold(month)
    output = Path(output).resolve()
    args, identity, config, base = inputs(data, output)
    if identity != protocol['dataset_identity']:
        raise ValueError('Data identity changed')
    index = month_folds().index(fold)
    args.seed += index
    seed_run(args.seed, args.torch_num_threads)
    model = _build_controller(config, base, args, PVGRUController)
    shared = dict(config=asdict(config), dataset_scaler=base.state_dict(),
        model_architecture=_model_architecture(config, base, args), dataset_identity=identity,
        code_sha256=protocol['code_sha256'], seed=args.seed)
    parent_record = None
    if index:
        previous = month_folds()[index-1]
        marker = output / previous['month'] / 'coupled_test_completed.json'
        if not marker.exists():
            raise ValueError('Complete the preceding month coupled test before training the next month')
        parent_path = output / previous['month'] / 'training/best.pt'
        parent = load_checkpoint(parent_path)
        if json.loads(marker.read_text())['checkpoint_sha256'] != sha256_file(parent_path):
            raise ValueError('Preceding test and selected parent weights differ')
        verify_parent(parent, shared, previous)
        model.load_state_dict(parent['controller_state_dict'])
        args.learning_rate = protocol['monthly_update']['learning_rate']
        args.max_epochs = protocol['monthly_update']['max_epochs']
        args.warm_start_baseline = True
        parent_record = dict(path=str(parent_path), sha256=sha256_file(parent_path))
    views, counts = fold_views(base, fold)
    destination = output / month / 'training'
    if destination.exists() and resume:
        if not (destination/'last.pt').is_file():
            raise ValueError('Interrupted training has no epoch-boundary checkpoint; inspect before retrying')
        args.resume = destination/'last.pt'
    else:
        destination.mkdir(parents=True, exist_ok=False)
    _configure_logging(destination)
    json_write(destination.parent / 'protocol.json', protocol)
    metadata = {**shared, 'annual_fold':fold, 'sample_counts':counts, 'warm_start_source':parent_record}
    train_m0(model, views['train'], views['val'], destination, args, metadata,
             dict(annual_fold=fold, arguments=vars(args), sample_counts=counts,
                  warm_start_source=parent_record, runtime=runtime_versions()))
    review = review_month(output, month)
    print(json.dumps(review, ensure_ascii=False, indent=2))
    return review


def test_month(data, output, month, scenario_config):
    """Explicit post-review test: model -> ScenarioMPC -> physical environment."""
    if scenario_config is None:
        raise ValueError('Coupled testing requires an explicitly selected --scenario-config for this month')
    fold = select_fold(month)
    root = Path(output).resolve() / month
    checkpoint_path = root / 'training/best.pt'
    review = json.loads((root / 'training_review.json').read_text())
    if review['checkpoint_sha256'] != sha256_file(checkpoint_path):
        raise ValueError('Checkpoint changed since the training review')
    config = json.loads(Path(scenario_config).read_text())
    start = pd.Timestamp(config['start'])
    steps = float(config['days']) * 96
    if not steps.is_integer() or steps < 1:
        raise ValueError('Invalid scenario length')
    if start != pd.Timestamp(fold['test_origin_start']) or start + pd.Timedelta(minutes=15*steps) != pd.Timestamp(fold['test_origin_end_exclusive']):
        raise ValueError('Monthly coupled test config must cover exactly the issuing month; do not silently shorten or skip gaps')
    # This entry uses the existing tier-prioritised S0--S5 execution model.
    # Missing PV/load still stops in its preflight before a case is executed.
    import run_m0_scenarios as scenarios
    settings = scenarios.parser().parse_args(['--config',str(scenario_config),
        '--checkpoint',str(checkpoint_path),'--dispatch',str(Path(data)/'dispatch/dispatch.csv'),
        '--load-tiers',str(Path(data)/'dispatch/load_tiers.csv'),
        '--invalid-pv','error','--output',str(root/'coupled_test')])
    scenarios.main(settings)
    state = json.loads((root/'coupled_test/suite_status.json').read_text())
    if not state.get('full_matrix_completed'):
        raise RuntimeError('Coupled test incomplete; next monthly update remains blocked')
    report_path = root/'coupled_test/report_data/provenance.json'
    report_metrics = json.loads(report_path.read_text())
    if report_metrics['checkpoint_sha256'] != review['checkpoint_sha256']:
        raise ValueError('Report metrics checkpoint differs from reviewed model')
    # Forecast scores are produced only inside this explicit testing stage.
    args, identity, model_config, base = inputs(data, output)
    selected = load_checkpoint(checkpoint_path)
    if selected['dataset_identity'] != identity:
        raise ValueError('Testing data and trained model differ')
    views, counts = fold_views(base, fold)
    for arm, path in (('updated',checkpoint_path),
                      ('fixed_initial',Path(output)/'2023-03/training/best.pt')):
        saved = load_checkpoint(path)
        model = _build_controller(model_config,base,args,PVGRUController)
        model.load_state_dict(saved['controller_state_dict'])
        destination = root / arm
        destination.mkdir(exist_ok=False)
        evaluate_m0(model,views['test'],destination,args,
            dict(annual_fold=fold,arm=arm,sample_counts=counts,checkpoint=str(path),
                 checkpoint_sha256=sha256_file(path),annual_protocol=True,
                 note='Forecast controls; coupled scenario results are for the updated model'))
    json_write(root/'coupled_test_completed.json',dict(month=month,
        checkpoint_sha256=sha256_file(checkpoint_path), config_sha256=sha256_file(scenario_config),
        status='completed', start=str(start), steps=int(steps),
        report_metrics=str(report_path), report_metrics_sha256=sha256_file(report_path),
        note='Monthly scenario cases; no claim of continuous SOC across independent cases or months'))
    completed = [row['month'] for row in month_folds()
                 if (Path(output)/row['month']/'coupled_test_completed.json').exists()]
    pool_results(Path(output),completed)
    # Retain each month/case and its checkpoint; do not average supply rates or
    # add independent alternative scenarios into one fictitious annual case.
    report_tables = [pd.read_csv(Path(output)/m/'coupled_test/report_data/全部案例绘图指标.csv')
                     for m in completed]
    pd.concat(report_tables, ignore_index=True).to_csv(
        Path(output)/'全年逐月逐案例报告指标.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/annual_18_2_12_m0")
    parser.add_argument("--update-learning-rate", type=float, default=1e-4)
    parser.add_argument("--update-max-epochs", type=int, default=20)
    parser.add_argument('--stage', choices=['prepare','train-month','review','test-month'], default='prepare')
    parser.add_argument('--month', default='2023-03')
    parser.add_argument('--scenario-config',type=Path)
    parser.add_argument("--run", action="store_true", help="Retired: refuses automatic combined training/testing")
    args = parser.parse_args()
    if args.run:
        parser.error('--run has been disabled. Explicitly choose train-month, review or test-month; training never starts testing.')
    if args.stage == 'review':
        print(json.dumps(review_month(args.output,args.month),ensure_ascii=False,indent=2))
        return
    if args.stage == 'test-month':
        test_month(args.data,args.output,args.month,args.scenario_config)
        return
    protocol = prepare_protocol(args.data, args.output, update_lr=args.update_learning_rate, update_epochs=args.update_max_epochs)
    if args.stage == 'train-month':
        train_month(args.data,args.output,protocol,args.month)
    else:
        print(json.dumps(dict(status="prepared_not_run", folds=len(protocol["folds"]),
            protocol=str(args.data / "annual_forecast_protocol.json"),
            samples={row["month"]: row["sample_counts"]["test"]["valid_origins"] for row in protocol["folds"]}), indent=2))


if __name__ == "__main__":
    main()
