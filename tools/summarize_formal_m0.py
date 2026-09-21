"""Summarize the two-model PV-only M0 suite; --legacy audits old three-seed M0."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def audit_predictions(output: Path, summary: dict) -> None:
    """Rebuild every reported error group directly from exported predictions."""
    # Count, signed, absolute and squared error, independently of trainer code.
    stats = {(method, period): np.zeros((4, 96), dtype=np.float64)
             for method in ("M0", "persistence_24h")
             for period in ("all", "daylight", "non_daylight")}
    total = 0
    threshold = float(summary["arguments"]["daylight_threshold_kw"])
    for part in pd.read_csv(output / "predictions.csv.gz", chunksize=100000):
        total += len(part)
        h = part.horizon_step.to_numpy(dtype=int) - 1
        assert np.array_equal(h + 1, part.horizon_step) and ((h >= 0) & (h < 96)).all()
        actual = part.pv_actual_kw.to_numpy()
        day = part.daylight_by_truth.to_numpy(dtype=bool)
        assert np.all((day == (actual > threshold)) | (np.abs(actual - threshold) < 1e-7))
        for method, column in (("M0", "pv_forecast_kw"), ("persistence_24h", "persistence_24h_kw")):
            error = part[column].to_numpy() - actual
            assert np.isfinite(error).all()
            for period, mask in (("all", np.ones(len(part), dtype=bool)), ("daylight", day), ("non_daylight", ~day)):
                selected, values = h[mask], error[mask]
                accum = stats[method, period]
                accum[0] += np.bincount(selected, minlength=96)
                accum[1] += np.bincount(selected, weights=values, minlength=96)
                accum[2] += np.bincount(selected, weights=np.abs(values), minlength=96)
                accum[3] += np.bincount(selected, weights=values ** 2, minlength=96)
    assert total == summary["forecast_origins"] * 96
    for method in ("M0", "persistence_24h"):
        assert (stats[method, "all"][0] == summary["forecast_origins"]).all()
    for table, horizon in ((pd.DataFrame(summary["metrics"]), False),
                           (pd.read_csv(output / "forecast_metrics.csv"), False),
                           (pd.read_csv(output / "horizon_metrics.csv"), True)):
        keys = ["method", "period"] + (["horizon_step"] if horizon else [])
        assert len(table) == 6 * (96 if horizon else 1) and not table.duplicated(keys).any()
        for row in table.itertuples():
            accum = stats[row.method, row.period]
            values = accum[:, row.horizon_step - 1] if horizon else accum.sum(axis=1)
            count, signed, absolute, squared = values
            assert count == row.count
            expected = {"mae_kw": absolute / count, "mse_kw2": squared / count,
                        "rmse_kw": np.sqrt(squared / count), "mbe_kw": signed / count} if count else {}
            for metric in ("mae_kw", "mse_kw2", "rmse_kw", "mbe_kw"):
                actual_metric = getattr(row, metric)
                if count:
                    assert np.isclose(actual_metric, expected[metric], rtol=2e-6, atol=1e-6), (row, metric)
                else:
                    assert actual_metric is None or np.isnan(actual_metric)


def summarize_legacy(suite: Path) -> None:
    suite = suite.resolve()
    assert read(suite / "suite_status.json")["stage"] == "completed", "Suite is not finished"
    protocol = read(suite / "protocol.json")
    frozen = ROOT / "output/frozen/hkust_m0_v1_20260909"
    manifest = read(frozen / "manifest.json")
    assert sha256(frozen / "manifest.json") == protocol["data_manifest_sha256"]
    for entry in manifest["outputs"]:
        assert sha256(frozen / entry["path"]) == entry["sha256"]
    source_hashes = read(suite / "source_sha256.json")
    for relative, expected in source_hashes.items():
        assert sha256(suite / "source_snapshot" / relative) == expected, f"Snapshot changed: {relative}"
    expected_identity = {"dispatch_sha256": sha256(frozen / "dispatch.csv"),
                         "load_tiers_sha256": sha256(frozen / "load_tiers.csv"),
                         "manifest_sha256": protocol["data_manifest_sha256"]}
    prefix = "external/ol-ems-hkust/"
    expected_code = {relative[len(prefix):]: value for relative, value in source_hashes.items()
                     if relative.startswith(prefix) and
                     (relative == prefix + "main.py" or relative.startswith(prefix + "src/")
                      or relative.startswith(prefix + "configurations/"))}
    training, evaluations, curves, checkpoints = [], [], {}, {}
    for seed in protocol["seeds"]:
        run = suite / f"seed_{seed}"
        metadata = read(run / "run_metadata.json")
        assert metadata["status"] == "completed" and metadata["model_status"] == "trained_M0"
        assert metadata["test_evaluated"] is False
        curve = pd.read_csv(run / "metrics.csv")
        curves[seed] = curve
        best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        for artifact in (metadata, best, last):
            assert artifact["dataset_identity"] == expected_identity
            assert artifact["code_sha256"] == expected_code
            assert artifact["seed"] == seed
        entry = next(entry for entry in protocol["runs"] if entry["seed"] == seed)
        command = entry["train_command"][2:]
        command_args = dict(zip((x.removeprefix("--") for x in command[::2]), command[1::2]))
        for field in ("seed", "learning_rate", "weight_decay", "train_batch_size", "val_batch_size",
                      "max_epochs", "patience", "min_delta", "num_workers", "device", "torch_num_threads",
                      "hidden_dim", "num_layers"):
            actual = metadata["arguments"][field]
            assert actual == type(actual)(command_args[field]), field
            if field in last["training_signature"]:
                assert actual == last["training_signature"][field], field
        assert metadata["arguments"]["limit_train_batches"] == 1.0
        assert metadata["arguments"]["limit_val_batches"] == 1.0
        assert metadata["arguments"]["fast_dev_run"] == 0
        assert best["model_architecture"] == {"hidden_dim": 32, "num_layers": 1, "input_dim": 5}
        assert best["config"]["lookback_steps"] == 672 and best["config"]["prediction_horizon"] == 96
        assert np.array_equal(curve.epoch, np.arange(1, len(curve) + 1))
        best_row = curve.loc[curve.val_loss_mse.idxmin()]
        assert best["epoch"] == best_row.epoch == metadata["best_epoch"]
        assert last["epoch"] == curve.epoch.max() == metadata["epochs_completed"]
        assert np.isclose(best["val_loss_mse"], best_row.val_loss_mse, rtol=1e-12)
        assert curve.train_samples.eq(manifest["splits"]["train"]["valid_windows_stride_1"]).all()
        assert curve.val_samples.eq(manifest["splits"]["val"]["valid_windows_stride_1"]).all()
        assert "optimizer_state_dict" in last and "rng_state" in last and "best_model_snapshot" in last
        checkpoints[str(seed)] = {name: sha256(run / name) for name in ("best.pt", "last.pt")}
        training.append({"seed": seed, "epochs": metadata["epochs_completed"],
                         "best_epoch": metadata["best_epoch"], "stop_reason": metadata["stop_reason"],
                         "best_val_mse_kw2": best["val_loss_mse"],
                         "training_seconds": float(curve.seconds.sum())})
        for split in ("val", "test"):
            output = suite / f"seed_{seed}_{split}"
            summary = read(output / "evaluation_summary.json")
            assert summary["status"] == "completed" and summary["complete_split"]
            assert summary["forecast_origins"] == manifest["splits"][split]["valid_windows_stride_1"]
            assert summary["checkpoint_sha256"] == checkpoints[str(seed)]["best.pt"]
            assert summary["dataset_identity"] == expected_identity
            assert summary["code_sha256"] == expected_code
            assert summary["code_sha256"] == summary["checkpoint_code_sha256"]
            assert summary["checkpoint_seed"] == seed and summary["checkpoint_epoch"] == best["epoch"]
            metrics = pd.DataFrame(summary["metrics"])
            val_mse = metrics.loc[(metrics.method == "M0") & (metrics.period == "all"), "mse_kw2"].item()
            if split == "val":
                assert np.isclose(val_mse, best["val_loss_mse"], rtol=1e-10)
            audit_predictions(output, summary)
            metrics.insert(0, "split", split)
            metrics.insert(0, "seed", seed)
            evaluations.append(metrics)
    table = pd.concat(evaluations, ignore_index=True)
    table.to_csv(suite / "all_forecast_metrics.csv", index=False)
    train_table = pd.DataFrame(training)
    train_table.to_csv(suite / "training_summary.csv", index=False)
    stats = table.groupby(["split", "method", "period"])[["mae_kw", "rmse_kw", "mbe_kw", "mse_kw2"]].agg(["mean", "std"])
    stats.to_csv(suite / "metrics_mean_std.csv")
    generator_snapshot = suite / "analysis_source/summarize_formal_m0.py"
    generator_snapshot.parent.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), generator_snapshot)
    audit = {"status": "passed", "report_generator_sha256": sha256(generator_snapshot),
             "checks": ["source_and_frozen_data_hashes", "complete_train_val_epochs",
              "best_matches_minimum_validation", "last_contains_resume_state", "no_training_test_selection",
              "complete_independent_val_test", "checkpoint_and_evaluation_code_match",
              "exported_prediction_count_and_error_reconstruction"], "checkpoint_sha256": checkpoints}
    (suite / "artifact_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
    colors = ["#176b9a", "#b55a25", "#418154"]
    for color, (seed, curve) in zip(colors, curves.items()):
        axes[0].plot(curve.epoch, curve.val_loss_mse, color=color, label=f"Seed {seed}: validation")
        axes[0].plot(curve.epoch, curve.train_loss_mse, color=color, alpha=.45, linestyle="--")
        best = curve.loc[curve.val_loss_mse.idxmin()]
        axes[0].scatter([best.epoch], [best.val_loss_mse], color=color, marker="*", s=90)
        horizon = pd.read_csv(suite / f"seed_{seed}_test/horizon_metrics.csv")
        m0 = horizon.loc[(horizon.method == "M0") & (horizon.period == "all")]
        axes[1].plot(m0.horizon_step * .25, m0.rmse_kw, color=color, label=f"M0 seed {seed}")
    baseline = horizon.loc[(horizon.method == "persistence_24h") & (horizon.period == "all")]
    axes[1].plot(baseline.horizon_step * .25, baseline.rmse_kw, color="#777777", linestyle="--", label="Previous-day persistence")
    axes[0].set(xlabel="Epoch", ylabel="MSE (kW²)", title="Training (dashed) and validation; star = best")
    axes[1].set(xlabel="Forecast horizon (hours)", ylabel="Test RMSE (kW)", title="Full test split, all origin / horizon pairs")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(suite / "training_and_forecast.png", dpi=180)
    fig.savefig(suite / "training_and_forecast.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    for color, seed in zip(colors, protocol["seeds"]):
        first = pd.read_csv(suite / f"seed_{seed}_test/predictions.csv.gz", nrows=96)
        hours = np.arange(96) * .25
        ax.plot(hours, first.pv_forecast_kw, color=color, label=f"M0 seed {seed}")
    ax.plot(hours, first.pv_actual_kw, color="#222222", linewidth=2, label="Constructed PV label")
    ax.plot(hours, first.persistence_24h_kw, color="#888888", linestyle="--", label="Previous-day persistence")
    ax.set(xlabel="Hours after forecast origin", ylabel="PV power (kW)",
           title=f"First eligible test forecast, fixed example: {first.forecast_origin.iloc[0]}")
    ax.grid(alpha=.18)
    ax.legend(fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(suite / "first_test_forecast.png", dpi=180)
    plt.close(fig)

    report = ["# M0 数据冻结与正式训练结果（2026-09-09）", "",
              "数据修复、版本冻结、best/last 保存、独立评估和三种子正式 M0 训练均已完成。", "",
              "## 数据与固定方案", "",
              f"冻结版本：`{manifest['dataset_version']}`。35040 个连续 15 分钟时点；完整历史 672 步（7 天），预测 96 步（24 小时），stride=1。", "",
              "|集合|目标所属日期（右端不含）|有效窗口|质量过滤排除|", "|---|---|---:|---:|"]
    for split, values in manifest["splits"].items():
        report.append(f"|{split}|{values['start_inclusive']} 至 {values['end_exclusive']}|{values['valid_windows_stride_1']:,}|{values['filtered_windows_stride_1']:,}|")
    report += ["", "PV 基准仅使用此前 31 天同时间槽，由 2022 年 12 月初始化；气象保留缺测和补值标记，长缺口进入的历史窗口被排除。标准化仅拟合训练期有效行。", "",
               "负荷系数在 2022 年拟合后固定，2023 年负荷约 62,268.39 kWh。标签是 PVGIS 基线与 HKUST 实测扰动构造的浮式建筑 PV 场景，负荷也为合成场景，均不等同于浮式建筑实测数据。", "",
               "单层 LSTM，隐藏维度 32，5 个历史输入；Adam lr=0.001、weight_decay=1e-6、batch=64、CPU 单线程/种子。最多 50 轮；验证 MSE 改善超过 0.01 kW² 重置耐心，连续 8 轮无此改善即早停。`best.pt` 始终取验证 MSE 的严格最小值。", "",
               "三个种子的超参数相同，全部训练结束后才重新加载各自 best 进行完整验证集/测试集评估。未按测试集挑选种子或调整参数。", "",
               "## 训练结果", "", "|Seed|完成轮数|最佳轮次|最佳验证 MSE（kW²）|停止原因|", "|---:|---:|---:|---:|---|"]
    for row in training:
        report.append(f"|{row['seed']}|{row['epochs']}|{row['best_epoch']}|{row['best_val_mse_kw2']:.4f}|{row['stop_reason']}|")
    report += ["", "## 独立测试集结果", "", "|方法|MAE（kW）|RMSE（kW）|MBE（kW）|", "|---|---:|---:|---:|"]
    for seed in protocol["seeds"]:
        row = table.query("split == 'test' and method == 'M0' and period == 'all' and seed == @seed").iloc[0]
        report.append(f"|M0，seed {seed}|{row.mae_kw:.4f}|{row.rmse_kw:.4f}|{row.mbe_kw:.4f}|")
    for method in ("M0", "persistence_24h"):
        values = table.query("split == 'test' and method == @method and period == 'all'")
        mean, std = values[["mae_kw", "rmse_kw", "mbe_kw"]].mean(), values[["mae_kw", "rmse_kw", "mbe_kw"]].std()
        cells = [f"{mean[k]:.4f} ± {std[k]:.4f}" if method == "M0" else f"{mean[k]:.4f}" for k in mean.index]
        report.append("|" + ("M0，三种子均值 ± 样本标准差" if method == "M0" else "昨日同期 persistence") + "|" + "|".join(cells) + "|")
    m0_rmse = table.query("split == 'test' and method == 'M0' and period == 'all'").rmse_kw.mean()
    baseline_rmse = table.query("split == 'test' and method == 'persistence_24h' and period == 'all'").rmse_kw.mean()
    reduction = (1 - m0_rmse / baseline_rmse) * 100
    m0_all = table.query("split == 'test' and method == 'M0' and period == 'all'")
    persistence_all = table.query("split == 'test' and method == 'persistence_24h' and period == 'all'")
    m0_near_zero = table.query("split == 'test' and method == 'M0' and period == 'non_daylight'")
    report += ["", f"M0 三种子平均 RMSE 相对昨日同期基线{'降低' if reduction >= 0 else '增加'} {abs(reduction):.2f}%。这只是本构造数据、此时间划分下的预测比较，尚未检验调度经济性或韧性。", "",
               f"**M0 并非所有误差指标都更优。** 全时段 MAE 为 {m0_all.mae_kw.mean():.4f} kW，昨日同期基线为 {persistence_all.mae_kw.mean():.4f} kW；全时段 MBE 为 +{m0_all.mbe_kw.mean():.4f} kW。真值 PV≤0.1 kW 子集的平均预测偏差为 +{m0_near_zero.mbe_kw.mean():.4f} kW，近零发电时段存在明显高估。验证集也出现同类偏差；本版本保留原始评估结果，未依据测试集重新训练或事后修正预测。", "",
               "|方法|真值 PV > 0.1 kW 子集 MAE（kW）|RMSE（kW）|MBE（kW）|", "|---|---:|---:|---:|"]
    for method in ("M0", "persistence_24h"):
        values = table.query("split == 'test' and method == @method and period == 'daylight'")
        report.append(f"|{method}（跨种子均值）|{values.mae_kw.mean():.4f}|{values.rmse_kw.mean():.4f}|{values.mbe_kw.mean():.4f}|")
    report += ["", "统计口径：每个有效预测起点的 96 步分别计分，重叠目标时刻会按不同预测重复计入；MBE=预测−真值。`daylight` 是真值 PV>0.1 kW 的事后报告子集，并非太阳高度定义，也不作为未来输入。种子标准差反映训练随机性，不是统计置信区间。", "",
               f"![训练与预测曲线]({suite / 'training_and_forecast.png'})", "",
               f"![固定首个测试预测样例]({suite / 'first_test_forecast.png'})", "",
               "上图预先固定使用首个有效测试预测起点，仅说明输出形态；不代表全部测试样本。", "",
               "## 验证与文件", "",
               "21 个因果预处理、数据集隔离、训练恢复与评估行为测试通过（分批执行）；真实数据入口 smoke 和完整验证评估通过。正式产物另核对了数据/源码 hash、best 选模、last 恢复状态、完整评估、checkpoint 对应关系以及从压缩预测明细重算指标。", "",
               f"- [完整协议和逐种子运行命令]({suite / 'protocol.json'})", f"- [产物核验]({suite / 'artifact_audit.json'})",
               f"- [训练摘要]({suite / 'training_summary.csv'})", f"- [全部误差指标]({suite / 'all_forecast_metrics.csv'})",
               f"- [冻结清单]({frozen / 'manifest.json'})", f"- [运行指南]({ROOT / 'deliverables/M0训练与独立评估使用说明.md'})", "",
               "每个 `seed_42/43/44` 目录包含 `best.pt`、`last.pt`、`metrics.csv`、`run_metadata.json`、`run.log`；对应 `_val` / `_test` 目录含完整独立评估和 `predictions.csv.gz`。训练源码快照和环境依赖列表保存在实验根目录。", "",
               "后续正式闭环评估仍需补齐闭环入口的数据 hash 校验，并统一 MPC 与执行环境的供电/失供约束；本轮没有用 M0 预测误差推断控制收益，也没有训练 M1/M2。", ""]
    destination = ROOT / "deliverables/M0正式训练结果_2026-09-09.md"
    destination.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"report": str(destination), "training": training, "test_rmse_mean": m0_rmse,
                      "persistence_test_rmse": baseline_rmse, "relative_rmse_reduction_percent": reduction,
                      "audit": "passed"}, ensure_ascii=False, indent=2))


def summarize(suite: Path, *, legacy: bool = False, allow_partial: bool = False) -> None:
    if legacy:
        if allow_partial:
            raise ValueError("--allow-partial is only supported for the current PV-only suite")
        summarize_legacy(suite)
        return
    command = [sys.executable, str(ROOT / "tools/summarize_pvonly_models.py"),
               "--suite", str(suite.resolve())]
    if allow_partial:
        command.append("--allow-partial")
    subprocess.run(command, cwd=ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True,
                        help="Completed experiment directory to summarize.")
    parser.add_argument("--legacy", action="store_true",
                        help="Explicitly use the historical 672-to-96, single-layer, three-seed audit.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Mark a current PV-only smoke/partial report explicitly.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summarize(args.suite, legacy=args.legacy, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
