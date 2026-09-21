#!/usr/bin/env python3
"""Summarise an already trained PV-only LSTM/GRU suite; never selects a seed.

Run with the project Python environment and --suite EXPERIMENT_DIRECTORY.
Full seed-42 training and complete val/test evaluations are required by default.
--allow-partial is exclusively for checking smoke/unfinished reporting artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_NAMES = {"persistence_24h": "昨日同期PV", "persistence_last": "上一时刻PV"}
PERIOD_NAMES = {"all": "全时段", "daylight": "真实PV > 0.1 kW"}
METRIC_COLUMNS = ["count", "r2", "rmse_kw", "mae_kw", "mbe_kw", "mse_kw2"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_training(suite: Path) -> dict:
    runs = {}
    for path in sorted(suite.glob("*/run_metadata.json")):
        metadata = read_json(path)
        arguments = metadata.get("arguments", {})
        if arguments.get("mode") != "train":
            continue
        model = metadata.get("model_architecture", {}).get("recurrent_type")
        if model not in ("LSTM", "GRU"):
            continue
        require(model not in runs, f"Multiple {model} runs found; pass a suite with exactly one seed-42 run per model")
        require(metadata.get("seed") == 42, f"{path}: only predeclared seed 42 is supported")
        config = metadata["config"]
        require(config["history_features"] == ["pv"] and config["lookback_steps"] == 96
                and config["prediction_horizon"] == 1, f"{path}: not the PV-only 96-to-1 task")
        require(metadata.get("status") == "completed", f"{path}: training has not completed")
        history = pd.read_csv(path.parent / "metrics.csv")
        require(len(history) > 0, f"{path}: empty training metrics")
        require(int(history.epoch.max()) == metadata["epochs_completed"], f"{path}: epoch metadata mismatch")
        best_row = history.loc[history.val_loss_mse.idxmin()]
        require(int(best_row.epoch) == metadata["best_epoch"], f"{path}: best epoch was not the validation minimum")
        require(np.isclose(best_row.val_loss_mse, metadata["best_val_loss_mse"], rtol=1e-10),
                f"{path}: best validation loss mismatch")
        partial = (metadata.get("model_status") == "smoke_trained" or bool(arguments.get("fast_dev_run"))
                   or arguments.get("limit_train_batches", 1.0) != 1.0
                   or arguments.get("limit_val_batches", 1.0) != 1.0)
        if not partial:
            require((history.train_samples == metadata["train_samples_available"]).all(),
                    f"{path}: not all training samples were used")
            require((history.val_samples == metadata["val_samples_available"]).all(),
                    f"{path}: not all validation samples were used")
        runs[model] = {"path": path.parent, "metadata": metadata, "history": history,
                       "checkpoint_sha256": sha256(path.parent / "best.pt"), "partial": partial}
    require(set(runs) == {"LSTM", "GRU"}, "Suite must contain completed LSTM and GRU training runs")
    require(runs["LSTM"]["metadata"]["dataset_identity"] == runs["GRU"]["metadata"]["dataset_identity"],
            "The models used different frozen datasets")
    return {model: runs[model] for model in ("LSTM", "GRU")}


def load_evaluations(suite: Path, runs: dict, allow_partial: bool) -> tuple[pd.DataFrame, dict, bool]:
    evaluations, rows, baselines = {}, [], {}
    partial = any(run["partial"] for run in runs.values())
    for path in sorted(suite.glob("*/evaluation_summary.json")):
        summary = read_json(path)
        model = summary.get("model_architecture", {}).get("recurrent_type")
        split = summary.get("split")
        if model not in runs or split not in ("val", "test"):
            continue
        key = model, split
        require(key not in evaluations, f"Duplicate {model}/{split} evaluation in suite")
        run = runs[model]
        require(summary.get("status") == "completed", f"{path}: evaluation incomplete")
        require(summary.get("dataset_identity") == run["metadata"]["dataset_identity"], f"{path}: dataset mismatch")
        require(summary.get("checkpoint_sha256") == run["checkpoint_sha256"], f"{path}: not the current best.pt")
        require(summary.get("checkpoint_epoch") == run["metadata"]["best_epoch"], f"{path}: checkpoint epoch mismatch")
        require(summary.get("checkpoint_seed") == 42, f"{path}: checkpoint seed mismatch")
        require(summary.get("horizon_steps") == 1, f"{path}: expected one target per origin")
        require(summary.get("arguments", {}).get("daylight_threshold_kw") == 0.1, f"{path}: daylight threshold mismatch")
        expected = run["metadata"]["dataset_identity"]["splits"][split]["valid_windows"]
        require(summary["available_origins"] == expected, f"{path}: split size differs from frozen data")
        require(summary["complete_split"] == (summary["forecast_origins"] == expected),
                f"{path}: incorrect complete-split flag")
        partial = partial or not summary["complete_split"]
        metrics = pd.read_csv(path.parent / "forecast_metrics.csv")
        json_metrics = pd.DataFrame(summary["metrics"])
        require(not metrics.duplicated(["method", "period"]).any(), f"{path}: duplicate metric groups")
        # The CSV and JSON are separate evaluator artifacts; check they agree.
        left = metrics.set_index(["method", "period"]).sort_index()
        right = json_metrics.set_index(["method", "period"]).sort_index()
        require(left.index.equals(right.index) and np.allclose(left[METRIC_COLUMNS].to_numpy(dtype=float),
                right[METRIC_COLUMNS].to_numpy(dtype=float),
                rtol=1e-10, atol=1e-10, equal_nan=True), f"{path}: CSV/JSON metric disagreement")
        require(int(left.loc[("M0", "all"), "count"]) == summary["forecast_origins"],
                f"{path}: metric count differs from evaluated origins")
        for method in ("M0", "persistence_24h", "persistence_last"):
            for period in ("all", "daylight"):
                metric = left.loc[(method, period)]
                require(int(metric["count"]) == int(left.loc[("M0", period), "count"]),
                        f"{path}: unequal group sizes across methods")
                label = model if method == "M0" else METHOD_NAMES[method]
                row = {"split": split, "period": period, "method": label, "seed": 42 if method == "M0" else None,
                       **{column: metric[column] for column in METRIC_COLUMNS},
                       "complete_split": summary["complete_split"],
                       "evaluation_path": str(path.parent), "source_training_run": str(run["path"])}
                row["count"] = int(row["count"])
                if method == "M0":
                    rows.append(row)
                else:
                    baseline_key = split, period, method
                    if baseline_key in baselines:
                        previous = baselines[baseline_key]
                        require(np.allclose([previous[c] for c in METRIC_COLUMNS],
                                            [row[c] for c in METRIC_COLUMNS],
                                            rtol=1e-9, atol=1e-9, equal_nan=True),
                                f"{path}: baselines differ between models")
                    else:
                        baselines[baseline_key] = row
        evaluations[key] = summary
    missing = {(model, split) for model in runs for split in ("val", "test")} - set(evaluations)
    partial = partial or bool(missing)
    require(not partial or allow_partial,
            f"Formal report requires full training and all four complete evaluations; missing={sorted(missing)}. "
            "Use --allow-partial only to validate smoke reporting.")
    require(len(rows) > 0, "No compatible evaluation results found")
    result = pd.DataFrame(rows + list(baselines.values()))
    result["report_status"] = "smoke_or_partial_not_formal" if partial else "formal_single_seed"
    result["rmse_reduction_vs_last_pct"] = np.nan
    result["rmse_reduction_vs_24h_pct"] = np.nan
    for index, row in result.iterrows():
        for method, column in (("persistence_last", "rmse_reduction_vs_last_pct"),
                               ("persistence_24h", "rmse_reduction_vs_24h_pct")):
            base = baselines[(row["split"], row["period"], method)]["rmse_kw"]
            if pd.notna(base) and base > 0:
                result.loc[index, column] = 100 * (1 - row["rmse_kw"] / base)
    result["_split_order"] = result.split.map({"val": 0, "test": 1})
    result["_period_order"] = result.period.map({"all": 0, "daylight": 1})
    result["_method_order"] = result.method.map({"LSTM": 0, "GRU": 1, "昨日同期PV": 2, "上一时刻PV": 3})
    result = result.sort_values(["_split_order", "_period_order", "_method_order"]).drop(
        columns=["_split_order", "_period_order", "_method_order"]).reset_index(drop=True)
    return result, evaluations, partial


def plot_history(suite: Path, runs: dict, partial: bool) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for axis, model in zip(axes, ("LSTM", "GRU")):
        run, history = runs[model], runs[model]["history"]
        axis.plot(history.epoch, history.train_loss_mse, label="Train", color="#2463a7", linewidth=1.8,
                  marker="o", markersize=3)
        axis.plot(history.epoch, history.val_loss_mse, label="Validation", color="#d47722", linewidth=1.8,
                  marker="o", markersize=3)
        axis.axvline(run["metadata"]["best_epoch"], color="#6c7785", linestyle=":", label="Selected best epoch")
        axis.set(title=f"{model} | seed 42", xlabel="Epoch", ylabel="MSE (kW²)")
        if (history[["train_loss_mse", "val_loss_mse"]].to_numpy() > 0).all():
            axis.set_yscale("log")
            axis.set_ylabel("MSE (kW², log scale)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("SMOKE / PARTIAL — not model performance" if partial else
                 "PV-only 96-to-1 | constructed PV | validation-selected checkpoints", fontsize=11)
    destination = suite / "training_loss_curves.png"
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)
    return destination


def fmt(value: object, digits: int = 4) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def write_report(suite: Path, runs: dict, result: pd.DataFrame, evaluations: dict,
                 partial: bool, plot_path: Path) -> None:
    title = "流程验证结果（SMOKE / PARTIAL，不能作为正式模型表现）" if partial else "LSTM / GRU 正式训练结果（seed 42）"
    lines = [f"# {title}", "", "输入：过去96个15分钟PV点；目标：下一个15分钟PV。输入特征仅PV。", "",
             "数据为 PVGIS 建筑发电基线 × HKUST 实测PV因果扰动构造的浮式建筑PV；不是浮式建筑实测发电数据。",
             "两个模型使用同一冻结数据、时间划分和仅训练期拟合的归一化参数。每个模型只运行预先固定的 seed 42，未进行多种子统计，也未根据测试集选择种子。", ""]
    if partial:
        lines += ["**本文件只验证汇总工具。训练使用了小批次或评估尚不完整；下面的数值不能用于评价正式拟合度或比较模型优劣。**", ""]
    lines += ["**训练配置及选取结果**", "", "| 模型 | 循环层/隐藏单元 | Dropout | 参数量 | 已完成轮次 | 最佳轮次 | 最佳验证MSE (kW²) | 停止原因 |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for model, run in runs.items():
        meta = run["metadata"]
        arch = meta["model_architecture"]
        lines.append(f"| {model} | {arch['num_layers']}层 / {arch['hidden_dim']} | {arch['dropout']} | "
                     f"{meta['trainable_parameters']:,} | {meta['epochs_completed']} | {meta['best_epoch']} | "
                     f"{meta['best_val_loss_mse']:.4f} | {meta['stop_reason']} |")
    lines.append("")
    for model, run in runs.items():
        args = run["metadata"]["arguments"]
        lines.append(f"- {model}：Adam；学习率 {args['learning_rate']}；weight decay {args['weight_decay']}；"
                     f"训练/验证 batch size {args['train_batch_size']}/{args['val_batch_size']}；最多 {args['max_epochs']} 轮；"
                     f"patience={args['patience']}；min_delta={args['min_delta']} kW²；"
                     f"输出头 {args['hidden_dim']}→{args['fc_hidden_size']}→1，sigmoid 映射至训练PV范围；设备 {args['device']}。")
    lines += ["", "最佳权重仅按验证集MSE的严格最小值保存；min_delta仅用于早停耐心计数。独立评估重新加载 best.pt，不更新参数或利用测试集校准输出。", "",
              "**固定数据划分**", "", "| 数据集 | 有效目标起点 | 有效目标终点 | 样本数 |", "|---|---|---|---:|"]
    for split, item in runs["LSTM"]["metadata"]["dataset_identity"]["splits"].items():
        lines.append(f"| {split} | {item['first_target']} | {item['last_target']} | {item['valid_windows']:,} |")
    lines += ["", "**独立预测评估**", "", "R² = 1 − SSE/SST，按所列分组独立计算；常量目标或样本不足时为空。RMSE、MAE、MBE单位为kW，MBE = 预测 − 真值，正值表示高估。", "",
              "真实PV > 0.1 kW为事后报告筛选条件，不作为模型输入；全时段包含夜间。昨日同期基线使用 PV[t−96]，上一时刻基线使用 PV[t−1]，两者均只使用预测前数据。", ""]
    for split in ("val", "test"):
        for period in ("all", "daylight"):
            subset = result[(result.split == split) & (result.period == period)]
            if subset.empty:
                continue
            lines += [f"**{split} · {PERIOD_NAMES[period]}**", "", "| 方法 | 样本数 | R² | RMSE | MAE | MBE |",
                      "|---|---:|---:|---:|---:|---:|"]
            for row in subset.to_dict("records"):
                lines.append(f"| {row['method']} | {row['count']:,} | {fmt(row['r2'])} | {fmt(row['rmse_kw'])} | "
                             f"{fmt(row['mae_kw'])} | {fmt(row['mbe_kw'])} |")
            lines.append("")
    if not partial:
        for model in ("LSTM", "GRU"):
            item = result[(result.split == "test") & (result.period == "all") & (result.method == model)].iloc[0]
            lines.append(f"- {model} 测试集全时段RMSE相对上一时刻基线降低 {fmt(item.rmse_reduction_vs_last_pct, 2)}%，"
                         f"相对昨日同期基线降低 {fmt(item.rmse_reduction_vs_24h_pct, 2)}%（负数表示变差）。")
        lines += ["", "这些结果描述本次单个种子的预测表现。与旧M0的输入特征、预测范围不同，不能据此单独归因于训练年份增加，也不能推断实际调度降本效果。", ""]
    lines += [f"![训练与验证损失曲线]({plot_path})", "", "**可追溯文件**", "",
              f"- [完整指标表]({suite / 'result_comparison.csv'})",
              f"- [训练损失曲线]({plot_path})"]
    for model, run in runs.items():
        lines.append(f"- {model}：[最佳权重]({run['path'] / 'best.pt'})；[训练记录]({run['path'] / 'metrics.csv'})；"
                     f"[配置及数据校验]({run['path'] / 'run_metadata.json'})。")
    for (model, split), summary in sorted(evaluations.items()):
        row = result[(result.method == model) & (result.split == split)].iloc[0]
        lines.append(f"- {model} {split}：[逐点预测]({Path(row.evaluation_path) / 'predictions.csv.gz'})；"
                     f"评估 {summary['forecast_origins']:,}/{summary['available_origins']:,} 个样本。")
    lines.append("")
    (suite / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Produce an explicitly marked smoke/partial report")
    args = parser.parse_args()
    suite = args.suite.resolve()
    runs = load_training(suite)
    result, evaluations, partial = load_evaluations(suite, runs, args.allow_partial)
    plot = plot_history(suite, runs, partial)
    result.to_csv(suite / "result_comparison.csv", index=False, float_format="%.10g")
    write_report(suite, runs, result, evaluations, partial, plot)
    print(json.dumps({"suite": str(suite), "report_status": "smoke_or_partial_not_formal" if partial else "formal_single_seed",
                      "comparison_rows": len(result), "evaluations": len(evaluations),
                      "outputs": ["result_comparison.csv", "result_summary.md", plot.name]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
