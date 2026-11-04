# M1 预测:经济 = 1:2（2026-10-18）

HPC4 独立目录：`/data/user/jwei484/projects/m1_econ12_20261018`。
从原 1:1 实验第 3 轮 last.pt 开始，恢复 GRU、Adam 和随机状态；新阶段重置早停并重新做起点验证。训练目标为归一化预测 MSE + 2 × 归一化实际经济成本。
选模和早停使用 `economic_CNY_per_step`：先对每个验证场景求每步平均经济成本，再对场景等权平均；数值越低越好。不是场景总成本的平均值。经济成本包含购售电净支出、电池循环成本、弃光成本，不含失供惩罚。

本阶段最多再跑 7 轮（累计最多 10 轮）、patience=3、学习率 1e-4、seed=42、单线程、16 GB、Slurm 7 天时限，以及数据、求解精度等保持原设置。若未早停，新增 7 轮可能超过 7 天，需要从 last.pt 续跑；时间上限不等于自动完成全部轮次。

启动：
```bash
ssh hpc4
cd /data/user/jwei484/projects/m1_econ12_20261018
sbatch m1_hpc4_econ12.sbatch
```

结果自动保存在该项目 `experiments/M1/<时间戳>/`。`best.pt` 按经济指标选取；`best_val_loss` 是该最佳模型对应的混合 loss，`best_selection_value` 是经济选模指标。`last.pt` 为最近一次更新后的恢复点。选模指标纳入恢复一致性校验。

验证：HPC4 Python 环境中 25 项训练及求解默认设置测试通过（含经济改善但混合 loss 恶化时的选模、经济早停）；Slurm --test-only 通过。此前作业147347已提交但在初始化校验失败；未修改求解器误差接受策略。

阶段起点：`/data/user/jwei484/projects/m1_original_20261013/experiments/M1/20261013_221100_123720/last.pt`。本阶段 epoch 0 对应旧模型累计 epoch 3；checkpoint 的 cumulative_epoch 记录累计轮数。best.pt 若保留本阶段起点，指旧 M1 第 3 轮，不是原始 M0。恢复本阶段中断需使用同一配置的 --resume，不再同时指定 --stage-from。

2026-10-18 初始化修复：hardware 的 data_path / load_tiers_path 保留了旧默认数据路径，独立项目目录导致绝对路径前缀不同。stage_checkpoint 仅将两个项目内默认路径按各自项目根目录归一化；相对路径、物理参数、实际训练数据内容哈希、运行环境及非入口代码继续严格比对。旧检查点不改写。7项阶段恢复和保护测试通过。

修复后完整预检通过：tmp/m1_preflight/20261018_180905_705997/status.json。已用原脚本重新提交作业147349，2026-10-18 18:10核验RUNNING（arm1-2）；此前147347失败记录保留。

18:13日志确认恢复原epoch 3，已进入新阶段epoch 0验证（4/94）。正式目录：experiments/M1/20261018_181214_463943。
