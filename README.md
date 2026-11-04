# MPhil Floating Building — 光伏预测与储能滚动调度

面向 HKUST(GZ) 浮式建筑能源系统，研究光伏预测如何影响储能与电网调度，并在并网、停电预警及低光伏场景下比较经济性和供电可靠性。

**更新日期：2026-11-04。** 当前代码覆盖数据准备、PV-only 预测、M0 场景评估、M1/M2 决策导向训练，以及 RBC / MILP / MPC 基线。研究采用的构造数据和情景不等于浮式建筑现场运行记录；“零碳”是研究背景，当前代码不构成已实现碳中和的证据。

## 当前研究主线

```text
PVGIS 基线 / 清水湾实测参照 + 负荷与电价情景
  → 数据清洗、时间对齐、训练期归一化与冻结
  → 光伏预测
  → MPC 优化储能、电网与分级负荷分配
  → 执行当前时段动作、更新 SOC、滚动至下一时段
  → 分别评估预测误差、成本、缺电与求解器状态
```

预测模型输出可用 PV，优化器决定其使用和弃电，以及电池充放电、电网交换和未供负荷。系统功率平衡与电池 SOC 动态分别建模。

| 方法 | 训练或控制目标 | 代码入口 |
|---|---|---|
| M0 | 预测误差监督训练，接入场景调度评估 | `model_variants/`、`tools/run_m0_simulation.py`、`tools/run_m0_scenarios.py` |
| M1 | 预测 MSE + 经 MPC 和执行层计算的经济成本；不含显式 ENS 惩罚 | `tools/train_m1.py` |
| M2 | 可配置预测监督 + 经济成本 + 分级实际未供能量 ENS 惩罚 | `tools/train_m2.py` |
| 工程控制基线 | 规则控制、固定电池 MILP 与滚动 MPC | `dispatch_rbc_15min.py`、`rbc_extreme_events_fixed_battery_15min.py`、`milp_e4*.py`、`mpc_e5*.py` |

M1/M2 的可微训练与正式 MILP 评估需区分优化形式和求解器。已有训练入口与测试，不等于 M1/M2 已优于 M0；正式结论须使用一致的场景、硬件、初始 SOC、数据切分和执行规则比较。

## 数据与场景

- **PVGIS 与建筑物理基线**：`pv_generation.py` 保留多朝向 BIPV 发电计算；`load_generation.py` 保留按设计年电量校准的合成负荷。年总量匹配是构造约束，不能作为实测验证。
- **15 分钟及多年准备**：`prepare_m0_data.py`、`prepare_pv_only_multiyear.py`、`prepare_pv_only_multihorizon.py` 和 `tools/prepare_annual_*.py` 提供相应数据流程。
- **构造 PV 标签**：部分冻结数据由建筑 PVGIS 基线乘清水湾实测扰动得到，不是浮式建筑实测功率；跨地点组合须标注为合成情景。
- **预测任务**：已有双层 LSTM/GRU 的 96→1 旧任务，以及 96/192/288 步历史→未来 96 步的多步配置；15 分钟间隔下分别对应 1/2/3 天历史与未来 24 小时。不同任务不得直接混比。
- **情景配置**：S0–S5 的版本定义见 `configs/m0_s0_s5_v2.json` 与 `tools/m0_scenarios.py`；另有 S1–S4 和不同电池配置。配置名称不能替代对事件、容量和时序的检查。

原始数据、冻结输入、训练权重和完整运行结果保留本地。新检出环境需要取得对应数据及 manifest，并检查脚本中的本机/HPC 路径。冻结数据的身份校验规则见 [模型说明](model_variants/README.md)。

## 环境与快速检查

早期年度能量分析使用根目录 `requirements.txt`。预测与 MPC 集成使用独立的兼容环境，建议 Python 3.11，遵循 [集成环境说明](external/ol-ems-hkust/ENVIRONMENT.md)：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "pip==23.3.2" "setuptools==59.5.0" "wheel==0.42.0"
python -m pip install -r external/ol-ems-hkust/requirements.txt
python -m pip install --no-build-isolation --no-deps -e external/python-microgrid-hkust
python -m pytest tests -q
```

`pymgrid` 由仓库中的 `python-microgrid-hkust` 提供，勿替换成同名的其他 PyPI 包。求解器和许可证也须在本机配置。正式场景评估的 MILP 求解器可通过 `MPC_MILP_SOLVER=gurobi|highs|auto` 选择，以当前代码的默认值和可用性检查为准。

2026-11-04 本地全量验证：**263 项测试通过**，有 14 条依赖警告；HiGHS 回归测试显式选择对应求解器。测试通过表示这些检查通过，不代表新跑完年度实验或现场验证。

## 运行入口

先查看参数，再准备匹配的冻结输入、checkpoint 与配置：

```bash
python tools/run_m0_scenarios.py --help
python tools/train_m1.py --help
python tools/train_m2.py --help
```

已有多步预测冻结输入时，可先检查训练命令：

```bash
python tools/run_pv_multihorizon.py --dry-run
```

M1/M2 支持 `--mode preflight`、`smoke` 和 `train`。先通过输入预检及短步验证，再开展正式训练；预检也需要相应本地数据。输出分别写入 `experiments/M1/` 与 `experiments/M2/`，续训与阶段初始化规则以脚本参数为准。

早期年度能量分析仍可在输入齐备后运行：

```bash
python pv_generation.py
python load_generation.py
python pv_load_daily_compare.py
```

该流程只分析未调度的 PV 与负荷匹配。年度净能量盈余不代表每小时供电充足，也不等于停电时具备对应续航能力。

## 项目结构

```text
Mphil_floating_building/
├── model_variants/                 # PV 预测、数据加载、可微 MPC 与执行层
├── tools/                          # 年度实验、场景评估、M1/M2、天气分析与 HPC 工具
├── configs/                        # 电池、事件、年度与训练配置
├── tests/                          # 时间边界、预测、训练、调度与求解回归测试
├── external/
│   ├── ol-ems-hkust/               # 项目维护的 EMS 集成副本
│   ├── python-microgrid-hkust/     # 项目维护的 microgrid 副本
│   └── forecast_baselines/        # DLinear/PatchTST 及来源、许可证
├── prepare_*.py                    # 数据准备入口
├── pv_generation.py               # 原始 BIPV 物理基线
├── load_generation*.py             # 合成负荷
├── milp_e4*.py / mpc_e5*.py        # 历史调度基线
├── output/                        # 本地生成数据与冻结输入
└── experiments/                   # 本地训练、评估与 checkpoint
```

只读上游检出与项目维护副本的职责见 [external/README.md](external/README.md)。本次提交包含必要源码、配置、测试与上游许可证；虚拟环境、报告、演示材料及批量生成结果留在本地。

## 结果解释与下一步

正式对比至少分别报告预测误差、经济成本、分级 ENS/供电情况、SOC 和求解器诊断。未来实际 PV 只能用于明确标注的 perfect-forecast 对照，不能混入可部署策略。

50 kW / 320 kWh 等配置文件属于实验设定，不能直接宣称已验证 24 小时关键负荷保障。早期约 79 kWp、60,461 kWh 年负荷等物理/设计口径结果也不能直接套用于后续不同电池和场景版本。当前主要工作是固定条件下比较 M0、M1、M2 与完美预测对照，完善闭环验证、梯度及求解器一致性检查。

## 文档与来源

- [2026-11-04 源码更新及验证说明](SOURCE_UPDATE_20261104.md)
- [预测模型、数据协议与多步入口](model_variants/README.md)
- [集成环境说明](external/ol-ems-hkust/ENVIRONMENT.md)
- [工作日志](WORKLOG.md)：历史记录，具体结果须结合对应运行配置和输出核验。
- [早期研究计划](research_plan.md)、[数据申请说明](data_request.md)：保留早期背景和设计资料口径。
- [预测模型上游来源与许可证](external/forecast_baselines/UPSTREAM.md)
- [独立光伏预测仓库](https://github.com/wjulien888-tech/PV_prediction)

外部代码和数据遵循各自许可，使用前核对对应来源文件。
