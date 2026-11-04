# 新增：1/2/3 天历史 → 未来 24 小时（96 步）

六配置已准备，双层 LSTM/GRU 各对应 L=96/192/288、H=96，PV-only。
用户在项目根目录激活 `external/ol-ems-hkust/.venv` 后，运行：

```bash
python -u tools/run_pv_multihorizon.py --dry-run
python -u tools/run_pv_multihorizon.py --jobs 6
```

第二条才会开始训练；第一条只检查冻结文件和打印命令。新套件显式使用
`--fc_hidden_size 64`，输出头为 64→64→96；训练结束自动汇总验证集，测试单独评估。
完整说明见 [终端训练指南](../deliverables/PV双层模型_1至3天预测24小时_终端训练指南_2026-09-21.md)。
新冻结数据为 `output/frozen/pv_only_2021_2023_multihorizon96_v1_20260921`。

下文为保留的旧 96→1 使用说明；单独运行模型脚本而不指定新参数时仍使用旧默认值。

# 双层 LSTM / GRU：96步历史 → 下一个15分钟

M0正式入口 `external/ol-ems-hkust/main.py` 已直接接入本目录的双层网络，通过 `--model_type lstm`（默认）或 `--model_type gru` 选择。任务固定为 **96个15分钟历史PV点 → 紧接着的1个15分钟PV点**。本目录两个独立脚本仍可运行，和正式入口使用同一模型类、加载器及训练/评估实现。

| 项目 | 双层 LSTM | 双层 GRU |
|---|---|---|
| 入口 | `lstm_2layer_m0.py` | `gru_2layer_m0.py` |
| 模型类 | `PVLSTMController` | `PVGRUController` |
| 输入 | `[B, 96, 1]`：过去24小时的PV，15分钟间隔 | 相同 |
| 循环网络 | 2层 LSTM，hidden=64，单向 | 2层 GRU，hidden=64，单向 |
| 层间 Dropout | 0.2 | 0.2 |
| 预测头 | 最后时间步 → Linear(64,5) → ReLU → Linear(5,1) → Sigmoid | 相同 |
| 输出 | 按训练期 PV 范围反归一化，`[B,1,1]`，单位 kW | 相同 |
| 可训练参数 | 50,763 | 38,155 |
| 默认输出根目录 | `experiments/m0_lstm_2layer_pvonly_2021_2023_96to1/` | `experiments/m0_gru_2layer_pvonly_2021_2023_96to1/` |

输入仅为历史 PV，不加入温度、相对湿度、辐照度或降雨。造窗严格为 `X[t-96:t] → PV[t]`：最后一个输入点为 t-15分钟，标签为紧接着的 t 时段，输出只有1个点。M0 训练只优化预测 MSE。

例如：输入昨日12:00至今日11:45的96点，预测今日12:00这个15分钟时段。15分钟时间戳按原冻结数据的区间约定解释。

## 与参考脚本的适配

参考文件位于 `/Users/julia/Transfer/0000PV_prediction/extracted_scripts/`，分别为 `lstm_24h_to_1h.py` 和 `gru_24h_to_1h.py`。它们原来是小时级、历史24点单变量输入、预测下一小时，输出单位 W；本副本移植其双层循环结构与全连接预测头，时间窗口改为用户要求的15分钟级96→1、仅PV输入、单位kW。

多年数据按目标时间划分：2021年7月1日—2023年8月31日训练，2023年9—10月验证，2023年11—12月独立测试。只在 `split=train AND history_valid=1` 的PV行拟合 scaler；验证 MSE 选择 best 并早停；保存完整 best/last；测试不参与选模。没有移植参考脚本用测试集早停的逻辑。各样本要求此前96个历史点全部有效、目标点有效，保留完整时间轴并排除跨缺测区间的窗口；验证/测试允许使用目标之前已经观测到的历史PV。

默认 Adam lr=0.001、weight_decay=1e-6、batch=64、最多50轮、patience=8、min_delta=0.01 kW²、seed=42、CPU单线程。模型参数可通过 `--hidden_dim`、`--num_layers`、`--dropout`、`--fc_hidden_size` 显式调整；上表对应默认值。

默认冻结数据目录为 `output/frozen/pv_only_2021_2023_96to1_v1_20260920`，直接读取 `pv_timeseries.csv` 和 `manifest.json`。核心CSV列为 `datetime,pv_kW,history_valid,pv_target_valid,split`；`pv_kW`是各年建筑PVGIS基线乘HKUST实测扰动得到的构造标签，属于模拟场景，不是浮式建筑实测功率。加载器不读取或补造负荷、气象、电价列。冻结目录附带的训练数组供独立检查和其他模型使用，两个入口从CSV按相同规则重新造窗。

`--data_format legacy_dispatch` 可显式使用旧2023年五特征加载器；默认仍为多年PV-only。旧模式需要 `dispatch.csv` 与 `load_tiers.csv`，新模式不需要负荷文件且拒绝 `--load_tiers_path`。两类数据的输入维度、scaler与checkpoint不可混用。参考项目权重、旧单层 M0 权重和两个新模型之间也不能混用；加载时会核对循环类型、网络结构、分割及数据身份。

## 运行

在 `/Users/julia/Transfer/0000Mphil_floating_building` 下使用当前 M0 的 Python 3.11 环境。模型脚本依赖本项目的加载器、训练器和 MPC 模块，不是可脱离项目单独复制使用的文件。

正式训练一个种子：

```bash
external/ol-ems-hkust/.venv/bin/python external/ol-ems-hkust/main.py --mode train --model_type lstm --seed 42 --experiment_name seed_42
external/ol-ems-hkust/.venv/bin/python external/ol-ems-hkust/main.py --mode train --model_type gru --seed 42 --experiment_name seed_42
```

其余种子用 `--seed 43 --experiment_name seed_43` 和 `--seed 44 --experiment_name seed_44`。训练完成后，分别重载各自的 best，执行完整独立评估：

```bash
external/ol-ems-hkust/.venv/bin/python external/ol-ems-hkust/main.py --mode evaluate --model_type lstm --load_model experiments/m0_lstm_2layer_pvonly_2021_2023_96to1/seed_42/best.pt --eval_split test --experiment_name seed_42_test
external/ol-ems-hkust/.venv/bin/python external/ol-ems-hkust/main.py --mode evaluate --model_type gru --load_model experiments/m0_gru_2layer_pvonly_2021_2023_96to1/seed_42/best.pt --eval_split test --experiment_name seed_42_test
```

验证集独立评估使用 `--eval_split val` 和新的实验名。各输出目录拒绝覆盖，重新开始一轮训练须更换实验名。续训示例：

```bash
external/ol-ems-hkust/.venv/bin/python external/ol-ems-hkust/main.py --mode train --model_type gru --resume experiments/m0_gru_2layer_pvonly_2021_2023_96to1/seed_42/last.pt --max_epochs 60
```

续训要求其余训练参数、代码、运行环境和数据与该轮记录一致，且尚未触发早停；如最初使用自定义参数，续训时必须重传对应参数。

整套正式实验使用 `external/ol-ems-hkust/.venv/bin/python tools/run_formal_m0.py --output experiments/<新的实验目录>`，依次完成两模型的训练、独立验证/测试和汇总；当前套件固定seed42。历史2026-09-20两个best.pt可通过上述正式入口直接加载评估，网络状态键和数据身份规则没有变。由于入口代码指纹已变化，旧last.pt不能作为代码相同的逐位续训；本次已完成运行也已触发早停。

训练数据与权重保留在本地 `output/frozen/`、`experiments/`，不随本次代码提交打包。新检出环境需要先取得完整冻结目录（包括manifest列出的所有文件），加载器会逐文件验证SHA-256；Python依赖按 `external/ol-ems-hkust/ENVIRONMENT.md` 配置。

## 结果与可追溯性

- 训练目录包含 `best.pt`、`last.pt`、逐轮 `metrics.csv`、`run_metadata.json`、`run.log`。
- 评估目录包含 MAE、RMSE、MBE、MSE、**R²**，全时段/实际PV>0.1 kW/近零PV分组指标、逐预测步指标和压缩预测明细。当前每个预测起点仅有一个目标；R²按组汇总计算，恒定真值组记为 null。评估同时输出昨日同期 `PV[t-96]` 和上一时刻 `PV[t-1]` 两种基线，二者不会混淆。
- `_m0_variant_runner.py` 是正式M0入口和两个独立入口共用的预测运行器；`_m0_variant_evaluation.py` 负责含R²的评估。`_pv_only_dataset.py` 是多年单变量预测加载器；`_m0_variant_dataset.py` 保留旧五特征接口，共同复用训练实现。
- 模型类型、Dropout、预测头宽度、历史长度、特征顺序、输入和输出维度都进入 checkpoint 架构记录；数据CSV与manifest的SHA-256、任务和分割一起记录。新manifest须为 `schema_version=2`、`artifact_type=pv_only_forecast_96to1`，任务参数和时间分割必须匹配，所有冻结输出文件按尺寸与SHA-256校验。原 M0 代码和本目录 Python 文件一并记录 SHA-256，支持识别不匹配的恢复操作。
- 正式入口已更新为两模型选择；所有运行保留当时的代码指纹，既有冻结数据与已完成实验文件不覆盖。

## 与原24小时预测及MPC的区别

当前M0提供 `train` 和 `evaluate`。单步预测不直接接入原96步MPC；正式入口显式 `--mode closed-loop` 保留历史多步MPC参数和模型兼容路径，并明确拒绝96→1权重。旧单层Controller只服务该历史路径，不再是M0默认预测网络。若以后让新网络参与MPC，须另行明确多步预测或递推方案。

旧M0的R²≈0.7123属于672→96的24小时预测任务，不能作为本96→1任务的同口径基线。旧单层模型、旧权重和既有结果均保留。

历史流程检查文件不代表正式精度：`validation_smoke_672to96_20260920.json`和`validation_smoke_96to1_20260920.json`为此前五特征版本，不能作为多年PV-only结果。

本次多年PV-only检查记录为`validation_smoke_pvonly_2021_2023_20260920.json`：25项测试通过；两个模型各完成1个训练批次、1个验证批次，并重新加载best权重评估2个验证批次。未进行正式训练或测试集评估。

上述流程验证之后，已在 `experiments/pvonly_lstm_gru_96to1_20260920_230516/` 完成seed42正式训练：LSTM最佳第40轮、GRU最佳第14轮，测试R²分别0.977240、0.974578。该结果仍为构造PV、单种子96→1任务，完整证据见实验目录中的报告与逐点预测。
