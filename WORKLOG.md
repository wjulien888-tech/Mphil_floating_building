# 工作日志

本文件用于记录项目每次工作内容，包括目标、思路、修改的文件以及生成的文件。

## 2026-04-21 - 按 PDF 要求整理目录结构

### 目标

按 PDF 中约定的项目结构整理当前工作区。

### 已完成内容

- 创建 `data/`
- 创建 `output/`
- 创建 `plots/`
- 将项目根目录下全部 `pvgis_*.csv` 移动到 `data/`
- 保留 `Cursor_workflow.pdf` 在项目根目录
- 新增 `path_convention_note.txt`

### 修改或移动的文件

- `data/pvgis_south_45.csv`
- `data/pvgis_east_45.csv`
- `data/pvgis_west_45.csv`
- `data/pvgis_north_45.csv`
- `data/pvgis_deck.csv`
- `data/pvgis_top.csv`

### 生成的文件

- `path_convention_note.txt`

### 工作说明

- PDF 约定项目应包含 `data/`、`output/` 和 `plots/` 三个目录。
- `pvgis_top.csv` 没有删除，而是保留在 `data/` 中作为备查数据。

## 2026-04-21 - 光伏发电脚本

### 目标

实现 PDF 中定义的光伏发电处理流程，并验证生成的 `8760` 小时时序结果。

### 已完成内容

- 新建 `pv_generation.py`
- 实现 PVGIS CSV 解析，包括跳过元数据和清理文件末尾说明行
- 实现按面积、效率、温度修正和水面微气候修正计算发电功率
- 导出 `output/pv_generation_8760.csv`
- 生成日尺度和月尺度发电图
- 校验 `pvgis_deck.csv` 与 `pvgis_top.csv` 内容一致，并避免对相同水平面辐照重复计入

### 修改的文件

- `pv_generation.py`

### 生成的文件

- `output/pv_generation_8760.csv`
- `plots/pv_daily_kwh.png`
- `plots/pv_monthly_kwh.png`

### 工作说明

- 按物理公式直接计算得到的年光伏发电量约为 `110,888 kWh`。
- 该结果明显高于 PDF 中给出的 `55,580 kWh`。
- 结合设计院资料交叉分析后，较低的那个数更像是设计院汇报口径，或由能源自持率反推得到的数值，而不是同一口径下的光伏毛发电量。
- 当前决定：后续工作以物理计算得到的光伏结果作为基准。

## 2026-04-21 - 依赖文件

### 目标

为后续 Python 脚本添加统一的依赖清单。

### 已完成内容

- 新建 `requirements.txt`

### 生成的文件

- `requirements.txt`

### 工作说明

- 包含 PDF 中提到的库，并额外加入 `tqdm`，因为后续流程说明里明确提到了进度条需求。

## 2026-04-21 - PDF 转 Markdown 与日志机制

### 目标

将 PDF 转成便于阅读的文本版，并建立长期维护的工作日志文件。

### 已完成内容

- 将 `Cursor_workflow.pdf` 转换整理为 `Cursor_workflow.md`
- 新建 `WORKLOG.md`

### 生成的文件

- `Cursor_workflow.md`
- `WORKLOG.md`

### 工作说明

- Markdown 版本对 PDF 提取内容进行了结构化整理，补齐了标题、任务分节、公式和目录结构，便于直接阅读。
- 后续每次工作如有修改文件、生成文件或关键思路变化，都应继续追加到本日志中。

## 2026-04-21 - 增加全年每日发电图

### 目标

在 `pv_generation.py` 中加入专门的「全年每日发电」可视化，方便一眼看完 365 天的发电分布。

### 已完成内容

- 在 `pv_generation.py` 中新增 `plot_daily_full_year(df)` 函数
  - 输出一张 365 天的逐日柱状图，按月份上色
  - 输出一张月×日的日历式热力图，颜色越深代表当日发电越多
- 在 `main()` 中调用新函数，并把两个新图加入终端打印的输出列表

### 修改的文件

- `pv_generation.py`

### 预计生成的文件

- `plots/pv_daily_full_year_bar.png`
- `plots/pv_daily_full_year_heatmap.png`

### 工作说明

- 原来的 `plot_daily_monthly` 画的是折线日图和月度柱状图，这次新增的两张图更适合「一眼看全年」。
- 本次运行脚本时 Windows 报 `页面文件太小，无法完成操作`，导致 `numpy / pandas / matplotlib` 的 DLL 加载失败。代码本身已通过 lint 检查，等系统资源充足后直接运行 `python pv_generation.py` 即可生成新图。

## 2026-04-21 - 负荷曲线 load_generation.py

### 目标

按确认方案生成 2023 全年逐时负荷，与 `pv_generation_8760.csv` 时间戳对齐；节假日用 `holidays` + 国务院放假/补班表；图表与光伏脚本同类四套输出。

### 已完成内容

- 新增 `load_generation.py`
- HVAC 口径（南沙）：**1–3、11–12 月** 基准 `26.1 kW`；**4–10 月** 基准 `47.7 kW`（与 PDF 原 5–9 月不同，全年总电量会高于设计院 `60,461 kWh` 的 PDF 口径）
- 日程：`holidays.country_holidays('CN', years=2023)` 与国务院办公厅公布的 2023 年放假日期取并集；补班日强制使用工作日系数
- 输出 `output/load_8760.csv`；生成 `load_daily_kwh.png`、`load_monthly_kwh.png`、`load_daily_full_year_bar.png`、`load_daily_full_year_heatmap.png`
- `requirements.txt` 增加 `holidays>=0.45`
- 图表标题使用英文，避免默认字体缺中文字形警告（与 `pv_generation.py` 图表风格一致）

### 修改的文件

- `load_generation.py`（新建）
- `requirements.txt`

### 生成的文件

- `output/load_8760.csv`
- `plots/load_daily_kwh.png`
- `plots/load_monthly_kwh.png`
- `plots/load_daily_full_year_bar.png`
- `plots/load_daily_full_year_heatmap.png`

### 工作说明

- 用户未选定是否将全年总电量缩放至 `60,461 kWh`，当前为**不缩放**，仅打印与参考值的差值。
- 若需与设计院年报电量一致，可后续增加统一缩放系数或改回 PDF 的 5–9 月空调季。

## 2026-04-21 - 逐日光伏与负荷对比图（未调度）

### 目标

在同一张图里展示全年逐日光伏发电与逐日负荷，并直观表示两者差值。

### 已完成内容

- 新增 `pv_load_daily_compare.py`
- 读取 `output/pv_generation_8760.csv` 与 `output/load_8760.csv`，按日汇总 kWh
- 上图：PV 与负荷两条曲线；两者之间填色（绿 = 光伏盈余，红 = 负荷大于光伏）
- 下图：逐日净电量柱状图 `PV - Load`（绿为正、红为负）
- 输出 `plots/pv_load_daily_compare.png`

### 修改的文件

- `pv_load_daily_compare.py`（新建）

### 生成的文件

- `plots/pv_load_daily_compare.png`

### 工作说明

- 差值用两种方式表达：上图阴影面积对应「两曲线之间的垂直差距」，下图柱状图给出每日净电量数值方向。

## 2026-04-21 - 负荷脚本重构（分项模型 + 校准口径）

### 目标

旧版 `load_generation.py` 用「总基准功率 × 日程系数」导致年电量被抬高到 168 MWh，与设计院口径相差近 3 倍。根据新增设计院资料（空调耗电约占 60%、分项负荷表、日用电量 390.7 kWh 仅作结构参考），重构为分项负荷模型，并校准到设计院 60,461 kWh 口径。

### 已完成内容

- 清理旧产物：`output/load_8760.csv`、`plots/load_daily_kwh.png`、`plots/load_monthly_kwh.png`、`plots/load_daily_full_year_bar.png`、`plots/load_daily_full_year_heatmap.png`
- 重写 `load_generation.py`，采用三段式分项模型：
  - **基础负荷 base_kW**（照明、插座、机房等）：全年存在；工作日/周末使用原 ASHRAE 90.1 形状
  - **空调负荷 hvac_kW**：仅 4–10 月；月度强度按 `HVAC_MONTH_SHAPE`（4 月 0.30、5 月 0.70、6 月 0.95、7–8 月 1.00、9 月 0.85、10 月 0.40）；日内形状白天强、夜间 0
  - **附属负荷 aux_kW**（水泵、室外照明等）：傍晚/夜间较重
- 目标标定：全年 `60,461 kWh`，占比 `30 / 60 / 10 %`（基础/空调/附属），每条分项按各自目标电量缩放
- 节假日规则保留：`holidays` 中国法定假日 + 国务院 2023 年放假调休表取并集；补班日强制使用工作日形状
- CSV 扩展为四列：`datetime, base_kW, hvac_kW, aux_kW, load_kW`，便于后续调度/灵敏度分析分别处理
- 图表：
  - 沿用日/月、全年日柱、全年日历热力图（与 PV 图风格一致）
  - 月度图改为 **三分量堆叠柱**，一眼看出 HVAC 季节性
  - 新增 `load_typical_day_stack.png`：冬季代表日 (2023-01-18) 与夏季代表日 (2023-07-19) 分量叠加

### 修改的文件

- `load_generation.py`（整体重写）

### 生成的文件

- `output/load_8760.csv`（新格式，四列）
- `plots/load_daily_kwh.png`
- `plots/load_monthly_kwh.png`（三分量堆叠柱）
- `plots/load_daily_full_year_bar.png`
- `plots/load_daily_full_year_heatmap.png`
- `plots/load_typical_day_stack.png`

### 验证结果

- 年总：**60,461.0 kWh**（精确命中目标）
- 基础：18,138.3 kWh（30.0 %）
- HVAC：36,276.6 kWh（60.0 %）
- 附属：6,046.1 kWh（10.0 %）

### 工作说明

- HVAC 月份沿用此前用户口径（11–3 月不开、4–10 月开）。PDF 原 5–9 月口径若需切换，只改 `HVAC_MONTH_SHAPE` 几个月份系数即可。
- `pv_load_daily_compare.py` 无需改代码，仍读 `output/load_8760.csv` 的 `load_kW` 列；但那条曲线会基于新总量（≈60 MWh 级）重绘，需要等下一次调用。

## 2026-04-21 - 对比脚本新增小时级周叠加图

### 目标

在 `pv_load_daily_compare.py` 中新增“小时级周叠加图”，以直观看到夜间负荷缺口与连续阴雨天多日缺口。

### 已完成内容

- 新增周窗口参数：`WEEK_START='2023-07-17'`、`WEEK_END='2023-07-23'`
- 同图绘制逐时 `PV` 与 `Load` 两条曲线，并用红/绿填色表示每小时缺口/盈余
- 新增输出 `plots/pv_load_weekly_hourly_overlay.png`
- 保留原全年逐日对比图输出 `plots/pv_load_daily_compare.png`

### 修改的文件

- `pv_load_daily_compare.py`

### 生成的文件

- `plots/pv_load_daily_compare.png`（基于新负荷重绘）
- `plots/pv_load_weekly_hourly_overlay.png`（新增）

### 工作说明

- 小时图直观显示：每天夜间都是红区（PV=0、base+aux 仍在），以及当出现多日阴雨时会跨日红区，印证「日级绿 ≠ 小时级够」的判断。

## 2026-04-21 - 重写 README.md

### 目标

原 README 偏简略，有几处与现状不符（装机容量、建筑面积数字不准），并未覆盖 Git 流程、设计院/物理口径差异、当前模型局限等。

### 已完成内容

- 整体重写为结构化长文档，分 10 节：背景 / 结构 / 使用 / 模块详解 / 关键结果 / 讨论 / 局限 / 进度 / Git / 参考
- 修正装机容量：由 ~60 kWp 修正为 **≈ 79 kWp**（`250×0.21 + 40×0.12 + 67.6×0.21 + 62.4×0.12`）
- 修正有效光伏面积：由 ~350 m² 修正为 **420 m²**（290 + 67.6 + 62.4，与设计院“浮体光伏分布分析”一致）
- 增加讨论章节「与设计院口径的差异」，记录 110,888 kWh vs 55,580 kWh 的约 2 倍差，以及 55,580 ≈ 60,461 × 91.93% 的强吻合
- 增加「当前模型局限与待改进」列表，显式说明年总量吻合是构造出来的
- 增加 Git 与版本控制简明流程
- 交叉引用 `Cursor_workflow.md`、`WORKLOG.md`、设计院三份资料

### 修改的文件

- `README.md`（整体重写）

## 2026-04-21 - 对比脚本新增小时级周叠加图

### 目标

在 `pv_load_daily_compare.py` 中新增“小时级周叠加图”，以直观看到夜间负荷缺口与连续阴雨天多日缺口。

### 已完成内容

- 新增周窗口参数：`WEEK_START='2023-07-17'`、`WEEK_END='2023-07-23'`
- 新增输出图：`plots/pv_load_weekly_hourly_overlay.png`
- 同图绘制逐时 `PV` 与 `Load` 两条曲线，并用红/绿填色表示每小时缺口/盈余
- 保留原全年逐日对比图输出 `plots/pv_load_daily_compare.png`

### 修改的文件

- `pv_load_daily_compare.py`

### 生成的文件

- `plots/pv_load_daily_compare.png`（重绘）
- `plots/pv_load_weekly_hourly_overlay.png`（新增）

## 2026-05-05 - 组会主线文档与 README 重构

### 目标

将项目叙事对齐组会：**浮式建筑为主体**、校园实测 PV 为**本地扰动参照**、后续重点为**储能调度与离岛韧性**；补充数据申请清单与汇报逐页稿。

### 已完成内容

- 重写 [README.md](README.md) 开篇与新增 **§0 研究定位与路线**（主体分工、`PV = PVGIS × F` 框架、实验矩阵摘要）
- 新增 [research_plan.md](research_plan.md)（数据流、学习产物、E0–E5 矩阵、指标与代码顺序）
- 新增 [data_request.md](data_request.md)（向学校申请能源中心/数据中心字段与时间粒度）
- 新增 [deliverables/group_meeting_slides.md](deliverables/group_meeting_slides.md)（约 10 页组会逐页稿）
- 在 [Cursor_workflow.md](Cursor_workflow.md) 文首增加指向 `research_plan.md` 与 `load_generation.py` 的说明
- [README.md](README.md) §8 开发进度勾选上述文档交付项

### 修改或新增的文件

- `README.md`
- `research_plan.md`
- `data_request.md`
- `deliverables/group_meeting_slides.md`
- `Cursor_workflow.md`

### 工作说明

实现层脚本（`pv_local_pattern.py`、`pv_scenario_generation.py`、`dispatch.py`）仍待开发；当前交付为**研究路线与汇报材料**，便于组会统一口径与数据申请。

## 2026-05-05 - 组会口径二次收敛（简化外部数据叙事）

### 目标

按组会准备需求，将文档叙事收敛为：只并列说明**光伏出力来源**与**负荷来源**，并聚焦后续调度与离岛工作。

### 已完成内容

- 调整 `README.md` 开篇与第 0 节，简化外部数据路线描述
- 重写 `research_plan.md`，改为数据来源、实验矩阵、实施顺序三段式
- 重写 `deliverables/group_meeting_slides.md`，将汇报结构改为「数据来源 -> 基准结果 -> 调度与离岛」
- 更新 `Cursor_workflow.md` 文首提示语，删除校园/F 因子表述
- 重写 `data_request.md` 为「数据来源说明（光伏与负荷）」

### 修改的文件

- `README.md`
- `research_plan.md`
- `deliverables/group_meeting_slides.md`
- `Cursor_workflow.md`
- `data_request.md`

### 工作说明

该版本可直接用于组会展示：不引入新数据路线讨论，先把当前可复现链路和后续调度工作讲清楚。

## 2026-05-05 - 恢复实测 PV 扰动系数路线

### 目标

修正上一版过度收敛的问题：本项目不展开外部建模或换主体讨论，但应保留上午讨论形成的**能源中心 / 数据中心实测 PV → 扰动系数 → 合成光伏情景**路线。

### 已完成内容

- 在 `README.md` 第 0 节恢复实测 PV 参照数据与 `LocalPVScenario` 实验
- 重写 `research_plan.md`，加入扰动系数情景公式 `PV_scenario = PVGIS_floating × disturbance_factor`
- 重写 `data_request.md`，加入实测 PV 申请字段与扰动系数用途
- 更新 `deliverables/group_meeting_slides.md`，在数据来源、核心发现、实验矩阵和讨论点中恢复扰动系数情景
- 更新 `Cursor_workflow.md` 文首主线说明
- 在 `Cursor_workflow.md` 中新增 **任务 1b：实测 PV 扰动系数与情景光伏曲线**，明确其属于光伏出力数据工作的一部分

### 工作说明

当前口径：**PVGIS 是基准光伏来源，实测 PV 是情景扰动来源；负荷仍采用设计院年电量 + 分项负荷模型；后续重点仍是储能调度与离岛韧性。**

## 2026-05-05 - 明确极端天气事件窗口尺度

### 目标

避免将极端天气误解为“整年极端气候”。明确全年基准由 2023 PVGIS 表示，极端天气仅作为嵌入全年基准的**事件窗口**，用于调度和离岛压力测试。

### 已完成内容

- 在 `research_plan.md` 新增 **极端天气尺度设置**：`24 h`、`48 h`、`72 h`、可选 `7 d`
- 在 `README.md` 实验矩阵后补充极端天气不是整年情景的说明
- 在 `deliverables/group_meeting_slides.md` 第 6 页补充极端天气尺度和用途
- 在 `Cursor_workflow.md` 孤岛工况中将“典型日”改为事件窗口（后续已进一步收敛为只用实测 PV 低出力片段）

### 工作说明

极端天气脚本后续应输出 `output/pv_generation_8760_extreme_*.csv` 或并入情景文件，作为 `dispatch.py` 输入，而不是独立研究结论。

## 2026-05-05 - 极端天气来源收敛为实测 PV

### 目标

进一步收敛极端天气口径：极端低出力事件直接从能源中心 / 数据中心实测 PV 中提取，不再并列写 PVGIS 低发电窗口或人工合成压力测试作为来源。

### 已完成内容

- `research_plan.md`：将 E2 和极端天气生成方式改为“从实测 PV 提取最低 24h / 48h / 72h / 可选 7d 低出力事件”
- `README.md`：将 E2 描述改为实测 PV 低出力事件叠加到 E0/E1
- `Cursor_workflow.md`：孤岛工况中删除 PVGIS 低发电窗口来源，只保留实测 PV 事件窗口
- `deliverables/group_meeting_slides.md`：第 6 页改为从实测 PV 提取连续低出力事件
- `data_request.md`：补充极端天气事件提取字段用途

### 工作说明

当前口径：**极端天气事件 = 能源中心 / 数据中心实测 PV 中的真实连续低出力片段**，再转为扰动系数并作用到浮式建筑 PVGIS 基准。

## 2026-06-01 - HKUST Clear Water Bay PV + Wind 数据体检

### 目标

初步接入 `HKUST_clea_bay_PV+wind_data` 数据包，确认其文件结构、时间范围、采样间隔和缺失情况，为后续 PVGIS 校准、风电估算和调度输入表构建做准备。

### 已完成内容

- 新增 `inspect_hkust_data.py`
- 扫描 `HKUST_clea_bay_PV+wind_data/Dataset/Time series dataset/` 下的 CSV/XLSX 时序文件
- 统计每个文件的类别、列名、行数、起止时间、典型采样间隔和缺失单元格比例
- 输出文件级清单 `output/hkust_data_inventory.csv`
- 输出 Markdown 摘要 `output/hkust_data_summary.md`

### 主要发现

- 共检查 `125` 个时序文件，合计约 `19,015,875` 行。
- 气象数据覆盖 `2021-01-01` 至 `2023-12-31`，典型采样间隔为 `1 min`。
- PV 站点级数据主要为 `15 min`，逆变器级数据主要为 `5 min`，无优化器站点数据主要为 `60 min`。
- 站点级 PV 数据中，`UG Hall4.csv`、`SQ567.csv`、`UG Hall7.csv`、`SQ1.csv` 等覆盖较长且缺失率较低，可优先作为 PVGIS 校准样本。

### 修改的文件

- `inspect_hkust_data.py`（新建）
- `WORKLOG.md`

### 生成的文件

- `output/hkust_data_inventory.csv`
- `output/hkust_data_summary.md`

### 工作说明

- 当前脚本只做数据体检，不修改原始数据。
- 本机默认 `python3` 环境缺少 `pandas`，本次使用 Codex bundled Python 运行脚本。
- 下一步建议基于低缺失率站点构建小时级实测 PV 扰动系数，并与现有 `output/pv_generation_8760.csv` 对齐。

## 2026-06-01 - 15 分钟天气驱动分项负荷模型

### 目标

将负荷侧从小时级 `load_8760.csv` 扩展为 15 分钟级合成负荷，以便和 HKUST 实测 PV 站点数据及 15 分钟气象聚合数据对齐；避免仅将小时负荷复制为 4 段。

### 已完成内容

- 新增 `load_generation_15min.py`
- 读取 HKUST 2023 年 1 分钟气象数据：温度、相对湿度、辐照、降雨
- 聚合为完整 15 分钟时间轴，共 `35,040` 行
- 构建分项负荷：`base_kW`、`plug_kW`、`lighting_kW`、`hvac_kW`、`aux_kW`
- 使用工作日/周末/法定假日/补班日 schedule 生成 occupancy 与 operation 因子
- HVAC 由空调季、占用时段、室外温度、湿度和辐照共同驱动
- 各分项分别校准到设计年电量 `60,461 kWh`

### 修改的文件

- `load_generation_15min.py`（新建）
- `WORKLOG.md`

### 生成的文件

- `output/hkust_weather_15min_2023.csv`
- `output/load_15min_2023.csv`

### 主要结果

- 年总负荷：`60,461.0 kWh`
- `base_kW`：`7,255.3 kWh`（12.0%）
- `plug_kW`：`7,859.9 kWh`（13.0%）
- `lighting_kW`：`6,046.1 kWh`（10.0%）
- `hvac_kW`：`36,276.6 kWh`（60.0%）
- `aux_kW`：`3,023.1 kWh`（5.0%）

### 工作说明

- 当前 15 分钟负荷为合成模型，不是本建筑实测负荷。
- 模型依据为“分项负荷 + schedule + 天气驱动 + 年电量校准”，不是小时曲线插值。
- 当前运行环境缺少 `matplotlib`，因此脚本跳过图表生成，但 CSV 输出和电量校准已验证通过。

## 2026-06-01 - HKUST 实测 PV 15 分钟对齐

### 目标

将 HKUST 站点级实测 PV 数据接入 15 分钟时间轴，并与 `load_15min_2023.csv` 对齐，作为后续提取实测扰动系数的数据基础。

### 已完成内容

- 新增 `align_hkust_pv_15min.py`
- 默认选择 `8` 个覆盖长、缺失率低的站点：`UG Hall4`、`SQ567`、`UG Hall7`、`SQ1`、`SQ4`、`SQ2`、`LSK North`、`SQ3`
- 读取站点级 `power(W)`，转换为 `kW`，对齐到 2023 年完整 15 分钟时间轴
- 对短缺测进行时间插值，并保留 `raw_missing_site_count` 作为数据质量标记
- 输出 HKUST 实测 PV 聚合表
- 输出直接合并实测 PV 与合成负荷的数据核查表
- 删除此前生成的 `output/dispatch_input_15min_hkust_scaled_2023.csv`，避免误解为 HKUST 实测 PV 可直接缩放代表漂浮建筑光伏

### 修改的文件

- `align_hkust_pv_15min.py`（新建，并移除缩放 PV 情景逻辑）
- `WORKLOG.md`

### 生成的文件

- `output/hkust_pv_measured_15min_2023.csv`
- `output/dispatch_input_15min_2023.csv`

### 主要结果

- 实测 HKUST PV 聚合年发电量：`316,760.4 kWh`
- 实测 HKUST PV 聚合峰值：`265.5 kW`
- 至少一个站点原始值缺失的 15 分钟时段：`15`

### 工作说明

- `dispatch_input_15min_2023.csv` 中的 HKUST 实测 PV 是多个屋顶站点聚合值，规模大于漂浮建筑，主要用于数据核查和扰动形状来源。
- HKUST 实测 PV 不用于决定漂浮建筑光伏装机容量或绝对发电量。
- 后续正确路线是从 HKUST 实测 PV 提取扰动系数，再作用到原有漂浮建筑 PVGIS 发电曲线上。

## 2026-06-01 - 15 分钟 PV 扰动系数叠加

### 目标

从 HKUST 实测 15 分钟 PV 数据中提取无量纲扰动系数，并叠加到漂浮建筑原有 PVGIS 光伏出力上，生成 15 分钟 PV 情景和调度输入。

### 已完成内容

- 新增 `build_pv_disturbance_15min.py`
- 将原有小时级 `output/pv_generation_8760.csv` 转为本地时间 15 分钟 step-hold PVGIS 基准
- 用 HKUST 实测 PV 构建每个 15 分钟时刻的 clear-like 高分位基线
- 计算扰动系数 `disturbance_factor = measured / clear_like_baseline`
- 对扰动系数进行限幅，当前上限为 `1.20`
- 将扰动系数叠加到漂浮建筑 PVGIS 基准上
- 同时输出两种 PV 情景：
  - `raw`：真实扰动直接作用后的 PV，年发电量可低于 PVGIS
  - `annual_preserved`：将扰动后 PV 归一化回原 PVGIS 年发电量，用于只研究短时波动影响

### 修改的文件

- `build_pv_disturbance_15min.py`（新建）
- `WORKLOG.md`

### 生成的文件

- `output/pv_disturbance_15min_2023.csv`
- `output/pv_floating_pvgis_disturbed_15min_2023.csv`
- `output/dispatch_input_15min_disturbed_2023.csv`

### 主要结果

- PVGIS 基准年发电量：`110,888.0 kWh`
- 原始扰动后年发电量：`75,367.0 kWh`
- 年总量保持版年发电量：`110,888.0 kWh`
- 年总量归一化系数：`1.4713`
- 扰动系数均值：`0.805`
- 扰动系数 `5% / 50% / 95%` 分位数：`0.104 / 1.000 / 1.000`
- 原始扰动 PV 下净负荷范围：`-68.5` 至 `28.6 kW`
- 年总量保持 PV 下净负荷范围：`-104.8` 至 `25.5 kW`

### 工作说明

- `raw` 情景更接近真实天气折减，可用于极端低出力/韧性压力测试。
- `annual_preserved` 情景保持原 PVGIS 年发电量，可用于分析短时波动对调度的影响。
- 后续极端天气事件应优先从 `pv_disturbance_15min_2023.csv` 中提取连续低扰动窗口，而不是再次缩放 HKUST 实测绝对 PV。

## 2026-06-01 - 15 分钟 RBC 储能调度基线

### 目标

基于 `dispatch_input_15min_disturbed_2023.csv` 建立第一版 15 分钟规则控制（RBC）储能调度，用于评估扰动 PV 情景下的缺电、弃光和电池 SOC。

### 已完成内容

- 新增 `dispatch_rbc_15min.py`
- 默认使用 `pv_floating_disturbed_raw_kW` 作为 PV 输入
- 默认电池参数：`215 kWh` 容量、`50 kW` 功率、`92%` 往返效率、初始 SOC `50%`
- 规则：
  - PV 大于负荷时，优先给电池充电，剩余计为弃光
  - PV 小于负荷时，优先电池放电，剩余计为缺电
- 输出完整 15 分钟调度结果与年度汇总指标

### 修改的文件

- `dispatch_rbc_15min.py`（新建）
- `plot_dispatch_rbc_15min.js`（新建，用于在无 matplotlib 环境下输出 SVG 图）
- `WORKLOG.md`

### 生成的文件

- `output/dispatch_rbc_15min_raw_battery215_2023.csv`
- `output/dispatch_rbc_15min_raw_battery215_summary.csv`
- `plots/dispatch_rbc_15min_raw_battery215_soc.svg`
- `plots/dispatch_rbc_15min_raw_battery215_worst_day.svg`
- `plots/dispatch_rbc_15min_raw_battery215_monthly.svg`

### 主要结果

- 负荷年电量：`60,461.0 kWh`
- raw 扰动 PV 年发电量：`75,367.0 kWh`
- 缺电量：`11,132.4 kWh`（`18.41%`）
- 弃光量：`24,631.3 kWh`（PV 的 `32.68%`）
- PV 自发自用率：`67.32%`
- 最低 SOC：`0.0 kWh`
- 最长连续缺电时长：`95.50 h`

### 工作说明

- 当前结果是项目给定 `215 kWh / 50 kW / 0.92` 电池配置下的压力测试基线。
- 后续调度实验应保持电池容量和功率不变，对比 RBC / MILP / MPC 在同一硬件约束下的运行表现。

## 2026-06-18 - 下周汇报可视化进度补充

### 目标

为下周组会准备肉眼可见的阶段性进展：把原计划中的 E2 极端低 PV 事件做成可展示的表格和图，并整理固定设计电池下的 E3 RBC 基线结果。

### 已完成内容

- 新增 `extract_extreme_pv_events_15min.py`
- 从 `dispatch_input_15min_disturbed_2023.csv` 中提取最低连续 `24 h`、`48 h`、`72 h`、`168 h` raw 扰动 PV 窗口
- 输出下周汇报简报 `deliverables/next_week_progress_brief.md`

### 生成的文件

- `output/extreme_low_pv_events_15min.csv`
- `output/extreme_low_pv_events_15min.md`
- `plots/extreme_low_pv_events_15min.svg`
- `deliverables/next_week_progress_brief.md`

### 主要结果

- 最低 `24 h` 低 PV 窗口：`2023-09-07 16:45` 至 `2023-09-08 16:30`，raw 扰动 PV `1.8 kWh`，同期负荷 `166.5 kWh`
- 最低 `48 h` 低 PV 窗口：`2023-10-08 10:00` 至 `2023-10-10 09:45`，raw 扰动 PV `10.9 kWh`，同期负荷 `276.7 kWh`
- 最低 `72 h` 低 PV 窗口：`2023-09-07 15:15` 至 `2023-09-10 15:00`，raw 扰动 PV `19.7 kWh`，同期负荷 `365.1 kWh`
- 当前 `215 kWh / 50 kW` RBC 缺电量仍为 `11,132.4 kWh`，缺电率 `18.41%`

### 工作说明

- E2 现在有了可落地的实测扰动低 PV 事件窗口，不再只是研究计划文字。
- E3 仍保持项目给定的 `215 kWh / 50 kW` 电池配置，不做容量优化或重新选型。
- 当前结果说明在 raw 扰动 PV + 简单 RBC 下，既定设计电池仍存在明显缺电，后续应继续推进 MILP / MPC，并区分并网年度运行与离岛极端事件韧性。

## 2026-06-24 - 固定设计电池的极端低 PV 事件 RBC 实验

### 目标

在不改变电池容量和功率的前提下，继续推进 E2 极端天气 / 离岛韧性实验：用固定设计电池 `215 kWh / 50 kW` 跑已提取的 `24 h`、`48 h`、`72 h`、`7 d` 低 PV 事件窗口。

### 已完成内容

- 新增 `rbc_extreme_events_fixed_battery_15min.py`
- 沿用当前 RBC 规则和固定电池参数：`215 kWh` 容量、`50 kW` 功率、`0.92` 往返效率、初始 SOC `50%`
- 对 `output/extreme_low_pv_events_15min.csv` 中四个低 PV 事件分别运行事件级 RBC
- 输出每个事件的详细 15 分钟调度表
- 输出事件级汇总表和 5 张汇报用 SVG 图

### 生成的文件

- `output/rbc_extreme_events_fixed_battery_summary.csv`
- `output/rbc_extreme_events_fixed_battery_summary.md`
- `output/rbc_extreme_event_24h_fixed_battery_15min.csv`
- `output/rbc_extreme_event_48h_fixed_battery_15min.csv`
- `output/rbc_extreme_event_72h_fixed_battery_15min.csv`
- `output/rbc_extreme_event_7d_fixed_battery_15min.csv`
- `plots/rbc_extreme_events_fixed_battery_summary.svg`
- `plots/rbc_extreme_event_24h_fixed_battery_dispatch.svg`
- `plots/rbc_extreme_event_48h_fixed_battery_dispatch.svg`
- `plots/rbc_extreme_event_72h_fixed_battery_dispatch.svg`
- `plots/rbc_extreme_event_7d_fixed_battery_dispatch.svg`

### 主要结果

- `24 h` 事件：PV `1.8 kWh`，负荷 `166.5 kWh`，缺电 `61.6 kWh`，缺电率 `37.0%`
- `48 h` 事件：PV `10.9 kWh`，负荷 `276.7 kWh`，缺电 `162.7 kWh`，缺电率 `58.8%`
- `72 h` 事件：PV `19.7 kWh`，负荷 `365.1 kWh`，缺电 `242.3 kWh`，缺电率 `66.4%`
- `7 d` 事件：PV `124.1 kWh`，负荷 `521.5 kWh`，缺电 `295.5 kWh`，缺电率 `56.7%`
- 四个事件中电池最低 SOC 均达到 `0.0 kWh`

### 工作说明

- 该实验保持电池硬件参数固定，不做容量 / 功率优化。
- 结果说明：即使全年 PV 能量有盈余，连续低 PV 事件下固定设计电池仍会被打空，因此离岛韧性需要按事件窗口单独评估。
- 下一步可在相同固定硬件和相同事件窗口下实现 MILP，对比 RBC 与理论最优调度差距。

## 2026-06-24 - E4-b 全年并网 MILP 调度实验

### 目标

在固定设计电池 `215 kWh / 50 kW` 和 raw 扰动 PV 输入下，开展全年并网 MILP 调度实验，用于和已有 RBC 并网经济性结果对比。

### 已完成内容

- 新增 `milp_e4b_full_year_grid_fixed_battery.py`
- 使用 `pv_floating_disturbed_raw_kW` 和 `load_kW`
- 电池容量和功率固定，不做容量优化
- 电池仅允许由 PV 充电，不允许电网充电
- 加入同一时间步购电 / 售电互斥约束，避免不合理套利
- 按月求解全年 12 个 MILP，并将 SOC 在月间传递
- 目标函数为最小化净电网成本：购电成本 - 售电收益，并加入小的弃光和循环惩罚

### 生成的文件

- `output/milp_e4b_full_year_grid_dispatch_2023.csv`
- `output/milp_e4b_full_year_grid_summary.csv`
- `output/milp_e4b_full_year_grid_summary.md`
- `output/milp_e4b_full_year_grid_monthly_summary.csv`
- `plots/milp_e4b_progress.svg`
- `plots/milp_e4b_full_year_soc.svg`
- `plots/milp_e4b_monthly_import_export_cost.svg`
- `plots/milp_e4b_rbc_vs_milp_grid_cost.svg`

### 主要结果

- 全年 12 个月均求解为 `Optimal`
- 年购电量：`15,695.6 kWh`
- 年售电量：`29,560.3 kWh`
- 弃光量：`0.0 kWh`
- 购电成本：`7,539.8 CNY`
- 售电收益：`10,641.7 CNY`
- 净电网成本：`-3,101.9 CNY`
- 最低 SOC：`0.0 kWh`
- 期末 SOC：`0.0 kWh`

### 工作说明

- 与已有 RBC 经济性口径相比，MILP 的净电网成本从约 `-3.0 CNY` 改善到 `-3,101.9 CNY`。
- MILP 相比 RBC 增加了购电量和售电量，说明其利用分时电价与上网电价，在固定电池约束下进行经济调度。
- 当前 E4-b 为按月求解、SOC 月间传递的全年 MILP 第一版；若后续需要更严格的理论上界，可再尝试一次性全年优化。
# 2026-08-02 - 无设备清单条件下的 CL1/CL2/CL3 代理负荷分级

## 目标

在无法取得浮式建筑真实设备表、配电回路表和关键负荷构成的条件下，基于现有 15 分钟合成分项负荷建立透明、可替换的离岛运行优先级情景，避免把假设描述成建筑实测分级。

## 已完成内容

- 新增 `deliverables/load_classification_framework.md`，定义 `CL1 安全关键 / CL2 功能保障 / CL3 可中断`，并说明其不直接等同于国标一级、二级、三级供电负荷。
- 新增 `deliverables/load_inventory_template.csv`，保留未来接入设备或回路资料的字段。
- 新增 `build_tiered_load_scenarios.py`。
- 将现有 `base / plug / lighting / HVAC / aux` 视为五组母回路，通过显式分配矩阵生成三套功能代理情景：
  - `emergency`；
  - `minimum_operation`；
  - `extended_operation`。
- 每套情景在每个 15 分钟步严格保持 `CL1 + CL2 + CL3 = load_kW`。

## 生成的文件

- `output/load_tier_allocation_matrix.csv`
- `output/load_tiers_15min_2023.csv`
- `output/load_tiers_15min_summary.csv`
- `output/load_tiers_15min_summary.md`

## 主要结果

- `emergency`：CL1 `7.2%`，CL2 `17.4%`，CL1+CL2 峰值 `6.27 kW`。
- `minimum_operation`：CL1 `10.1%`，CL2 `29.8%`，CL1+CL2 峰值 `13.71 kW`。
- `extended_operation`：CL1 `13.4%`，CL2 `47.6%`，CL1+CL2 峰值 `25.61 kW`。
- 三套情景的最大逐时负荷重构误差均约为 `7.1e-15 kW`。

## 工作说明

- 上述比例是研究情景，不是实测设备占比或竣工回路清单。
- 中心情景 `CL1+CL2` 峰值与旧任务说明中的固定 `13 kW` 量级接近，只作为交叉检查，不作为其来源验证。
- 后续可直接把三套时序接入离岛储能 sizing 和 survivability 分析；若未来得到真实设备资料，只需替换分配矩阵。

# 2026-09-09 - 数据冻结与正式 M0 预测训练

- 修复 PV 扰动基准的未来信息依赖，改为此前 31 天同时间槽基准，由 2022 年 12 月初始化；气象采用有界因果补值并保留观测和质量标记，长缺测影响的窗口排除。
- 使用 2022 年拟合的固定负荷系数，加入显式日历电价，冻结 `output/frozen/hkust_m0_v1_20260909/`。35,040 行连续时间轴，有效 train/val/test 窗口为 21,695 / 4,311 / 5,377；标准化仅拟合训练期有效行。旧输出未覆盖。
- 增加 `train`、`evaluate` 独立入口、严格验证最优 `best.pt`、可复现续训 `last.pt`、逐轮日志、数据哈希绑定及昨日同期基线。数据/训练/评估共 21 项行为测试通过；压缩预测明细可独立重算全部分组和预测步指标。
- 正式 M0 使用相同单层 LSTM（hidden 32、历史 672 步、预测 96 步），seeds 42/43/44，最多 50 轮，patience 8，min_delta 0.01 kW²；分别于 20/24/18 轮早停，最佳轮次为 12/16/10。未根据测试集选 seed 或调参。
- 实验目录 `experiments/m0_frozen_v1_20260909/` 保存协议、数据清单、源代码快照、环境、三份 best/last 和完整独立 val/test 评估。套件运行约 14 分钟，最终产物核验通过。
- 测试集三种子平均 RMSE 7.1234 kW，昨日同期基线 8.3905 kW，下降 15.10%；MAE 为 4.3916 kW，高于基线 3.6022 kW，近零 PV 时段正偏明显，不能表述为全面优于基线。
- 结果与使用方式见 `deliverables/M0正式训练结果_2026-09-09.md` 和 `deliverables/M0训练与独立评估使用说明.md`。标签仍是构造浮式 PV 场景；本轮不包含 M1/M2、正式 MPC 闭环或经济/韧性结论。
