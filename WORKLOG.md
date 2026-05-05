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
