# MPhil Floating Building — 水上零碳建筑光伏 / 负荷 / 能量对比分析

> HKUST(GZ) Waterfront Living Lab 浮体零碳建筑 MPhil 研究项目。  
> 基于 2023 年 8760 小时 PVGIS 辐照数据 + ASHRAE 负荷日程 + 设计院资料，建立**可复现**的光伏发电、分项建筑负荷与未调度供需对比基准。  
> 后续将在此基础上做储能调度（RBC / MILP / MPC）与灵敏度分析。

---

## 目录

- [1. 研究背景与目标](#1-研究背景与目标)
- [2. 项目结构](#2-项目结构)
- [3. 使用方法](#3-使用方法)
- [4. 模块详解](#4-模块详解)
  - [4.1 pv_generation.py](#41-pv_generationpy)
  - [4.2 load_generation.py](#42-load_generationpy)
  - [4.3 pv_load_daily_compare.py](#43-pv_load_daily_comparepy)
- [5. 关键结果](#5-关键结果)
- [6. 讨论：与设计院口径的差异](#6-讨论与设计院口径的差异)
- [7. 当前模型局限与待改进](#7-当前模型局限与待改进)
- [8. 开发进度](#8-开发进度)
- [9. Git 与版本控制](#9-git-与版本控制)
- [10. 参考](#10-参考)

---

## 1. 研究背景与目标

研究对象：**HKUST(GZ) Waterfront Living Lab** 浮体零碳建筑，主体由球形幕墙（约 290 m²）与外圈浮体平台（约 130 m²，含玻璃栏杆）组成，坐标位于广州南沙附近（22.896°N, 113.480°E）。

浮体建筑与陆上建筑相比，能源系统有几个明显差异：

1. **水面微气候**：水面反照率略高，且水体冷却可提升组件效率（`+α` & `+β`）。
2. **多朝向 BIPV**：球体同一时刻面对不同方向，需按朝向分别积分。
3. **孤岛潜力**：浮体脱离母岸时需自持运行，因此调度策略尤其重要。
4. **设计院口径 vs 物理口径**：设计院给出的 “年发电 55,580 kWh / 年耗电 60,461 kWh / 能源自持率 91.93%” 通常带综合折减或反推，与纯物理公式不一致，需要说明和对齐。

本仓库要回答的问题是：

> 在 **2023 年** 的 PVGIS 辐照与**设计院口径负荷**下，在**未接入储能调度**的情况下，光伏与负荷在 **逐日 / 逐时** 上是否匹配？缺口主要出现在何时何处？

后续再加入储能与调度策略对比、灵敏度分析。

---

## 2. 项目结构

```
g:\0000Mphil_floating_building\
├── data/                                  PVGIS 原始辐照数据（2023）
│   ├── pvgis_south_45.csv                 南向, 倾角 45°
│   ├── pvgis_east_45.csv                  东向, 倾角 45°
│   ├── pvgis_west_45.csv                  西向, 倾角 45°
│   ├── pvgis_north_45.csv                 北向, 倾角 45°
│   ├── pvgis_deck.csv                     水平面 (甲板)
│   └── pvgis_top.csv                      水平面 (备份, 与 deck 相同, MD5 校验)
├── output/                                逐时结果
│   ├── pv_generation_8760.csv             datetime, pv_kW
│   └── load_8760.csv                      datetime, base_kW, hvac_kW, aux_kW, load_kW
├── plots/                                 可视化结果（共 11 张）
│   ├── pv_daily_kwh.png                   PV 日能量折线
│   ├── pv_monthly_kwh.png                 PV 月能量柱状
│   ├── pv_daily_full_year_bar.png         PV 全年逐日柱（按月着色）
│   ├── pv_daily_full_year_heatmap.png     PV 日历热力图
│   ├── load_daily_kwh.png                 负荷日能量折线
│   ├── load_monthly_kwh.png               负荷月能量（三分量堆叠）
│   ├── load_daily_full_year_bar.png       负荷全年逐日柱
│   ├── load_daily_full_year_heatmap.png   负荷日历热力图
│   ├── load_typical_day_stack.png         冬/夏典型日三分项叠加
│   ├── pv_load_daily_compare.png          全年逐日 PV vs 负荷（含缺口/盈余填色）
│   └── pv_load_weekly_hourly_overlay.png  典型周小时级 PV vs 负荷叠加
├── pv_generation.py                       光伏发电脚本
├── load_generation.py                     分项负荷脚本
├── pv_load_daily_compare.py               未调度供需对比脚本
├── requirements.txt                       Python 依赖
├── Cursor_workflow.md                     PDF 研究任务的中文版说明
├── Cursor_workflow.pdf                    原始研究任务 PDF
├── WORKLOG.md                             中文工作日志（每次改动都会追加）
└── path_convention_note.txt               路径约定备忘
```

---

## 3. 使用方法

### 3.1 环境依赖

`requirements.txt` 如下：

```text
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
pulp>=2.7
scipy>=1.10
tqdm>=4.65
holidays>=0.45
```

推荐在项目虚拟环境内安装：

```powershell
cd g:\0000Mphil_floating_building
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> 若运行时出现 Windows 的 `页面文件太小` 报错，不是代码问题，是系统虚拟内存不足，需要在系统设置里把 pagefile 调到 16–32 GB 后重启。

### 3.2 复现实验的 3 步

```powershell
python pv_generation.py          # 产出 output/pv_generation_8760.csv 与 plots/pv_*.png
python load_generation.py        # 产出 output/load_8760.csv 与 plots/load_*.png
python pv_load_daily_compare.py  # 产出 plots/pv_load_daily_compare.png 与 plots/pv_load_weekly_hourly_overlay.png
```

三个脚本**互相解耦**：光伏和负荷分别生成，只在对比脚本里合并。

---

## 4. 模块详解

### 4.1 pv_generation.py

**作用**：读取 5 份 PVGIS CSV（4 个倾斜面 + 水平甲板），按分区 BIPV 面积/效率/温度修正计算逐时功率，叠加水面微气候修正后得到 `pv_kW`。

**关键参数**（来源于项目 PDF）：

| 分区 | 面积 (m²) | 类型 | 效率 |
|---|---|---|---|
| 球体 - 4 朝向 - 不透明 HJT | 每朝向 62.5（合计 250） | HJT | 0.21 |
| 球体 - 4 朝向 - 透明 CdTe | 每朝向 10（合计 40） | CdTe | 0.12 |
| 外圈浮体甲板 | 67.6 | HJT | 0.21 |
| 玻璃栏杆 | 62.4 | CdTe | 0.12 |

**等效装机容量 ≈ 79 kWp**（`250×0.21 + 40×0.12 + 67.6×0.21 + 62.4×0.12`）。

**单表面计算公式**：

```text
G_total     = Gb(i) + Gd(i) + Gr(i)                   # W/m^2
temp_factor = 1 + (-0.004) * (T2m - 25)
P_kW        = max(0, G_total * area * eff * temp_factor / 1000)
```

**水面微气候修正（对合成功率）**：

```text
P_corrected = P * albedo_factor * cooling_factor
albedo_factor  = 1.05
cooling_factor = 1.012
```

**数据质量处理**：

- PVGIS CSV 跳过前 8 行元数据
- 文件末尾的列说明行被 `to_datetime(errors="coerce")` 自动剔除
- 校验各朝向 CSV 的 `datetime` 严格一致
- 校验 `pvgis_top.csv` 与 `pvgis_deck.csv` 的 MD5：当前两者相同，因此**水平面辐照只计一次**，避免重复累加

**输出**：

- `output/pv_generation_8760.csv` 两列 `datetime, pv_kW`
- 4 张图：`pv_daily_kwh.png`、`pv_monthly_kwh.png`、`pv_daily_full_year_bar.png`、`pv_daily_full_year_heatmap.png`

**运行打印示例**：

```text
Annual PV generation (sum of hourly kW): 110,888.0 kWh
Design estimate (PDF):                   55,580.0 kWh
Difference vs design:                    55,308.0 kWh
```

---

### 4.2 load_generation.py

**作用**：按**分项模型**生成与光伏时间戳**逐行对齐**的逐时负荷。

**模型形式**：

```text
load_kW(t) = base_kW(t) + hvac_kW(t) + aux_kW(t)
```

**分项定义与校准目标**：

| 分项 | 内容 | 年电量目标 | 季节性 | 形状 |
|---|---|---|---|---|
| `base_kW` | 照明、普通插座、机房 | 30 % | 全年 | ASHRAE 工作日 / 周末 |
| `hvac_kW` | 空调 | 60 % | 4–10 月 | 日内白天为主 × 月度权重 |
| `aux_kW` | 水泵、室外照明等 | 10 % | 全年 | 傍晚/夜间较重 |

校准口径：全年合计 **60,461 kWh**（与设计院参考一致），HVAC 占比 **≈ 60%**（参考设计院“空调耗电近 60%”口径）。

**HVAC 月度权重**（南沙口径，用户确认）：

```text
1, 2, 3, 11, 12    -> 0.00    # 不开空调
4                  -> 0.30
5                  -> 0.70
6                  -> 0.95
7, 8               -> 1.00
9                  -> 0.85
10                 -> 0.40
```

> PDF 原口径是 5–9 月开空调，若需切回，只改 `HVAC_MONTH_SHAPE` 即可。

**节假日与调休**：

- 使用 `holidays.country_holidays("CN", years=[2023])`
- 额外维护 `OFFICIAL_REST_2023`（国务院 2023 年放假日期），取并集
- 额外维护 `MAKEUP_WORKDAYS_2023`（国务院 2023 年调休补班日）
- 判定顺序：**补班日 → 工作日形状**；否则**周末或节假日 → 周末形状**；否则 → 工作日形状

**缩放逻辑**：

对每条分项，先计算“未缩放的形状面积” `S`（按小时步长积分），再按目标电量整体乘一个系数：

```text
factor = target_annual_kWh / S
scaled = shape * factor
```

这保证了分项年电量精确命中目标；但也意味着**总量的吻合是构造出来的**，不是验证出来的（见下文 §6、§7）。

**输出**：

- `output/load_8760.csv` 五列 `datetime, base_kW, hvac_kW, aux_kW, load_kW`
- 5 张图：`load_daily_kwh.png`、`load_monthly_kwh.png`（三分量堆叠）、`load_daily_full_year_bar.png`、`load_daily_full_year_heatmap.png`、`load_typical_day_stack.png`（冬季 2023-01-18 + 夏季 2023-07-19 的分项叠加）

**运行打印示例**：

```text
年总负荷:     60,461.0 kWh   (目标 60,461.0 kWh)
  - 基础:    18,138.3 kWh (30.0 %, 目标 30 %)
  - HVAC:    36,276.6 kWh (60.0 %, 目标 60 %)
  - 附属:     6,046.1 kWh (10.0 %, 目标 10 %)
```

---

### 4.3 pv_load_daily_compare.py

**作用**：在未接入储能与调度的情况下，把光伏与负荷做**日级**与**小时级**对比，揭示哪些时段匹配、哪些时段缺口。

**生成的 2 张图**：

1. `plots/pv_load_daily_compare.png`
   - **上**：全年逐日 PV / 负荷曲线；两线之间用**绿色**填色表示盈余、**红色**填色表示缺口
   - **下**：逐日净能量 `PV − Load` 柱状图（绿为正、红为负）

2. `plots/pv_load_weekly_hourly_overlay.png`
   - **默认窗口 2023-07-17 ~ 2023-07-23**（可改 `WEEK_START` / `WEEK_END`）
   - 同图绘制逐时 `PV` 与 `Load`，再用绿 / 红填色按**每小时**展示盈余 / 缺口
   - 这张图用于看清楚“日视图掩盖的夜间缺口”和“连续阴雨天的多日缺口”

**终端输出示例**：

```text
Annual PV: 110,888.0 kWh | Annual load: 60,461.0 kWh | Net (PV-load): +50,427.0 kWh
Days with PV > load: 312 | Days with PV < load: 53
```

---

## 5. 关键结果

| 指标 | 数值 | 备注 |
|---|---|---|
| 等效装机容量 | **≈ 79 kWp** | 四朝向 HJT + CdTe + 甲板 + 栏杆 |
| 有效光伏面积 | **420 m²** | 球面 290 + 甲板 67.6 + 栏杆 62.4 |
| 年光伏发电 | **110,888 kWh** | 物理公式直接积分 |
| 年建筑负荷 | **60,461 kWh** | 分项模型校准到设计院口径 |
| 年净能量 | **+50,427 kWh** | 未调度、未扣储能损耗 |
| PV > 负荷的天数 | **312 天** | 全年 365 天 |
| PV < 负荷的天数 | **53 天** | 主要集中在 6–10 月的阴雨段 |
| 设计院自持率口径 | **91.93 %** | 不是本模型直接产物，见 §6 |

---

## 6. 讨论：与设计院口径的差异

### 6.1 光伏年发电量：物理模型 ≈ 2 × 设计院数

- **物理模型**：110,888 kWh
- **设计院**：55,580 kWh

这是 ~2 倍差。可能原因（按可能性排序）：

1. **设计院给的数是“自持供电量”或反推值**，不是光伏毛发电量：
   - `60,461.1 × 91.93% = 55,581.9 kWh`，几乎等于 55,580，吻合度过高，难以偶然。
2. **设计院在发电量里已经隐含了 ~0.5 的综合折减**（遮阴、BOS、逆变器、弃电等），但文档没明写。
3. **面积口径不同**：设计院另有一页写“可布置光伏面积 640 m²”，和 PDF 里的 420 m² 明显是两套口径。

结论：**当前物理口径结果更适合作为学术分析的基准**；55,580 这个数可作为“工程汇报口径”并列列出，但不应当作“真值”去反证模型。

### 6.2 负荷年电量：硬校准到 60,461 kWh

- 旧版 `load_generation.py` 用 “一个总基准功率 × 日程系数” 算出 168 MWh，与设计院差 ~3 倍，口径明显不对。
- 现在的 `load_generation.py` 改为**分项模型**，并**各自缩放**到目标电量，结果精确命中 60,461 kWh，HVAC 占比 60%。
- 需要记住：这个“吻合”是**构造出来的**，真正的验证需要实测账单。目前仅能说分项形状是**合理的工程假设**。

### 6.3 日级绿 ≠ 小时级够

`pv_load_daily_compare.png` 看起来 **312 天盈余**，似乎“覆盖得很好”。但 `pv_load_weekly_hourly_overlay.png` 显示：

- 每个自然日夜间几乎都存在缺口（红色）：PV → 0，但 `base + aux` 仍在；
- 连续阴雨日如果出现，会造成**跨日缺口**，靠日级净能量是看不出来的。

这是后续调度（`dispatch.py`）真正要处理的问题，也是为什么项目不能只到 §5 就结束。

---

## 7. 当前模型局限与待改进

| 局限 | 说明 |
|---|---|
| 负荷曲线过于规整 | 工作日 / 周末各自是严格方波，没有随机性、没有会议 / 活动 / 设备轮换扰动 |
| HVAC 月度权重是**阶梯**切换 | 4 月 0.30 → 5 月 0.70 是阶跃，不随室外温度连续过渡 |
| 年总量吻合是**构造**的 | 缩放系数保证命中，不是独立校验 |
| 光伏温度修正只用 `T2m` | 未使用组件温度模型（NOCT/Wind 修正） |
| 水面微气候用单一常数 | 反照率 1.05 / 冷却 1.012 是中值；实际随季节变化 |
| `pvgis_top.csv` 与 `pvgis_deck.csv` 当前相同 | 若日后补充真实 “顶面” 辐照，需要在 `pv_generation.py` 中新增顶面分区 |

---

## 8. 开发进度

- [x] PVGIS 数据解析与预处理（含末尾脚注行与朝向一致性校验）
- [x] 多朝向 BIPV 发电量计算（面积 / 效率 / 温度 / 微气候）
- [x] 分项建筑负荷建模（base / HVAC / aux）
- [x] 设计院口径校准（60,461 kWh；60/30/10）
- [x] 节假日与调休处理（`holidays` + 国务院官方表）
- [x] 未调度 PV-负荷对比（日级 + 小时级周叠加）
- [x] 工作日志机制（`WORKLOG.md`）
- [ ] 储能系统规则控制调度（`dispatch.py` - RBC）
- [ ] 储能系统 MILP 最优调度（`dispatch.py` - MILP）
- [ ] 储能系统 MPC 滚动调度（`dispatch.py` - MPC）
- [ ] 孤岛情景与续航时长分析
- [ ] 水面微气候灵敏度分析（`sensitivity.py`）
- [ ] 经济性评估（LCOE / 回收期）

---

## 9. Git 与版本控制

本项目用普通 Git 流程管理，推送到 GitHub 做备份。

推荐的 `.gitignore`（最小实用）：

```gitignore
__pycache__/
*.pyc
.venv/
venv/
.vscode/
.DS_Store
Thumbs.db
```

日常流程：

```powershell
git add .
git commit -m "说明这次改了什么"
git push
```

每次有实质性改动，**同时**更新 `WORKLOG.md` 里的对应条目；提交信息写一句话摘要即可，细节放日志里。

---

## 10. 参考

- **PDF 任务书**：`Cursor_workflow.pdf`（中文整理版 `Cursor_workflow.md`）
- **气象数据**：[PVGIS - JRC European Commission](https://joint-research-centre.ec.europa.eu/pvgis-photovoltaic-geographical-information-system_en)
- **负荷日程系数**：ASHRAE 90.1 Energy Standard（PDF 中给出具体数值）
- **节假日**：
  - `holidays` Python 包
  - 国务院办公厅《关于 2023 年部分节假日安排的通知》
- **设计院资料**：
  - 《能源自持方面总结》（年用电 60,461.1 kWh、自持率 91.93%）
  - 《浮体光伏分布分析》（有效发电面积 420 m²）
  - 《能源自持（太阳能发电 + 风力发电）》（日用电 390.7 kWh、空调占比近 60%、负荷计算书）

---

## License

MIT License
