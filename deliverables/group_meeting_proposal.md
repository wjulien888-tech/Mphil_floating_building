# 南沙水上浮式零碳建筑：光伏 / 负荷 / 储能调度与离岛韧性研究

## 组会汇报 Proposal — PPT 制作 Outline

阶段：MPhil 阶段汇报（基准已建立 + 调度路线规划）
数据年：2023 年 8760 h
研究对象：HKUST(GZ) Waterfront Living Lab 浮体零碳建筑
撰写依据：本仓库 `README.md`、`research_plan.md`、`Cursor_workflow.md`、`data_request.md`、`WORKLOG.md`

---

## 0. PPT 总体说明

- 建议总页数：22 页（封面 + 19 页正文 + 讨论 + 致谢）
- 单页结构：左文右图或上结论下证据；正文以条目陈述为主，避免长段落
- 配色语义：蓝（光伏）/ 橙（负荷）/ 绿（盈余）/ 红（缺口）/ 紫（电池）
- 字体：标题 28–32 pt，正文 18–22 pt，图注 14 pt
- 公式排版：建议使用 LaTeX 渲染或 PowerPoint 公式编辑器，避免位图截图
- 图源路径：所有已生成图位于 `plots/`，脚本位于项目根目录

---

# 一、PPT 逐页 Outline

---

## Slide 1 — 封面

**标题**：南沙水上浮式零碳建筑 —— 光伏 / 负荷 / 储能调度与离岛韧性研究

**副标题**：基于 PVGIS 基准与设计院负荷口径的 8760 h 仿真与调度规划

**信息块**：

- 研究对象：HKUST(GZ) Waterfront Living Lab 浮体零碳建筑
- 阶段：MPhil 阶段汇报
- 汇报人 / 导师 / 日期：（占位）

**视觉**：浮体建筑示意图或球体 BIPV 渲染图

---

## Slide 2 — 研究背景

**本页内容**：浮体建筑相对陆上建筑在能源系统上的四点结构性差异。

1. **水面微气候**：水体反照率高于一般地表，水冷效应可降低组件温度，使同等辐照下出力增加
2. **多朝向 BIPV**：球体表面在同一时刻面对不同方位，需按朝向分别积分逐时辐照
3. **离岛运行需求**：浮体可与母岸断开，需具备孤岛自持能力，调度策略对系统可靠性的作用被放大
4. **设计院口径与物理口径不一致**：设计院给出"年发电 55,580 kWh / 年耗电 60,461 kWh / 自持率 91.93%"通常已包含综合折减或反推关系，与纯物理积分结果不直接可比

**插图**：

- 左：项目位置示意（南沙坐标 22.896°N, 113.480°E）
- 右：浮体平台分区示意（球体约 290 m² + 外圈甲板约 130 m²）

**关键参数表**：


| 项      | 数值       |
| ------ | -------- |
| 球面     | 约 290 m² |
| 浮体平台   | 约 130 m² |
| 等效装机容量 | 约 79 kWp |
| 有效光伏面积 | 420 m²   |


---

## Slide 3 — 研究目标与核心问题

**研究目标**：

1. 在 2023 年 PVGIS 辐照与设计院负荷口径下，建立 8760 h 光伏-负荷基准
2. 在统一时间轴上比较 RBC / MILP / MPC 三种储能调度策略
3. 在极端天气与离岛窗口下评估系统韧性

**核心研究问题**：

- Q1：在未接入储能调度的条件下，光伏与负荷在逐日 / 逐时尺度上是否匹配？缺口集中于何时？
- Q2：在给定光伏出力与负荷条件下，何种调度策略能在并网与离岛运行下降低购电、控制缺电、应对极端天气？

**研究路线流程图**：

```
PVGIS 辐照 ─┐
            ├─→ pv_generation.py ─→ pv_kW(t)  ─┐
设计参数  ─┘                                     │
                                                 ├─→ dispatch.py ─→ 离岛韧性分析
设计院负荷 ─→ load_generation.py ─→ load_kW(t) ─┘
```

---

## Slide 4 — 数据来源

**本页内容**：光伏与负荷分别独立建链，仅在调度脚本中合并。

**数据来源对照表**：


| 数据项      | 来源                                 | 处理脚本                            | 输出                              |
| -------- | ---------------------------------- | ------------------------------- | ------------------------------- |
| 光伏出力（基准） | PVGIS ERA5 + 设计参数（面积/效率/温度修正/水面系数） | `pv_generation.py`              | `output/pv_generation_8760.csv` |
| 实测 PV 参照 | 能源中心 / 数据中心实测 PV（待申请）              | `pv_scenario_generation.py`（规划） | 扰动系数 / 情景 CSV                   |
| 建筑负荷     | 设计院年电量口径 + ASHRAE 日程 + 中国节假日/调休    | `load_generation.py`            | `output/load_8760.csv`          |
| 调度输入     | 上述逐时 `pv_kW`、`load_kW`             | `dispatch.py`（规划）               | `output/results_comparison.csv` |


**情景合成关系**：

$$PV_{scenario}(t) = PV_{PVGIS}^{floating}(t) \cdot f_{disturbance}(t)$$

其中 $f_{disturbance}(t) = P_{measured}(t) / P_{ref}(t)$ 由实测 PV 归一化曲线得到。

**插图**：数据来源流程图（PVGIS / 实测 / 设计院 三入一出至调度）

---

## Slide 5 — 光伏建模：分区与参数

**本页内容**：BIPV 共划分 6 个分区（球体 4 朝向 + 水平甲板 + 玻璃栏杆），按面积、组件类型与效率分别计算。

**分区参数表**：


| 分区               | 面积 (m²)     | 组件类型 | 效率   |
| ---------------- | ----------- | ---- | ---- |
| 球体南向（不透明 HJT）    | 62.5        | HJT  | 0.21 |
| 球体东向（不透明 HJT）    | 62.5        | HJT  | 0.21 |
| 球体西向（不透明 HJT）    | 62.5        | HJT  | 0.21 |
| 球体北向（不透明 HJT）    | 62.5        | HJT  | 0.21 |
| 球体 4 朝向（透明 CdTe） | 4 × 10 = 40 | CdTe | 0.12 |
| 外圈浮体甲板（水平）       | 67.6        | HJT  | 0.21 |
| 玻璃栏杆             | 62.4        | CdTe | 0.12 |


**等效装机容量**：

$$P_{cap} = 250 \times 0.21 + 40 \times 0.12 + 67.6 \times 0.21 + 62.4 \times 0.12 \approx 79\ \text{kWp}$$

**插图**：球体四朝向 + 浮体平台 BIPV 分区等距示意（建议自绘或抠自设计院 PDF）

---

## Slide 6 — 光伏建模：物理公式

**本页内容**：单表面 BIPV 出力公式与水面微气候修正。

**单表面逐时计算**：

$$G_{total} = G_b(i) + G_d(i) + G_r(i) \quad [W/m^2]$$

$$\eta_{T} = 1 + \alpha \cdot (T_{2m} - 25),\quad \alpha = -0.004$$

$$P_{kW} = \max\left(0,\ \frac{G_{total} \cdot A \cdot \eta \cdot \eta_T}{1000}\right)$$

**水面微气候修正**（在合成功率层级统一应用）：

$$P_{corr} = P \cdot k_{albedo} \cdot k_{cooling}$$

$$k_{albedo} = 1.05,\quad k_{cooling} = 1.012$$

文献区间：水面反照率修正区间 0.06–0.15；温降区间 2–5 °C；当前取中值。

**数据质量处理**：

- PVGIS CSV 跳过前 8 行元数据，末尾说明行通过 `to_datetime(errors="coerce")` 剔除
- 校验 4 朝向 datetime 严格一致；校验 `pvgis_top.csv` 与 `pvgis_deck.csv` 的 MD5 一致，水平面辐照仅计入一次以避免重复累加

**插图**：辐照三分量 $G_b + G_d + G_r$ 原理示意

---

## Slide 7 — 光伏建模：结果

**本页内容**：年光伏发电基准结果及全年时序分布。

**关键结果**：


| 指标      | 数值                 |
| ------- | ------------------ |
| 等效装机容量  | 约 79 kWp           |
| 有效光伏面积  | 420 m²             |
| 年光伏发电   | 110,888 kWh        |
| 设计院汇报口径 | 55,580 kWh         |
| 差额      | +55,308 kWh（约 2 倍） |


**插图**（建议 2×2 网格）：

- 左上：`plots/pv_monthly_kwh.png` — 月能量柱状图
- 右上：`plots/pv_daily_kwh.png` — 日能量折线
- 左下：`plots/pv_daily_full_year_bar.png` — 全年逐日柱（按月着色）
- 右下：`plots/pv_daily_full_year_heatmap.png` — 全年日历热力图

**结果观察**：

- 6–8 月日发电峰值约 500 kWh/d
- 12–1 月最低，但因多朝向积分仍保有显著贡献
- 热力图显示梅雨期（5–6 月）与台风季（7–9 月）存在低出力日

---

## Slide 8 — 负荷建模：分项模型

**本页内容**：将设计院年总量 60,461 kWh 拆分为三分项，按 ASHRAE 日程系数分别建立形状并独立缩放。

**模型形式**：

$$Load_{kW}(t) = Base_{kW}(t) + HVAC_{kW}(t) + Aux_{kW}(t)$$

**分项参数**：


| 分项        | 内容             | 年电量占比 | 季节性    | 形状              |
| --------- | -------------- | ----- | ------ | --------------- |
| `base_kW` | 照明 / 普通插座 / 机房 | 30 %  | 全年     | ASHRAE 工作日 / 周末 |
| `hvac_kW` | 空调             | 60 %  | 4–10 月 | 日内白天为主 × 月度权重   |
| `aux_kW`  | 水泵 / 室外照明      | 10 %  | 全年     | 傍晚 / 夜间较重       |


**HVAC 月度权重（南沙口径）**：

$$w_{HVAC} = 1:0,\ 2:0,\ 3:0,\ 4:0.30,\ 5:0.70,\ 6:0.95,\ 7:1.00,\ 8:1.00,\ 9:0.85,\ 10:0.40,\ 11:0,\ 12:0$$

**节假日规则**：以 `holidays.country_holidays('CN', years=2023)` 与国务院 2023 年放假表取并集；MAKEUP 工作日强制走工作日形状。

**逐分项缩放公式**（保证年总量精确命中目标）：

$$factor = \frac{E^{target}_{annual}}{\sum_t shape(t) \cdot \Delta t},\quad scaled(t) = shape(t) \cdot factor$$

---

## Slide 9 — 负荷建模：结果

**本页内容**：全年负荷分项校准结果。

**结果数据**：


| 分项     | 年电量              | 占比     |
| ------ | ---------------- | ------ |
| Base   | 18,138.3 kWh     | 30.0 % |
| HVAC   | 36,276.6 kWh     | 60.0 % |
| Aux    | 6,046.1 kWh      | 10.0 % |
| **总计** | **60,461.0 kWh** | 100 %  |


**插图**：

- 主图：`plots/load_monthly_kwh.png` — 月度三分量堆叠柱
- 副图 1：`plots/load_typical_day_stack.png` — 冬季日（2023-01-18）+ 夏季日（2023-07-19）分项叠加
- 副图 2：`plots/load_daily_full_year_heatmap.png` — 负荷年日历热力图

**结果观察**：

- 冬季典型日仅 base + aux，HVAC 为零
- 夏季典型日 HVAC 在白天占主导
- 周末与节假日 base 形状显著低于工作日
- 注意：年总量吻合是构造结果（缩放命中），并非独立验证；后续 Slide 12 单列说明

---

## Slide 10 — 未调度供需对比（实验 E0 Baseline）

**本页内容**：在未接入储能调度的条件下，光伏与负荷在日级与小时级的对比结果。

**关键结果**：


| 指标              | 数值                       |
| --------------- | ------------------------ |
| 年 PV 发电         | 110,888 kWh              |
| 年负荷             | 60,461 kWh               |
| 年净能量（PV − Load） | +50,427 kWh              |
| 日级 PV > Load    | 312 天                    |
| 日级 PV < Load    | 53 天（集中于 6–10 月梅雨 / 台风段） |


**插图**：

- 上：`plots/pv_load_daily_compare.png` — 全年逐日 PV vs Load（绿盈余 / 红缺口）+ 净能量柱
- 下：`plots/pv_load_weekly_hourly_overlay.png` — 典型周（2023-07-17 至 23）小时级叠加

**结果观察**：

- 日级视角下全年 312 天 PV 多于 Load，盈余明显
- 小时级视角下夜间 PV 趋零而 base + aux 仍存在，每日均存在缺口窗口
- 上下两图配合可揭示日级聚合掩盖了小时级缺失

---

## Slide 11 — 核心结论：日级聚合不能替代小时级匹配

**本页内容**：基于 E0 Baseline 的三条主要观察。

1. 年总量层面盈余明显（+50 MWh），但小时级仍存在系统性缺口
2. 夜间普遍缺口：每个自然日 0–6 时与 19–23 时段为缺口区
3. 跨日缺口：连续阴雨 / 台风段会形成 24–72 h 连续缺口窗口，对储能容量提出明确约束

**结论**：

仅依据年净能量盈余得出"自持率达标"的结论是有偏的；必须回到小时级 PV-Load 平衡与储能调度框架，才能完整回答系统可靠性与离岛韧性问题。

**插图**：自绘"日级 vs 小时级"对比小图（同一周左侧日级显示全绿，右侧小时级红绿交错）

---

## Slide 12 — 讨论：设计院口径 vs 物理口径

**本页内容**：物理模型与设计院汇报口径之间的差异成因。

**核心数据**：


| 口径   | 年 PV        | 年负荷        | 自持率     |
| ---- | ----------- | ---------- | ------- |
| 物理模型 | 110,888 kWh | 60,461 kWh | —       |
| 设计院  | 55,580 kWh  | 60,461 kWh | 91.93 % |


**关键吻合关系**：

$$60{,}461.1 \times 91.93 = 55{,}581.9\ \text{kWh} \approx 55{,}580$$

**可能成因**（按可能性排序）：

1. 设计院数值对应"自持供电量"或由自持率反推，非光伏毛发电
2. 设计院发电量已隐含约 0.5 的综合折减（遮阴 / BOS / 逆变器 / 弃电），但文档未明示
3. 面积口径不同：设计院另一页给出"可布置 PV 面积 640 m²"，与本研究采用的 420 m² 存在差异

**采用的口径**：

- 学术分析：以物理口径为基准
- 工程汇报：保留设计院口径并列展示，但不作为"真值"反向校核物理模型
- 后续储能调度全部基于物理口径 PV 与分项校准 Load

---

## Slide 13 — 实验矩阵 E0–E5

**本页内容**：实验 E0 至 E5 的输入、运行条件与目的。


| ID  | 名称              | PV 输入           | 负荷          | 运行条件      | 目的             |
| --- | --------------- | --------------- | ----------- | --------- | -------------- |
| E0  | Baseline        | PVGIS 物理模型      | `load_8760` | 未调度       | 已完成；建立年盈余与缺口时段 |
| E1  | LocalPVScenario | E0 × 实测扰动系数     | 同 E0        | 未调度       | 本地波动 PV 情景     |
| E2  | ExtremeWeather  | 实测 PV 提取低出力事件叠加 | 同 E0        | 部分时段离岛    | 压力测试           |
| E3  | RBC             | E0 / E1 / E2    | 同 E0        | 并网 + 可选离岛 | 工程可解释基线        |
| E4  | MILP            | E0 / E1 / E2    | 同 E0        | 并网 + 可选离岛 | 理论最优参照         |
| E5  | MPC             | E0 / E1 / E2    | 同 E0        | 并网 + 可选离岛 | 滚动优化、近实际运行     |


**说明**：极端天气不定义为"整年极端气候"，而是从能源中心 / 数据中心实测 PV 中提取的真实低出力事件窗口（24 h / 48 h / 72 h / 可选 7 d）嵌入全年基准。

**插图**：实验依赖关系示意（E0 → E1 → E2；E0–E2 → E3 / E4 / E5）

---

## Slide 14 — 储能系统参数

**本页内容**：调度模型采用的统一硬件假设。

**系统参数**（来源：`Cursor_workflow.md`）：


| 参数         | 符号          | 数值          |
| ---------- | ----------- | ----------- |
| 电池容量       | $E_{bat}$   | 215 kWh     |
| 最大充 / 放电功率 | $P_{max}$   | 50 kW       |
| 初始 SOC     | —           | 0.5         |
| SOC 下界     | $SOC_{min}$ | 0.1         |
| SOC 上界     | $SOC_{max}$ | 0.9         |
| 回路效率       | $\eta_{rt}$ | 0.92        |
| 购电电价       | $c_{buy}$   | 0.8 CNY/kWh |
| 售电电价       | $c_{sell}$  | 0.4 CNY/kWh |
| 离岛关键负荷     | —           | 13 kW       |


**SOC 跟踪方程**：

$$SOC(t+1) = SOC(t) + \frac{P_{ch}(t) \cdot \eta_{rt}}{E_{bat}} - \frac{P_{dis}(t)}{\eta_{rt} \cdot E_{bat}}$$

**插图**：能量流示意（PV 母线节点 → {Load, Battery, Grid_export}；Grid_import → Load；Battery ↔ Load）

---

## Slide 15 — 调度策略 1：规则控制（RBC）

**本页内容**：RBC 决策逻辑及其在策略集合中的定位。

**决策伪代码**：

```
net(t) = pv_kW(t) - load_kW(t)
if net > 0:
    P_ch(t)     ← min( net, P_max, (SOC_max - SOC) * E_bat / (Δt * η_rt) )
    P_export(t) ← net - P_ch(t)
else:
    deficit     ← -net
    P_dis(t)    ← min( deficit, P_max, (SOC - SOC_min) * E_bat * η_rt / Δt )
    P_import(t) ← deficit - P_dis(t)
```

**特征**：

- 优势：可解释、计算开销低、贴近运维实际
- 局限：未考虑分时电价、未引入前视信息，必非全年最优

**插图**：自绘"一周 SOC + 功率流"示意图（待 `dispatch.py` 完成后填入实际结果）

---

## Slide 16 — 调度策略 2：混合整数线性规划（MILP）

**本页内容**：以全年完美前视为假设的最优调度建模。

**目标函数**（最小化年度净电费）：

$$\min \sum_{t=0}^{8759} \left[ c_{buy} \cdot P_{import}(t) - c_{sell} \cdot P_{export}(t) \right] \cdot \Delta t$$

**关键约束**：

功率平衡：

$$P_{pv}(t) + P_{dis}(t) + P_{import}(t) = P_{load}(t) + P_{ch}(t) + P_{export}(t)$$

SOC 动态：

$$SOC(t+1) = SOC(t) + \frac{P_{ch}(t) \eta_{rt}}{E_{bat}} - \frac{P_{dis}(t)}{\eta_{rt} E_{bat}}$$

边界：

$$SOC_{min} \le SOC(t) \le SOC_{max},\quad 0 \le P_{ch}(t),\ P_{dis}(t) \le P_{max}$$

**实现细节**：

- 使用 `pulp` 调用自带 CBC 求解器
- 全年 8760 h 直接求解规模过大，实施时按月切分求解后拼接
- 假设具备全年完美前视信息，结果作为理论上限，不作为可运行策略

**插图**：MILP 按月分块求解流程图（按月切分 → 各月独立求解 → 时间拼接）

---

## Slide 17 — 调度策略 3：模型预测控制（MPC）

**本页内容**：滚动优化框架及与 MILP 的对比。

**MPC 在每个时刻 $t$ 求解的优化问题**：

$$\min_{P(\cdot)} \sum_{\tau=t}^{t+H-1} \left[ c_{buy} \cdot P_{imp}(\tau) - c_{sell} \cdot P_{exp}(\tau) \right] \cdot \Delta t$$

预测窗口：$H = 24\ h$
执行：仅落地 $P^*(t)$，丢弃 $\tau > t$ 的解；下一时刻 $t+1$ 重新求解。

**关键假设与说明**：

- 短期完美预测：使用真实 PV 与 Load 充当 24 h 预测，作为 MPC 上界估计
- 全年共需滚动求解 8760 个小规模 MILP，使用 `tqdm` 监控进度
- MPC 与 MILP 对比预期：经济性 $C_{MPC} \ge C_{MILP}$（MPC 为次优），鲁棒性方面 MPC 对预测误差更敏感且更接近真实运行

---

## Slide 18 — 极端天气与离岛韧性

**本页内容**：极端天气事件窗口的定义、提取方法与离岛韧性指标。

**事件窗口尺度**：


| 尺度      | 场景含义           | 用途            |
| ------- | -------------- | ------------- |
| 24 h    | 单日暴雨 / 强阴天低 PV | 短时离岛与日内电池支撑   |
| 48 h    | 连续两天低辐照        | 跨日 SOC 恢复能力   |
| 72 h    | 连续三天台风 / 暴雨    | 典型极端天气下关键负荷保障 |
| 7 d（可选） | 连续低出力周         | 严苛压力测试        |


**事件提取与情景合成**：

$$f_{disturbance}(t) = \frac{P_{measured}(t)}{P_{capacity}^{measured}}$$

$$PV_{extreme}(t) = PV_{floating}^{PVGIS}(t) \cdot f_{disturbance}(t)$$

**离岛工况设定**：

- 离岛关键负荷：13 kW
- 在事件窗口第 0 h 触发电网断开
- 计算 SOC 跌破 $SOC_{min}$ 之前可维持的小时数 $T_{island}$

**离岛韧性指标**：

$$\eta_{cover} = \frac{\sum_t \min\big(P_{load}(t),\ P_{supply}(t)\big)}{\sum_t P_{load}(t)},\quad T_{island}^{max}$$

**插图**：离岛窗口前后 SOC 演化示意（包含离岛触发时刻、SOC 下降至下限触发停机的关键事件）

---

## Slide 19 — 评价指标体系

**本页内容**：用于策略横向比较的统一指标集。


| 类别  | 指标      | 公式 / 定义                                                |
| --- | ------- | ------------------------------------------------------ |
| 能量  | 年购电量    | $E_{imp} = \sum_t P_{imp}(t)\Delta t$                  |
| 能量  | 年售电量    | $E_{exp} = \sum_t P_{exp}(t)\Delta t$                  |
| 能量  | 弃光量     | $E_{curt} = E_{pv} - E_{used} - E_{exp}$               |
| 能量  | 净电量     | $E_{net} = E_{exp} - E_{imp}$                          |
| 电池  | SOC 轨迹  | 8760 h 时序图                                             |
| 电池  | 循环次数    | $N_{cyc} = \sum_t P_{ch}(t)\Delta t / E_{bat}$         |
| 电池  | 循环深度    | DoD 分布直方图                                              |
| 可靠性 | 缺电量     | $LOL = \sum_t \max(0,\ P_{load} - P_{supply})\Delta t$ |
| 可靠性 | 缺电小时数   | $LOLH$                                                 |
| 可靠性 | 最大未满足负荷 | $\max_t (P_{load} - P_{supply})$                       |
| 离岛  | 覆盖比例    | $\eta_{cover}$                                         |
| 离岛  | 最大可维持时长 | $T_{island}^{max}$                                     |
| 经济性 | 年净电费    | $C_{net} = c_{buy} E_{imp} - c_{sell} E_{exp}$         |
| 经济性 | 自给率     | $SSR = (E_{load} - E_{imp}) / E_{load}$                |
| 经济性 | 自消费率    | $SCR = (E_{pv} - E_{exp}) / E_{pv}$                    |


---

## Slide 20 — 灵敏度分析（水面微气候）

**本页内容**：水面反照率与冷却系数在文献区间内变化时，发电量与系统经济性的响应。

**变量空间**：

$$k_{albedo} \in 1.00,\ 1.03,\ 1.05,\ 1.08,\ 1.10$$

$$k_{cooling} \in 1.000,\ 1.004,\ 1.008,\ 1.012,\ 1.016,\ 1.020$$

共 5 × 6 = 30 组组合。

**分析流程**：

1. 对每组 $(k_{albedo}, k_{cooling})$，按比例缩放基础 `pv_kW`
2. 在更新后的 PV 序列上运行 RBC
3. 记录三项响应：`annual_pv_total`、`self_consumption_rate`、`net_cost`
4. 输出 `output/sensitivity_results.csv`

**插图**：

- 主图：5 × 6 热力图，$netcost$ 关于 $(k_{albedo}, k_{cooling})$
- 副图 1：年 PV 总量 vs $k_{cooling}$（固定 $k_{albedo}$）
- 副图 2：年 PV 总量 vs $k_{albedo}$（固定 $k_{cooling}$）

**讨论方向**：水面系数 ±5% 不确定性对年净电费的传导幅度，及其对水面参数精细化建模优先级的指导意义。

---

## Slide 21 — 当前模型局限与待改进

**本页内容**：本研究当前阶段已识别的建模局限及对应改进方向。


| 局限                                | 说明                      | 改进方向                    |
| --------------------------------- | ----------------------- | ----------------------- |
| 负荷曲线过于规整                          | 工作日 / 周末为严格方波，缺少随机扰动    | 引入会议 / 活动 / 设备轮换扰动；蒙特卡洛 |
| HVAC 月度权重为阶梯切换                    | 4 月 0.30 → 5 月 0.70 阶跃  | 基于 CDD / 室外温度的连续过渡      |
| 年总量为构造性吻合                         | 缩放保证命中，非独立验证            | 待实测电费账单交叉校核             |
| PV 温度修正仅用 T2m                     | 未引入组件温度（NOCT / 风速）模型    | 引入显式组件温度模型              |
| 水面微气候采用单一常数                       | 反照率 1.05 / 冷却 1.012 为中值 | 季节化系数 / 现场实测            |
| `pvgis_top.csv` 与 `deck.csv` 当前相同 | 顶面与甲板暂用同一辐照             | 待获取真实顶面辐照后新增顶面分区        |


---

## Slide 22 — 近期任务排期

**本页内容**：阶段性交付计划。


| 时间       | 任务                        | 交付                             |
| -------- | ------------------------- | ------------------------------ |
| Week 1   | 复核固化 E0 输出 + 撰写脚本测试       | E0 结果表 + 4 张主图                 |
| Week 1–2 | 申请并接入实测 PV 数据             | `pv_scenario_generation.py` 雏形 |
| Week 2   | 实现 `dispatch.py` 的 RBC 模块 | 全年 RBC 运行 + SOC 图              |
| Week 3   | 实现 MILP（按月切分）             | RBC vs MILP 对比表                |
| Week 4   | 实现 MPC（滚动 24 h）           | 三策略统一对比表                       |
| Week 5   | 极端天气事件提取 + 离岛压力测试         | 事件窗口 + 续航小时数                   |
| Week 6   | 灵敏度分析 + 经济性 LCOE / 回收期    | `sensitivity.py` 输出            |


---

## Slide 23 — 组会讨论点

**本页内容**：尚未收敛、需组会讨论决策的开放性问题。

1. 离岛事件窗口是否在 24 / 48 / 72 h 之外加入 12 h 与 7 d 极端周
2. 当前电池参数（215 kWh / 50 kW / 0.92）是否需做容量与功率灵敏度
3. 实测 PV 数据若获批，扰动系数情景是否作为主结果之一报告
4. 调度策略报告优先级：是否优先报告 RBC vs MILP，MPC 推迟到下一阶段
5. 汇报中是否单列一页"设计院口径 vs 物理口径"对照说明

---

## Slide 24 — 致谢与 Q&A

- 致谢：导师组、设计院（数据资料）、HKUST(GZ) Waterfront Living Lab
- 数据来源致谢：PVGIS / JRC European Commission；ASHRAE 90.1 Energy Standard
- Q&A 留白页

---

# 二、PPT 制作辅助清单

## 2.1 图清单

**已存在的图**（直接取自 `plots/`）：

1. `pv_daily_kwh.png` — Slide 7 副图
2. `pv_monthly_kwh.png` — Slide 7 主图
3. `pv_daily_full_year_bar.png` — Slide 7 副图
4. `pv_daily_full_year_heatmap.png` — Slide 7 主图
5. `load_daily_kwh.png` — Slide 9 备选
6. `load_monthly_kwh.png` — Slide 9 主图
7. `load_daily_full_year_bar.png` — Slide 9 备选
8. `load_daily_full_year_heatmap.png` — Slide 9 副图
9. `load_typical_day_stack.png` — Slide 9 副图
10. `pv_load_daily_compare.png` — Slide 10 主图
11. `pv_load_weekly_hourly_overlay.png` — Slide 10 主图

**待自绘 / 待补的图**：

- Slide 1：浮体建筑总览渲染图（可抠自设计院 PDF）
- Slide 2：项目位置 + 浮体平台分区示意
- Slide 3：研究路线流程图
- Slide 4：数据来源流程图
- Slide 5：球体四朝向 + 浮体平台 BIPV 分区等距示意
- Slide 6：辐照三分量 $G_b + G_d + G_r$ 原理示意
- Slide 11：日级 vs 小时级对比小图
- Slide 13：E0–E5 实验依赖关系示意
- Slide 14：能量流示意
- Slide 15：RBC 一周 SOC + 功率流（需待 `dispatch.py` 完成后填入）
- Slide 16：MILP 按月分块求解流程
- Slide 18：离岛 SOC 演化示意
- Slide 20：水面系数 5×6 热力图

## 2.2 核心公式清单（建议 LaTeX 渲染）

1. PV 单表面：$P_{kW} = \max(0,\ G_{total} \cdot A \cdot \eta \cdot \eta_T / 1000)$
2. 温度修正：$\eta_T = 1 + \alpha (T_{2m} - 25),\ \alpha = -0.004$
3. 水面修正：$P_{corr} = P \cdot k_{albedo} \cdot k_{cooling}$
4. 等效装机：$P_{cap} = 250 \times 0.21 + 40 \times 0.12 + 67.6 \times 0.21 + 62.4 \times 0.12 \approx 79\ kWp$
5. 负荷分项：$Load_{kW}(t) = Base_{kW}(t) + HVAC_{kW}(t) + Aux_{kW}(t)$
6. 缩放系数：$factor = E^{target} / \sum_t shape(t)\Delta t$
7. 情景合成：$PV_{scenario}(t) = PV_{PVGIS}(t) \cdot f_{disturbance}(t)$
8. SOC 动态：$SOC(t+1) = SOC(t) + P_{ch}\eta_{rt}/E_{bat} - P_{dis}/(\eta_{rt}E_{bat})$
9. MILP 目标：$\min \sum_t [c_{buy} P_{imp}(t) - c_{sell} P_{exp}(t)]\Delta t$
10. MPC 滚动：$\min_P \sum_{\tau=t}^{t+H-1} [c_{buy} P_{imp}(\tau) - c_{sell} P_{exp}(\tau)]\Delta t,\ H=24$
11. 设计院吻合关系：$60{,}461 \times 91.93 \approx 55{,}580$
12. 自给率：$SSR = (E_{load} - E_{imp}) / E_{load}$
13. 自消费率：$SCR = (E_{pv} - E_{exp}) / E_{pv}$
14. 缺电量：$LOL = \sum_t \max(0,\ P_{load} - P_{supply})\Delta t$
15. 离岛覆盖比例：$\eta_{cover} = \sum_t \min(P_{load}, P_{supply}) / \sum_t P_{load}$

## 2.3 关键数据点

- 等效装机：79 kWp
- 有效光伏面积：420 m²
- 年 PV：110,888 kWh
- 年负荷：60,461 kWh
- 年净能量：+50,427 kWh
- 盈余 / 缺口天数：312 / 53
- 设计院自持率：91.93 %
- 设计院 PV 口径：55,580 kWh
- 电池：215 kWh / 50 kW / 0.92
- 电价：购电 0.8 / 售电 0.4 CNY/kWh
- 离岛关键负荷：13 kW
- 负荷分项占比：30 / 60 / 10 %（base / HVAC / aux）

---

# 三、附录：项目所有 markdown 文件总览


| 文件                                       | 角色                       |
| ---------------------------------------- | ------------------------ |
| `README.md`                              | 全局总览（含模块详解、关键结果、局限）      |
| `research_plan.md`                       | 研究路线、实验矩阵、实施顺序           |
| `Cursor_workflow.md`                     | PDF 任务书的中文整理（建模与调度技术细节）  |
| `data_request.md`                        | 数据来源说明与实测 PV 申请字段        |
| `WORKLOG.md`                             | 时间线工作日志                  |
| `deliverables/group_meeting_slides.md`   | 早期 9 页组会逐页稿              |
| `deliverables/group_meeting_proposal.md` | 本文件（22 页 PPT 详细 outline） |


---

*Generated 2026-05-05 — 基于本仓库所有 markdown 与 plots 内容综合整理*