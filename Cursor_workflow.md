## 项目背景

本项目围绕 HKUST(GZ) Waterfront Living Lab 漂浮零碳建筑，评估不同能源调度策略的表现。

需要生成 2023 全年共 `8760` 小时的两组时序数据：

1. 来自 PVGIS CSV 文件的逐时光伏发电功率（`kW`）
2. 基于 ASHRAE 90.1 日程系数生成的逐时建筑负荷功率（`kW`）

随后比较三种调度策略：`RBC`、`MILP` 和 `MPC`。

## 任务 1：处理 PVGIS 数据并生成光伏发电曲线

目标脚本：`pv_generation.py`

### 输入文件

2023 年 PVGIS ERA5 数据文件：

- `pvgis_south_45.csv`：倾角 45 度，方位角 0 度
- `pvgis_east_45.csv`：倾角 45 度，方位角 -90 度
- `pvgis_west_45.csv`：倾角 45 度，方位角 90 度
- `pvgis_north_45.csv`：倾角 45 度，方位角 180 度
- `pvgis_deck.csv`：倾角 0 度，方位角 0 度

### PVGIS CSV 格式

- 跳过前 8 行元数据
- 列名为：`time`、`Gb(i)`、`Gd(i)`、`Gr(i)`、`H_sun`、`T2m`、`WS10m`、`Int`
- 时间格式为 `YYYYMMDD:HHMM`，例如 `20230101:0030`

字段说明：

- `Gb(i)`：直射辐照度，单位 `W/m^2`
- `Gd(i)`：散射辐照度，单位 `W/m^2`
- `Gr(i)`：反射辐照度，单位 `W/m^2`
- `T2m`：环境温度，单位摄氏度
- `G_total = Gb(i) + Gd(i) + Gr(i)`

### 建筑 BIPV 参数

球体表面：

- 球体总表面积：`290 m^2`
- 四个朝向平均分配：每个朝向 `72.5 m^2`
- 不透明 HJT 组件总面积：`250 m^2`，效率 `0.21`
- 透明 CdTe 组件总面积：`40 m^2`，效率 `0.12`
- 每个朝向不透明面积比例：`250 / 290 = 0.862`
- 每个朝向透明面积比例：`40 / 290 = 0.138`

水平面，使用 `pvgis_deck.csv`：

- 外圈漂浮平台：面积 `67.6 m^2`，效率 `0.21`
- 玻璃栏杆：面积 `62.4 m^2`，效率 `0.12`

### 光伏计算公式

每个小时按以下公式计算：

```python
G_total = Gb(i) + Gd(i) + Gr(i)
temp_factor = 1 + (-0.004) * (T2m - 25)
power = G_total * area * efficiency * temp_factor / 1000
power = max(0, power)
```

### 水面微气候修正

对所有分区统一应用：

```python
albedo_factor = 1.05
cooling_factor = 1.012
corrected_power = power * albedo_factor * cooling_factor
```

这些系数代表文献中的中间取值范围：

- 水体反照率修正：`0.06` 到 `0.15`
- 温降幅度：`2` 到 `5 deg C`

### 输出要求

- 汇总所有分区，得到总逐时光伏输出
- 保存为 `output/pv_generation_8760.csv`
- 列名为：`datetime`、`pv_kW`
- 输出全年总发电量，并与设计值 `55,580 kWh` 比较
- 绘制日尺度和月尺度发电图用于校核

## 任务 2：生成建筑负荷曲线

目标脚本：`load_generation.py`

### 已知参数

- 开空调时总功率：`47.7 kW`，全年运行 `896 h`
- 不开空调时总功率：`26.1 kW`，全年运行 `679 h`
- HVAC 功率：`21.6 kW`
- 基础负荷（照明、插座、泵）：`26.1 kW`
- 年总耗电量：`60,461 kWh`

### 季节分配

按南沙气候条件：

- 空调开启月份：5 月到 9 月，即 `5-9`，基础功率取 `47.7 kW`
- 空调关闭月份：10 月到次年 4 月，即 `10-4`，基础功率取 `26.1 kW`

### ASHRAE 90.1 日程系数

工作日与周末/节假日使用不同的逐时系数。

工作日 `0-23` 时段系数：

```python
hour_coefficients_weekday = [
    0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
    0.50, 0.70, 1.00, 1.00, 1.00, 0.80,
    1.00, 1.00, 1.00, 1.00, 0.80, 0.60,
    0.50, 0.40, 0.35, 0.30, 0.30, 0.30,
]
```

周末或节假日 `0-23` 时段系数：

```python
hour_coefficients_weekend = [
    0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
    0.25, 0.30, 0.40, 0.40, 0.40, 0.35,
    0.40, 0.40, 0.40, 0.35, 0.30, 0.25,
    0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
]
```

### 负荷计算

对 2023 年的每一个小时：

1. 若月份属于 `5-9`，则设 `base_power = 47.7`，否则取 `26.1`
2. 选择工作日或周末/节假日系数数组
3. 按下式计算：

```python
load_kW = base_power * hour_coefficient[hour_of_day]
```

### 输出要求

- 保存为 `output/load_8760.csv`
- 列名为：`datetime`、`load_kW`
- 输出全年总负荷，并与设计值 `60,461 kWh` 比较
- 绘制日尺度和月尺度负荷图

## 任务 3：调度策略对比

目标脚本：`dispatch.py`

### 系统参数

- 电池容量：`215 kWh`
- 最大充放电功率：`50 kW`
- 初始 SOC：`0.5`
- 最低 SOC：`0.1`
- 最高 SOC：`0.9`
- 回路效率：`0.92`
- 电网默认始终可用，孤岛场景除外
- 购电电价：`0.8 CNY/kWh`
- 售电电价：`0.4 CNY/kWh`

### 输入文件

- `output/pv_generation_8760.csv`
- `output/load_8760.csv`

两者必须使用同一 `datetime` 索引，并且都有 `8760` 行。

### 策略 1：规则控制 `RBC`

每个小时按以下逻辑运行：

```python
net = pv_kW - load_kW
if net > 0:
    charge battery up to max SOC and max charge rate
    export remaining surplus to grid
else:
    discharge battery down to min SOC and max discharge rate
    import remaining deficit from grid
```

### 策略 2：最优调度 `MILP`

- 使用 `pulp`
- 每小时的决策变量包括：
  - `charge(t)`
  - `discharge(t)`
  - `grid_import(t)`
  - `grid_export(t)`
- 目标函数：最小化全年购电成本减去售电收益
- 约束包括：
  - SOC 动态约束
  - 容量约束
  - 充放电功率约束
  - 功率平衡约束
- 假设具备全年完美前视信息
- 使用 PuLP 自带的 CBC 求解器
- 对 `8760` 小时问题按月求解，再拼接全年结果

### 策略 3：模型预测控制 `MPC`

- 滚动预测窗口：`24` 小时
- 在每个时刻 `t`，求解未来 `24 h` 的 MILP
- 只执行第一个时刻的决策
- 然后滚动到 `t + 1` 继续求解
- 将真实 PV 和负荷视为完美短期预测
- 全年共需滚动求解 `8760` 个小规模 MILP

### 评价指标

对每种策略计算：

1. 全年购电成本：`sum(grid_import * 0.8)`
2. 全年售电收益：`sum(grid_export * 0.4)`
3. 全年净成本：购电成本减去售电收益
4. 光伏自消费率
5. 电池循环次数：`total_charge_kWh / 215`
6. 能源自给率：`(load - grid_import) / load * 100`

### 孤岛工况

- 孤岛负荷定义为 `13 kW`
- 在某个典型日的第 0 小时电网断开
- 对每种策略，计算电池跌破最低 SOC 前可维持的时长
- 测试三类典型日：
  - 最优日
  - 最差日
  - 平均日
- 从 `pv_generation_8760.csv` 的逐日总发电量中选取这些天

### 输出要求

- `output/results_comparison.csv`
- 图表包括：
  - 各策略一周 SOC 与功率流示例图
  - 月度成本对比柱状图
  - 孤岛续航时长柱状图

## 任务 4：水面微气候灵敏度分析

目标脚本：`sensitivity.py`

### 待变化参数

```python
albedo_factor = [1.00, 1.03, 1.05, 1.08, 1.10]
cooling_factor = [1.000, 1.004, 1.008, 1.012, 1.016, 1.020]
```

### 分析流程

对每一组 `(albedo_factor, cooling_factor)` 组合：

1. 按比例缩放基础 `pv_kW`，重新计算光伏发电序列
2. 运行 `RBC`
3. 记录：
   - `annual_pv_total`
   - `self_consumption_rate`
   - `net_cost`

### 输出要求

- `output/sensitivity_results.csv`
- 绘制 `net_cost` 相对于 `(albedo_factor, cooling_factor)` 的热力图
- 总结发电量波动范围及其对经济性的影响

## 预期目录结构

```text
project/
├── data/
│   ├── pvgis_south_45.csv
│   ├── pvgis_east_45.csv
│   ├── pvgis_west_45.csv
│   ├── pvgis_north_45.csv
│   └── pvgis_deck.csv
├── pv_generation.py
├── load_generation.py
├── dispatch.py
├── sensitivity.py
├── output/
│   ├── pv_generation_8760.csv
│   ├── load_8760.csv
│   ├── results_comparison.csv
│   └── sensitivity_results.csv
└── plots/
```

## 所需依赖

```bash
pip install pandas numpy matplotlib pulp scipy
```

## 重要说明

1. PV 和负荷的时间对齐必须统一使用 pandas 的 datetime。
2. PVGIS CSV 文件前部带有元数据，需跳过直到以 `time` 开头的表头行为止。
3. `8760` 小时的 MILP 建议按月求解，再拼接结果。
4. `8760` 次滚动 MPC 会比较慢，需要加入 `tqdm` 进度条。
5. 所有功率单位均为 `kW`，所有能量单位均为 `kWh`。
6. SOC 跟踪公式如下：

```python
SOC(t+1) = SOC(t) + charge(t) * eff / 215 - discharge(t) / (eff * 215)
```

7. 每个脚本结束时都应打印关键校核结果。
