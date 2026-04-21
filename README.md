# MPhil Floating Building - 水上建筑能源分析

MPhil研究项目，分析水上浮动建筑的光伏发电与建筑负荷匹配。

## 项目概述

**研究对象**：水上浮动建筑（球形主体 + 甲板）

**研究内容**：光伏发电量计算、建筑负荷建模、供需匹配分析

**数据年份**：2023年（8760小时逐时数据）

**数据来源**：PVGIS气象数据，坐标：22.896°N, 113.480°E（南沙）

## 项目结构

```
Mphil_floating_building/
├── data/                    # PVGIS原始数据
│   ├── pvgis_south_45.csv   # 南向45°倾斜
│   ├── pvgis_east_45.csv    # 东向45°倾斜
│   ├── pvgis_west_45.csv    # 西向45°倾斜
│   ├── pvgis_north_45.csv   # 北向45°倾斜
│   ├── pvgis_deck.csv       # 水平面（甲板）
│   └── pvgis_top.csv        # 水平面（顶部）
├── output/                  # 输出数据
│   ├── pv_generation_8760.csv   # 全年逐时光伏发电
│   └── load_8760.csv            # 全年逐时负荷
├── plots/                   # 可视化图表
├── pv_generation.py         # 光伏发电计算
├── load_generation.py       # 负荷建模
├── pv_load_daily_compare.py # PV与负荷对比
├── WORKLOG.md               # 工作日志
├── Cursor_workflow.pdf      # 工作流程文档
└── requirements.txt         # 依赖清单
```

## 开发状态

- [x] 光伏发电计算（多朝向BIPV + 水面微气候修正）
- [x] 建筑负荷建模（三分项模型，校准到设计院口径）
- [x] PV-负荷逐日对比可视化
- [x] 工作日志机制
- [ ] 储能调度优化
- [ ] 能源自持率分析

## 模块详解

### 1. pv_generation.py - 光伏发电计算

**建筑光伏配置**：
- 球形建筑4个朝向（东南西北45°倾斜）
- 水平甲板 + 玻璃栏杆

**技术路线**：
- HJT（不透明区域，效率21%）
- CdTe（透明区域，效率12%）

**水面微气候修正**：
- 反照率系数：1.05
- 冷却系数：1.012

**输出结果**：年发电量约 **110,888 kWh**

### 2. load_generation.py - 建筑负荷建模

**三分项模型**：
- 基础负荷（照明、插座、机房）：30%
- HVAC（空调）：60%
- 附属负荷（水泵、室外照明）：10%

**标定口径**：全年 **60,461 kWh**（与设计院资料一致）

**HVAC季节性**：
- 运行月份：4-10月
- 月度强度调制：4月0.30 → 7-8月1.00 → 10月0.40

**节假日规则**：中国法定假日 + 国务院2023年调休表

### 3. pv_load_daily_compare.py - 供需对比

**输出图表**：
- 全年逐日PV vs Load曲线
- 净电量柱状图（盈余/缺口）
- 典型周小时级叠加图

## 关键结果

| 指标 | 数值 |
|------|------|
| 年光伏发电量 | 110,888 kWh |
| 年建筑负荷 | 60,461 kWh |
| 净盈余 | ~50,000 kWh |

## 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 运行光伏计算
python pv_generation.py

# 运行负荷建模
python load_generation.py

# 生成对比图
python pv_load_daily_compare.py
```

## 后续计划

1. 储能系统调度优化
2. 能源自持率与经济性分析
3. 水面微气候影响深化研究

## 参考文献

- 数据来源：[PVGIS](https://joint-research-centre.ec.europa.eu/pvgis-photovoltaic-geographical-information-system_en)
- 负荷建模参考：ASHRAE 90.1

## License

MIT License
