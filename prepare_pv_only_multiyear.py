"""Construct and freeze 2021-2023 PV-only, 96-quarter-hours-to-one data.

Run with external/ol-ems-hkust/.venv/bin/python prepare_pv_only_multiyear.py.
Original building equations and causal HKUST disturbance functions are reused.
The output directory is write-once; this script does not train either model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import pv_generation as physical
from align_hkust_pv_15min import DEFAULT_SITES, PV_SITE_DIR, build_measured_pv_15min
from build_pv_disturbance_15min import build_disturbance_factors, build_pvgis_15min_baseline

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'output/frozen/pv_only_2021_2023_96to1_v1_20260920'
SPLITS = {'train': ('2021-07-01', '2023-09-01'),
          'val': ('2023-09-01', '2023-11-01'),
          'test': ('2023-11-01', '2024-01-01')}
LOOKBACK = 96


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def input_paths() -> list[Path]:
    faces = [*physical.TILT_FILES.values(), physical.DECK_FILE]
    return ([ROOT / 'data/pvgis_2021_2022' / str(year) / name for year in (2021, 2022) for name in faces]
            + [ROOT / 'data' / name for name in faces]
            + [PV_SITE_DIR / name for name in DEFAULT_SITES]
            + [ROOT / 'data/pvgis_2021_2022/manifest.json', ROOT / 'output/pv_generation_8760.csv',
               ROOT / 'output/frozen/hkust_m0_v1_20260909/dispatch.csv'])


def build_hourly_building_pv() -> pd.DataFrame:
    parts = []
    for year in (2021, 2022, 2023):
        directory = ROOT / 'data' if year == 2023 else ROOT / 'data/pvgis_2021_2022' / str(year)
        frames = {name: physical.load_pvgis_csv(directory / filename)
                  for name, filename in physical.TILT_FILES.items()}
        deck = physical.load_pvgis_csv(directory / physical.DECK_FILE)
        expected = pd.date_range(f'{year}-01-01 00:30', f'{year}-12-31 23:30', freq='h')
        for frame in [*frames.values(), deck]:
            if not pd.DatetimeIndex(frame.time).equals(expected):
                raise ValueError(f'Incomplete/misaligned hourly source for {year}')
        powers = []
        for frame in frames.values():
            irradiance = physical.total_irradiance_w_m2(frame)
            temperature = frame.T2m.to_numpy(dtype=float)
            powers.append(physical.power_kw_one_surface(irradiance, temperature, physical.SPHERE_OPAQUE_M2, physical.EFF_HJT))
            powers.append(physical.power_kw_one_surface(irradiance, temperature, physical.SPHERE_TRANS_M2, physical.EFF_CDTE))
        irradiance = physical.total_irradiance_w_m2(deck)
        temperature = deck.T2m.to_numpy(dtype=float)
        powers.append(physical.power_kw_one_surface(irradiance, temperature, physical.DECK_M2, physical.EFF_HJT))
        powers.append(physical.power_kw_one_surface(irradiance, temperature, physical.BALU_M2, physical.EFF_CDTE))
        total = np.sum(powers, axis=0) * physical.ALBEDO_FACTOR * physical.COOLING_FACTOR
        parts.append(pd.DataFrame({'datetime_utc_center': expected, 'pv_kW': total}))
    result = pd.concat(parts, ignore_index=True)
    if not np.isfinite(result.pv_kW).all() or (result.pv_kW < 0).any():
        raise ValueError('Invalid building hourly power')
    return result


def align_hourly_baseline(hourly: pd.DataFrame, output_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Local interval-start convention, continuous across calendar years.

    ERA5 hourly center label UTC hh:30 maps to local interval start hh:00+8h.
    Each hour is held over four quarter-hour rows. Missing hours stay invalid;
    unlike the old annual np.roll, the same year's end cannot fill its start.
    """
    times = pd.DatetimeIndex(pd.to_datetime(hourly.datetime_utc_center))
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('Hourly timestamps must be unique and increasing')
    if ((times.minute != 30) | (times.second != 0)).any():
        raise ValueError('Expected PVGIS ERA5 hourly center labels at hh:30')
    local_start = times - pd.Timedelta(minutes=30) + pd.Timedelta(hours=8)
    series = pd.Series(hourly.pv_kW.to_numpy(dtype=float), index=local_start)
    aligned = series.reindex(output_index, method='ffill', tolerance=pd.Timedelta(minutes=45))
    valid = np.isfinite(aligned.to_numpy())
    source_times = pd.Series(times.to_numpy(), index=local_start).reindex(
        output_index, method='ffill', tolerance=pd.Timedelta(minutes=45))
    return pd.DataFrame({'datetime': output_index,
                         'pvgis_source_utc_center': source_times.to_numpy(),
                         'pvgis_floating_baseline_kW': aligned.fillna(0).to_numpy(),
                         'baseline_valid': valid.astype(int)})


def make_samples(frame: pd.DataFrame, splits=SPLITS) -> dict:
    times = pd.DatetimeIndex(pd.to_datetime(frame.datetime))
    if times.has_duplicates or not times.is_monotonic_increasing or not (times[1:] - times[:-1] == pd.Timedelta(minutes=15)).all():
        raise ValueError('Samples require a complete, increasing 15-minute time axis')
    values = frame.pv_kW.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('PV values must be finite and nonnegative, including flagged placeholders')
    for flag in ('history_valid', 'pv_target_valid'):
        if not frame[flag].isin([0, 1]).all():
            raise ValueError(f'{flag} must contain 0/1')
    train_start, train_end = splits['train']
    history_valid = frame.history_valid.to_numpy(dtype=bool)
    target_valid = frame.pv_target_valid.to_numpy(dtype=bool)
    fit = (times >= train_start) & (times < train_end) & history_valid
    if not fit.any():
        raise ValueError('No valid training rows for scaler')
    lo, hi = np.min(values[fit]), np.max(values[fit])
    span = np.float32(hi - lo) if hi > lo else np.float32(1)
    scaled = (values - lo) / span
    scaler = {'pv_min': float(lo), 'pv_max': float(hi), 'scale': float(span),
              'dtype': 'float32', 'feature_names': ['pv'],
              'fit_start_inclusive': str(train_start), 'fit_end_exclusive': str(train_end),
              'fit_valid_rows': int(fit.sum()), 'fit_policy': 'training interval AND history_valid=1',
              'clip': False, 'target_in_model_loss': 'physical kW'}
    bad = np.r_[0, np.cumsum(~history_valid)]
    candidates = np.arange(LOOKBACK, len(frame), dtype=np.int64)
    history_ok = bad[candidates] == bad[candidates - LOOKBACK]
    usable = history_ok & target_valid[candidates]
    all_histories = np.lib.stride_tricks.sliding_window_view(scaled, LOOKBACK)
    arrays, summary, indices = {}, {}, []
    for split, (start, end) in splits.items():
        in_split = (times[candidates] >= start) & (times[candidates] < end)
        anchors = candidates[in_split & usable]
        if not len(anchors):
            raise ValueError(f'No valid samples for {split}')
        rows = (times >= start) & (times < end)
        arrays[split] = {'X': all_histories[anchors - LOOKBACK, :, None].copy(),
                         'y_kw': values[anchors, None, None].copy(),
                         'y_scaled': scaled[anchors, None, None].copy(),
                         'anchor_indices': anchors,
                         'target_timestamps': (times[anchors].asi8 // 10**9).astype(np.int64)}
        indices.append(pd.DataFrame({'split': split, 'anchor_index': anchors,
                                     'history_start': times[anchors - LOOKBACK], 'history_end': times[anchors - 1],
                                     'target_time': times[anchors], 'pv_target_kW': values[anchors]}))
        summary[split] = {'start_inclusive': str(start), 'end_exclusive': str(end),
                          'rows': int(rows.sum()), 'history_valid_rows': int((rows & history_valid).sum()),
                          'pv_target_valid_rows': int((rows & target_valid).sum()),
                          'candidate_windows': int(in_split.sum()), 'valid_windows': len(anchors),
                          'rejected_windows': int(in_split.sum()) - len(anchors),
                          'first_target': str(times[anchors[0]]), 'last_target': str(times[anchors[-1]]),
                          'X_shape': list(arrays[split]['X'].shape), 'y_shape': list(arrays[split]['y_kw'].shape),
                          'target_scaled_above_one': int((scaled[anchors] > 1).sum())}
    return {'arrays': arrays, 'index': pd.concat(indices, ignore_index=True), 'scaler': scaler, 'summary': summary}


def compare_2023(hourly, audit, measured) -> tuple[dict, pd.DataFrame]:
    original = pd.read_csv(ROOT / 'output/frozen/hkust_m0_v1_20260909/dispatch.csv', parse_dates=['datetime'])
    old_hourly = pd.read_csv(ROOT / 'output/pv_generation_8760.csv', parse_dates=['datetime'])
    new_hourly = hourly[hourly.datetime_utc_center.dt.year.eq(2023)]
    hourly_error = float(np.max(np.abs(new_hourly.pv_kW.to_numpy() - old_hourly.pv_kW.to_numpy())))
    if hourly_error > 1e-10:
        raise ValueError(f'Building equations no longer reproduce old 2023 hourly result: {hourly_error}')
    old_baseline = build_pvgis_15min_baseline()
    old_disturbance = build_disturbance_factors(measured[measured.datetime.ge('2022-12-01')], pd.DatetimeIndex(original.datetime))
    old_reconstructed = old_baseline.pvgis_floating_baseline_kW.to_numpy() * old_disturbance.disturbance_factor.to_numpy()
    legacy_error = float(np.max(np.abs(old_reconstructed - original.pv_floating_disturbed_raw_kW.to_numpy())))
    if legacy_error > 1e-9:
        raise ValueError(f'Original 2023 preprocessing cannot be reproduced: {legacy_error}')
    new = audit[audit.datetime.dt.year.eq(2023)].reset_index(drop=True)
    old_values = original.pv_floating_disturbed_raw_kW.to_numpy()
    new_values = new.pv_kW.to_numpy()
    differences = new_values - old_values
    changed = np.abs(differences) > 1e-9
    detail = pd.DataFrame({'datetime': new.datetime, 'old_pv_kW': old_values, 'new_pv_kW': new_values,
                           'difference_kW': differences,
                           'old_disturbance_factor': original.disturbance_factor,
                           'new_disturbance_factor': new.disturbance_factor,
                           'old_peak_threshold_kW': original.clear_baseline_min_kW,
                           'new_peak_threshold_kW': new.clear_baseline_min_kW,
                           'old_history_valid_5_features': original.history_valid,
                           'new_history_valid_PV_only': new.history_valid})
    result = {'old_hourly_reproduction_max_abs_error_kW': hourly_error,
              'old_frozen_labels_reproduction_max_abs_error_kW': legacy_error,
              'new_multiyear_vs_old_2023': {
                  'rows': len(new), 'changed_pv_rows_tolerance_1e_9': int(changed.sum()),
                  'max_abs_pv_difference_kW': float(np.abs(differences).max()),
                  'disturbance_factor_changed_rows': int((np.abs(new.disturbance_factor - original.disturbance_factor) > 1e-9).sum()),
                  'baseline_changed_rows': int((np.abs(new.pvgis_floating_baseline_kW - original.pvgis_floating_baseline_kW) > 1e-9).sum()),
                  'history_rows_recovered_by_PV_only': int(((new.history_valid == 1) & (original.history_valid == 0)).sum()),
                  'target_validity_changed_rows': int((new.pv_target_valid != original.pv_target_valid).sum())},
              'explanation': 'Building formula unchanged. Hourly alignment is continuous across years instead of annual circular roll. The expanding prior observed PV peak now uses all available prior years instead of initialization only from December 2022. PV-only history validity no longer depends on weather input availability.'}
    result['new_multiyear_vs_old_2023']['changed_pv_rows_by_split'] = {
        name: int((changed & new.datetime.ge(start).to_numpy() & new.datetime.lt(end).to_numpy()).sum())
        for name, (start, end) in SPLITS.items()
    }
    return result, detail.loc[changed].reset_index(drop=True)


def freeze(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output version: {output}')
    download_root = ROOT / 'data/pvgis_2021_2022'
    download_manifest = json.loads((download_root / 'manifest.json').read_text())
    for record in download_manifest['files']:
        path = (download_root / record['path']).resolve()
        if not path.is_relative_to(download_root.resolve()) or path.stat().st_size != record['size_bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Downloaded source differs from its manifest: {path}')
    inputs = input_paths()
    before = {str(p.relative_to(ROOT)): sha256(p) for p in inputs}
    print('Building three-year building PV with unchanged physical parameters...', flush=True)
    hourly = build_hourly_building_pv()
    index = pd.date_range('2021-01-01', '2024-01-01', freq='15min', inclusive='left')
    baseline = align_hourly_baseline(hourly, index)
    print('Aligning eight measured sites and constructing causal disturbances...', flush=True)
    measured = build_measured_pv_15min(start='2021-01-01', end='2024-01-01')
    disturbance = build_disturbance_factors(measured, index)
    audit = baseline.merge(disturbance, on='datetime', validate='one_to_one')
    audit['clear_baseline_min_kW'] = audit.clear_baseline_min_kW.fillna(0)
    audit = audit.merge(measured[['datetime', 'pv_imputed_site_count', 'pv_placeholder_site_count']], on='datetime', validate='one_to_one')
    audit['pv_kW'] = audit.pvgis_floating_baseline_kW * audit.disturbance_factor
    audit['history_valid'] = (audit.pv_history_valid.astype(bool) & audit.baseline_valid.astype(bool)).astype(int)
    audit['pv_target_valid'] = (audit.pv_target_valid.astype(bool) & audit.baseline_valid.astype(bool)).astype(int)
    audit['split'] = 'excluded'
    for name, (start, end) in SPLITS.items():
        audit.loc[audit.datetime.ge(start) & audit.datetime.lt(end), 'split'] = name
    frame = audit[['datetime', 'pv_kW', 'history_valid', 'pv_target_valid', 'split']].copy()
    samples = make_samples(frame)
    scaler = samples['scaler']
    frame['pv_scaled'] = (frame.pv_kW.to_numpy(dtype=np.float32) - np.float32(scaler['pv_min'])) / np.float32(scaler['scale'])
    frame['forecast_origin_valid'] = 0
    frame.loc[samples['index'].anchor_index, 'forecast_origin_valid'] = 1
    comparison, differences = compare_2023(hourly, audit, measured)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.building-', dir=output.parent))
    frame.to_csv(staging / 'pv_timeseries.csv', index=False)
    audit.to_csv(staging / 'pv_construction_audit.csv', index=False)
    hourly.to_csv(staging / 'pvgis_building_hourly.csv', index=False)
    samples['index'].to_csv(staging / 'window_index.csv', index=False)
    differences.to_csv(staging / 'changed_2023_rows.csv', index=False)
    for name, arrays in samples['arrays'].items():
        np.savez_compressed(staging / f'{name}.npz', **arrays)
    summary_rows = [{'split': name, **{k: v for k, v in stats.items() if not isinstance(v, list)}} for name, stats in samples['summary'].items()]
    pd.DataFrame(summary_rows).to_csv(staging / 'split_summary.csv', index=False)
    write_json(staging / 'scaler.json', scaler)
    write_json(staging / 'comparison_2023.json', comparison)
    # A full 96-row input and its one label, chosen near noon for readable PV values.
    train_indices = samples['arrays']['train']['anchor_indices']
    choices = train_indices[(frame.datetime.iloc[train_indices].dt.hour.to_numpy() == 12)
                            & (frame.datetime.iloc[train_indices].dt.minute.to_numpy() == 0)]
    anchor = int(choices[0])
    preview = audit.iloc[anchor - LOOKBACK:anchor + 1][['datetime', 'pvgis_floating_baseline_kW', 'disturbance_factor', 'pv_kW', 'history_valid', 'pv_target_valid']].copy()
    preview.insert(0, 'role', ['input'] * LOOKBACK + ['target'])
    preview.insert(1, 'step_relative_to_target', np.arange(-LOOKBACK, 1))
    preview['pv_scaled'] = frame.pv_scaled.iloc[anchor - LOOKBACK:anchor + 1].to_numpy()
    preview.to_csv(staging / 'training_sample_preview.csv', index=False)
    dictionary = [
        ('datetime', '本地15分钟时间标签；延续项目时段约定', 'Asia/Shanghai'),
        ('pv_kW', '构造的浮式建筑PV功率；模型唯一输入和预测目标', 'kW'),
        ('history_valid', '该行可作为历史输入；可含最多4步因果前填', '0/1'),
        ('pv_target_valid', '该行可作标签；8站原始观测完整且扰动基线可用', '0/1'),
        ('split', '按目标时间分配train/val/test；2021年7月前excluded', 'category'),
        ('pv_scaled', '仅训练期有效PV拟合的min-max归一化；不截断', 'dimensionless'),
        ('forecast_origin_valid', '此行可作标签且前96行历史全部有效', '0/1'),
    ]
    pd.DataFrame(dictionary, columns=['column', 'description_zh', 'unit']).to_csv(staging / 'data_dictionary.csv', index=False)
    quality = {'rows': len(frame), 'frequency': '15min',
               'earliest_valid_history': str(frame.loc[frame.history_valid.eq(1), 'datetime'].iloc[0]),
               'baseline_missing_rows': int(audit.baseline_valid.eq(0).sum()),
               'invalid_history_rows': int(frame.history_valid.eq(0).sum()),
               'invalid_target_rows': int(frame.pv_target_valid.eq(0).sum()),
               'raw_missing_rows_during_common_site_coverage': int(audit.loc[audit.datetime.ge('2021-07-01'), 'raw_missing_site_count'].gt(0).sum()),
               'history_rows_with_imputed_site': int((audit.history_valid.eq(1) & audit.pv_imputed_site_count.gt(0)).sum()),
               'annual_valid_rows': {str(year): int((frame.pv_target_valid.eq(1) & frame.datetime.dt.year.eq(year)).sum()) for year in (2021, 2022, 2023)}}
    write_json(staging / 'quality_report.json', quality)
    description = f'''# 2021—2023年PV-only训练数据：96步预测下一15分钟

本目录是构造的浮式建筑PV训练数据，不是建筑实测发电记录。两种模型共用同一冻结数据与划分，尚未开展正式训练。

## 打开哪些文件

- `pv_timeseries.csv`：完整15分钟序列，105,120行；模型读取此文件。
- `training_sample_preview.csv`：一个实际可用样本，前96行是输入，最后1行是预测目标；包含构造前基线、扰动因子和归一化值。
- `window_index.csv`：所有可用预测样本的历史起止、目标时间和原序列行号。
- `train.npz`、`val.npz`、`test.npz`：已造窗的X、y_kw、y_scaled、anchor_indices、target_timestamps；X为[N,96,1]，y_kw为[N,1,1]。时间编码沿用项目本地naive时间转整数约定，不是已转换成UTC的真实Unix时刻。
- `pv_construction_audit.csv`：逐行中间计算和质量标记；`data_dictionary.csv`：训练CSV字段解释。
- `split_summary.csv`、`quality_report.json`、`comparison_2023.json`：划分、质量和2023对照核验。

## 改造步骤

1. 下载的2021/2022和原2023五个表面辐照数据，均用原pv_generation.py中的面积、效率、温度修正和水面系数逐小时计算建筑PV；不改变建筑参数。
2. PVGIS中心时间标签减30分钟再加8小时，映射成本地小时起点，每小时保持4个15分钟值。跨年连续使用前一年的小时，不再把本年12月末环绕到1月初。2021年头8小时无前序源，保留占位并标为无效。
3. 合并原8个HKUST站点（W转kW），缺测仅允许最多4个15分钟步的向前填充作为历史。缺测行不作预测标签；不足数据仍保留时间轴并标无效。
4. 按同一日内时刻、此前31天（排除当天）、至少7个有效日的90%分位数估算参考出力。实测PV除参考出力形成扰动，限制在0—1.2；近零参考沿用原规则取1。峰值阈值只使用当前时刻之前的有效观测。
5. 建筑基线乘扰动得到pv_kW。历史有效性只取决于PV，不再因温度/湿度等非输入特征缺测而丢样本。全年来看不做能量归一化。
6. X[t-96:t]→PV[t]，要求96个历史点全部有效、单个目标原始观测有效。只在训练时段的有效PV上拟合归一化；验证、测试不参与，超范围也不截断。

## 时间划分及实际样本

| 集合 | 目标时间范围 | 有效样本 | 首个目标 |
|---|---|---:|---|
'''
    for name, stats in samples['summary'].items():
        description += f"| {name} | {stats['start_inclusive']} 至 {stats['end_exclusive']}（不含结束） | {stats['valid_windows']:,} | {stats['first_target']} |\n"
    description += f'''
共同站点数据从2021年7月开始；前7个有效日用于扰动预热，再加96步输入窗口，实际首个训练目标为{samples['summary']['train']['first_target']}。
完整时间轴含2021年上半年无效占位，不能直接删行拼接；使用forecast_origin_valid或window_index筛选样本。

## 2023对照与变化

原2023小时建筑PV重算最大误差：{comparison['old_hourly_reproduction_max_abs_error_kW']:.3g} kW。
按旧初始化方式复现原冻结PV标签最大误差：{comparison['old_frozen_labels_reproduction_max_abs_error_kW']:.3g} kW。
连续多年历史替代旧的2022年12月初始化后，2023有{comparison['new_multiyear_vs_old_2023']['changed_pv_rows_tolerance_1e_9']}行PV发生变化，最大差{comparison['new_multiyear_vs_old_2023']['max_abs_pv_difference_kW']:.8g} kW；逐行见changed_2023_rows.csv。
PV-only规则从旧五特征有效性限制中恢复了{comparison['new_multiyear_vs_old_2023']['history_rows_recovered_by_PV_only']}个2023历史输入点。
这是一版新的数据与单步任务，不能直接把旧672→96、五特征M0的R²作为同口径结果。

## 复现与完整性

生成命令：`external/ol-ems-hkust/.venv/bin/python prepare_pv_only_multiyear.py --output <新的目录>`。
原始输入、生成代码和所有输出的SHA-256记录在manifest.json。source_snapshot保留生成代码；既有冻结数据、2023原始文件和原M0源码均未覆盖。
模型默认从CSV现场造窗；NPZ为相同窗口的可直接读取副本。只读检查：`np.load('train.npz')['X'].shape`。
'''
    (staging / 'README.md').write_text(description, encoding='utf-8')
    source_files = [Path(__file__), ROOT / 'pv_generation.py', ROOT / 'align_hkust_pv_15min.py', ROOT / 'build_pv_disturbance_15min.py']
    (staging / 'source_snapshot').mkdir()
    for path in source_files:
        shutil.copy2(path, staging / 'source_snapshot' / path.name)
    for path in inputs:
        if sha256(path) != before[str(path.relative_to(ROOT))]:
            raise ValueError(f'Source changed during preparation: {path}')
    outputs = [{'path': str(path.relative_to(staging)), 'size_bytes': path.stat().st_size, 'sha256': sha256(path)}
               for path in sorted(staging.rglob('*')) if path.is_file()]
    manifest = {'schema_version': 2, 'artifact_type': 'pv_only_forecast_96to1',
                'dataset_version': output.name, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'frequency': '15min', 'timezone': 'Asia/Shanghai', 'rows': len(frame),
                'parameters': {'lookback_steps': 96, 'prediction_horizon': 1, 'history_features': ['pv'], 'target_feature': 'pv',
                               'building_parameters_unchanged': True, 'pv_max_history_ffill_steps': 4,
                               'disturbance_prior_days': 31, 'disturbance_min_valid_days': 7, 'disturbance_quantile': .9,
                               'max_disturbance_factor': 1.2, 'prior_peak_fraction': .02,
                               'annual_energy_normalization': False, 'source_column': 'pv_kW'},
                'splits': samples['summary'], 'scaler': scaler, 'quality': quality, 'comparison_2023': comparison,
                'inputs': [{'path': name, 'sha256': digest, 'size_bytes': (ROOT / name).stat().st_size} for name, digest in before.items()],
                'code': [{'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)} for path in source_files],
                'outputs': outputs,
                'provenance': 'Constructed floating-building PV = year-specific PVGIS/building physical baseline multiplied by causal HKUST rooftop measured-PV disturbance; not measured floating-building telemetry.',
                'time_policy': 'Local naive 15-minute labels; UTC hourly centers minus 30min plus 8h, continuous across years, step-held within each hour. Preserve existing HKUST labels; no additional timestamp offset is inferred.',
                'validation': 'Complete time axis, finite PV, causal flags, train-only scaler, 96-to-1 windows, exact legacy reconstruction within 1e-9 kW; model smoke validation recorded separately.'}
    write_json(staging / 'manifest.json', manifest)
    if output.exists():
        raise FileExistsError(output)
    staging.rename(output)
    print(json.dumps({'output': str(output), 'splits': samples['summary'], 'quality': quality, 'comparison': comparison}, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    freeze(parser.parse_args().output)
