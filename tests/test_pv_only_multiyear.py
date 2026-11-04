"""Cross-year timing, quality and leakage contracts for the new PV dataset."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_pv_only_multiyear import align_hourly_baseline, make_samples


def test_cross_year_alignment_uses_preceding_year_and_never_wraps_future():
    source = pd.DataFrame({'datetime_utc_center': pd.to_datetime([
        '2021-12-31 16:30', '2022-01-01 00:30', '2022-12-31 16:30']),
        'pv_kW': [12., 20., 999.]})
    index = pd.date_range('2022-01-01', periods=40, freq='15min')
    result = align_hourly_baseline(source, index)
    assert result.iloc[:4].pvgis_floating_baseline_kW.tolist() == [12.] * 4
    assert result.iloc[:4].baseline_valid.eq(1).all()
    assert result.iloc[4:32].baseline_valid.eq(0).all()  # missing source hours
    assert result.iloc[32:36].pvgis_floating_baseline_kW.tolist() == [20.] * 4
    assert result.iloc[36:].baseline_valid.eq(0).all()
    changed = source.copy()
    changed.loc[2, 'pv_kW'] = 1e9
    pd.testing.assert_frame_equal(result, align_hourly_baseline(changed, index))
    first = align_hourly_baseline(source.iloc[1:], index)
    assert first.iloc[:32].baseline_valid.eq(0).all()


def fixture_frame():
    index = pd.date_range('2021-07-01', periods=6 * 96, freq='15min')
    frame = pd.DataFrame({'datetime': index, 'pv_kW': np.arange(len(index)) / 10,
                          'history_valid': 1, 'pv_target_valid': 1, 'split': 'train'})
    splits = {'train': ('2021-07-01', '2021-07-04'),
              'val': ('2021-07-04', '2021-07-05'),
              'test': ('2021-07-05', '2021-07-07')}
    for name, (start, end) in splits.items():
        frame.loc[frame.datetime.ge(start) & frame.datetime.lt(end), 'split'] = name
    return frame, splits


def test_exact_96_to_one_windows_and_quality_rejections():
    frame, splits = fixture_frame()
    frame.loc[150, 'history_valid'] = 0
    frame.loc[260, 'pv_target_valid'] = 0
    output = make_samples(frame, splits)
    anchors = output['arrays']['train']['anchor_indices']
    assert anchors[0] == 96
    assert 150 in anchors  # its historical inputs are valid; target flag is valid
    assert not set(range(151, 247)).intersection(anchors)
    assert 247 in anchors and 260 not in anchors
    scaler = output['scaler']
    a = output['arrays']['train']
    expected = frame.pv_kW.iloc[:96].to_numpy(dtype=np.float32) / np.float32(scaler['scale'])
    np.testing.assert_array_equal(a['X'][0, :, 0], expected)
    assert a['X'].shape[1:] == (96, 1)
    assert a['y_kw'].shape[1:] == (1, 1)
    assert a['y_kw'][0, 0, 0] == np.float32(frame.pv_kW.iloc[96])
    for name, (start, end) in splits.items():
        times = frame.datetime.iloc[output['arrays'][name]['anchor_indices']]
        assert times.ge(start).all() and times.lt(end).all()


def test_heldout_changes_cannot_affect_training_scaler_or_examples():
    frame, splits = fixture_frame()
    before = make_samples(frame, splits)
    altered = frame.copy()
    altered.loc[altered.datetime.ge('2021-07-04'), 'pv_kW'] += 10000
    after = make_samples(altered, splits)
    assert before['scaler'] == after['scaler']
    for key in before['arrays']['train']:
        np.testing.assert_array_equal(before['arrays']['train'][key], after['arrays']['train'][key])
    assert after['arrays']['val']['y_scaled'].min() > 1  # no held-out clipping


def test_model_loader_matches_frozen_arrays_and_persistence_baselines():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'external' / 'ol-ems-hkust'))
    sys.path.insert(0, str(root / 'model_variants'))
    from _pv_only_dataset import DEFAULT_CONFIG, PVOnlyDataset
    if not DEFAULT_CONFIG.data_path.is_file():
        pytest.skip('Local multiyear frozen snapshot is unavailable')
    dataset = PVOnlyDataset()
    raw = dataset.source_frame.pv.to_numpy(dtype=np.float32)
    assert dataset.input_dim == 1
    for name, view in dataset.split_views().items():
        with np.load(DEFAULT_CONFIG.data_path.parent / f'{name}.npz') as frozen:
            np.testing.assert_array_equal(view._anchors, frozen['anchor_indices'])
            assert len(view) == len(frozen['X'])
            for i in sorted({0, len(view) // 2, len(view) - 1}):
                sample = view[i]
                anchor = int(view._anchors[i])
                np.testing.assert_array_equal(sample['history'].numpy(), frozen['X'][i])
                np.testing.assert_array_equal(sample['pv_target'].numpy(), frozen['y_kw'][i])
                assert sample['persistence'].item() == raw[anchor - 96]
                assert sample['persistence_last'].item() == raw[anchor - 1]
                assert sample['timestamp'].item() == frozen['target_timestamps'][i]
