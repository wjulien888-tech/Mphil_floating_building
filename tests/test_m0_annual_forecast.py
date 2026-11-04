"""Chronological leakage, quality and checkpoint-chain tests; no formal run."""
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'model_variants'), str(ROOT / 'external/ol-ems-hkust')]
import run_m0_annual_forecast as annual


def frame(start='2023-03-01', end='2023-05-02'):
    stamps = pd.date_range(start, end, freq='15min', inclusive='left')
    return pd.DataFrame(dict(datetime=stamps, pv=np.arange(len(stamps), dtype=float),
                             history_valid=1, pv_target_valid=1))


def test_monthly_update_uses_only_arrived_labels_and_last_14_days_validation():
    folds = annual.month_folds()
    assert len(folds) == 12
    assert folds[0]['training_end_exclusive'] == '2023-01-01 00:00:00'
    assert folds[0]['validation_start'] == '2023-01-01 00:00:00'
    for fold in folds[1:]:
        cutoff = pd.Timestamp(fold['labels_available_before'])
        assert cutoff - pd.Timestamp(fold['validation_start']) == pd.Timedelta(days=14)
        assert fold['training_end_exclusive'] == fold['validation_start']
        assert fold['test_origin_start'] == str(cutoff)
    f = frame()
    april = folds[1]
    for start, end in ((april['training_start'], april['training_end_exclusive']),
                       (april['validation_start'], april['labels_available_before'])):
        anchors, _, _ = annual.window_anchors(f, start, end)
        assert (f.datetime.iloc[anchors + 95] < pd.Timestamp(end)).all()


def test_month_end_forecasts_keep_issuing_month_model_and_no_annual_tail_score():
    f = frame()
    march, _, _ = annual.window_anchors(f, '2023-03-01', '2023-05-01', origin_end='2023-04-01')
    april, _, _ = annual.window_anchors(f, '2023-04-01', '2023-05-01', origin_end='2023-05-01')
    assert f.datetime.iloc[march[-1]] == pd.Timestamp('2023-03-31 23:45')
    assert f.datetime.iloc[march[-1]+95] == pd.Timestamp('2023-04-01 23:30')
    assert f.datetime.iloc[april[-1]+95] == pd.Timestamp('2023-04-30 23:45')
    assert not set(march).intersection(april)
    yearly, _, _ = annual.window_anchors(f, '2023-03-01', '2023-05-01')
    np.testing.assert_array_equal(np.r_[march, april], yearly)


def test_missing_point_rejects_target_and_later_history_without_compressing_axis():
    f = frame()
    bad = 200
    f.loc[bad, ['history_valid', 'pv_target_valid']] = 0
    anchors, possible, rejected = annual.window_anchors(f, '2023-03-01', '2023-04-01')
    assert rejected == 192
    assert not any(bad - 95 <= a <= bad + 96 for a in anchors)
    assert bad - 96 in anchors and bad + 97 in anchors
    assert possible == len(anchors) + rejected
    assert f.datetime.iloc[bad+1] - f.datetime.iloc[bad-1] == pd.Timedelta(minutes=30)


def test_future_values_and_future_quality_cannot_change_train_or_validation_samples():
    f = frame()
    altered = f.copy()
    future = altered.datetime.ge('2023-04-01')
    altered.loc[future, 'pv'] = 1e9
    altered.loc[future, ['history_valid', 'pv_target_valid']] = 0
    for start, end in [('2023-03-01','2023-03-18'), ('2023-03-18','2023-04-01')]:
        a, _, _ = annual.window_anchors(f, start, end)
        b, _, _ = annual.window_anchors(altered, start, end)
        np.testing.assert_array_equal(a, b)
        for anchor in a:
            np.testing.assert_array_equal(f.pv.iloc[anchor-96:anchor+96], altered.pv.iloc[anchor-96:anchor+96])


def test_warm_start_rejects_wrong_month_source_and_changed_scaler():
    fold = annual.month_folds()[0]
    shared = dict(dataset_identity={'hash':'x'}, model_architecture={'type':'GRU'},
                  dataset_scaler=dict(scaling_factors_min=torch.tensor([0.]), scaling_factors_max=torch.tensor([80.])))
    parent = {**copy.deepcopy(shared), 'annual_fold': fold}
    annual.verify_parent(parent, shared, fold)
    with pytest.raises(ValueError, match='preceding month'):
        annual.verify_parent(parent, shared, annual.month_folds()[1])
    parent['dataset_scaler']['scaling_factors_max'] = torch.tensor([90.])
    with pytest.raises(ValueError, match='scaler changed'):
        annual.verify_parent(parent, shared, fold)


def test_preparation_never_trains_and_monthly_scaler_stays_initial(tmp_path, monkeypatch):
    # Use the real source for boundary checks; no model is instantiated here.
    args, identity, config, base = annual.inputs(annual.DEFAULT_DATA, tmp_path)
    monkeypatch.setattr(annual, 'train_m0', lambda *a, **k: pytest.fail('Preparation trained a model'))
    monkeypatch.setattr(annual, 'inputs', lambda *a, **k: (args, identity, config, base))
    protocol = annual.prepare_protocol(tmp_path, tmp_path / 'runs')
    assert not (tmp_path / 'runs').exists()
    assert protocol['status'] == 'prepared_not_run'
    for fold in annual.month_folds():
        views, counts = annual.fold_views(base, fold)
        assert views['train'].pv_max == base.pv_max
        assert views['train'].state_dict()['scaler_fit_interval'][1] == '2023-01-01T00:00:00'
        assert pd.Timestamp(counts['val']['last_target']) < pd.Timestamp(fold['labels_available_before'])


def test_pooled_rmse_is_weighted_by_pairs_not_months(tmp_path):
    for month, errors in [('2023-03', [1.]), ('2023-04', [3.,3.,3.])]:
        for arm in ('updated','fixed_initial'):
            p = tmp_path / month / arm
            p.mkdir(parents=True)
            pd.DataFrame(dict(pv_actual_kw=np.zeros(len(errors)), pv_forecast_kw=errors,
                              persistence_24h_kw=np.zeros(len(errors)), persistence_last_kw=np.zeros(len(errors)))).to_csv(p/'predictions.csv.gz', index=False)
    annual.pool_results(tmp_path, ['2023-03','2023-04'])
    d = pd.read_csv(tmp_path/'pooled_metrics.csv')
    row = d.loc[d.source_group.eq('annual') & d.arm.eq('updated') & d.method.eq('M0')].iloc[0]
    assert row['count'] == 4
    assert row.rmse_kw == pytest.approx(7**.5)


def test_combined_training_testing_entry_is_disabled(tmp_path):
    with pytest.raises(RuntimeError, match='retired'):
        annual.run(tmp_path, tmp_path / 'experiment', {})
    assert not (tmp_path / 'experiment').exists()


def test_train_stage_cannot_invoke_test_or_mpc(tmp_path, monkeypatch):
    args, identity, config, base = annual.inputs(annual.DEFAULT_DATA, tmp_path)
    monkeypatch.setattr(annual, 'inputs', lambda *a, **k: (args, identity, config, base))
    monkeypatch.setattr(annual, '_configure_logging', lambda *a, **k: None)
    monkeypatch.setattr(annual, 'evaluate_m0', lambda *a, **k: pytest.fail('Training entered test evaluation'))
    monkeypatch.setattr(annual, 'test_month', lambda *a, **k: pytest.fail('Training entered MPC testing'))
    calls = []
    def stub_training(model, train, val, directory, settings, metadata, runtime):
        assert train.split == 'train' and val.split == 'val'
        calls.append(metadata['annual_fold']['month'])
    monkeypatch.setattr(annual, 'train_m0', stub_training)
    monkeypatch.setattr(annual, 'review_month', lambda *a: {'status':'awaiting_user_review_before_test'})
    protocol = annual.prepare_protocol(tmp_path, tmp_path/'experiment')
    result = annual.train_month(tmp_path,tmp_path/'experiment',protocol,'2023-03')
    assert calls == ['2023-03']
    assert result['status'] == 'awaiting_user_review_before_test'
    assert not (tmp_path/'experiment/2023-03/coupled_test').exists()
    assert not (tmp_path/'experiment/2023-04').exists()
