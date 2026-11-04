"""M2 loss, real ENS gradient, matched execution and resume contracts."""

import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "model_variants")]
import train_m2 as training
from m0_scenarios import C, DT, H, Scenario


class TinyForecaster(torch.nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
        self.dropout = torch.nn.Dropout(dropout)

    def forecast_pv(self, history):
        return self.dropout(torch.ones((1, H, 1), dtype=torch.float64)) * self.bias + history.mean() * .1


class TinyMPC:
    version = 'tiny-test-controller'

    def __init__(self):
        self.solver_statistics = {}
        self.warning_statistics = {}
        self.maximum_observed_constraint_violation = 0.
        self.last_failure = None

    def __call__(self, forecast, future, soc, reserve):
        # The SOC term makes incorrect detachment between chunks detectable.
        charge = .25 * forecast + .002 * (soc - C.min_soc_kWh)
        zero = forecast * 0.
        return dict(charge=charge, discharge=zero, grid_import=zero, grid_export=zero,
                    unmet_cl1=zero, unmet_cl2=zero, unmet_cl3=zero)


class ForbiddenTestEpisode:
    """Any attempt to use held-out observations in main must fail this test."""
    name = "held-out-test-must-not-be-used"
    split = "test"

    def __getattr__(self, name):
        raise AssertionError(f"Training accessed held-out test field: {name}")


def episode(name="episode", steps=5, split="train", lookback=4, case=None):
    start = "2023-01-10" if split == "train" else "2023-09-10"
    frame = pd.DataFrame(dict(pv=np.linspace(0., 3., steps + H - 1), load=3.,
                              cl1=1., cl2=1., cl3=1., import_price=.3, export_price=.2),
                         index=pd.date_range(start, periods=steps + H - 1, freq="15min"))
    history = torch.zeros((len(frame) + lookback, 1), dtype=torch.float32)
    return SimpleNamespace(name=name, steps=steps, frame=frame, histories=history,
                           split=split, case=case or Scenario(name, "S0", initial_soc=.5))


@pytest.fixture(autouse=True)
def clear_stop(monkeypatch):
    monkeypatch.setattr(training, "STOP", False)


def test_checkpoint_recomputation_matches_values_gradients_and_rng():
    model = TinyForecaster(dropout=.25).train()
    e = episode(steps=7)
    results = []
    for chunk in (0, 2):
        torch.manual_seed(817)
        model.zero_grad(set_to_none=True)
        sums, soc, n = training.differentiable_rollout(model, TinyMPC(), e, chunk_steps=chunk)
        (sums[1] + soc * .2).backward()
        results.append((sums.detach().clone(), soc.detach().clone(), model.bias.grad.clone(), torch.get_rng_state()))
        assert n == 7
    for direct, recomputed in zip(results[0], results[1]):
        torch.testing.assert_close(direct, recomputed, rtol=0., atol=1e-12)
    assert results[1][2].abs() > 0


def test_soc_gradient_crosses_every_checkpoint_chunk():
    model = TinyForecaster()
    n = 6
    _, soc, _ = training.differentiable_rollout(model, TinyMPC(), episode(steps=n), chunk_steps=1)
    soc.backward()
    multiplier = 1 + C.charge_efficiency * DT * .002
    expected = C.charge_efficiency * DT * .25 * sum(multiplier ** k for k in range(n))
    assert model.bias.grad.item() == pytest.approx(expected, rel=1e-10)
    assert model.bias.grad.item() > C.charge_efficiency * DT * .25 * 5


@pytest.fixture
def small_training_environment(tmp_path, monkeypatch):
    """Exercise real main/checkpoint/RNG code while replacing expensive work."""
    monkeypatch.setattr(training, "ROOT", tmp_path)
    source = tmp_path / "test_source.py"
    source.write_text("# isolated test source\n")
    base_path = tmp_path / "m0.pt"
    base_path.write_bytes(b"isolated M0 initial identity")
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text("{}")
    base = dict(format_version=1, config={"lookback_steps": 4, "split_windows": {
                    "train": {"start": "2021-07-01", "end": "2023-09-01"},
                    "val": {"start": "2023-09-01", "end": "2023-11-01"}}}, dataset_scaler={"pv_min": 0., "pv_max": 3.},
                dataset_identity={"source": "train-only-test-fixture"}, model_architecture={"kind": "tiny"})
    bundle = SimpleNamespace(episodes={"train": [episode(f"train-{j}", steps=3) for j in range(3)],
                                       "val": [episode("validation", steps=3, split="val")],
                                       "test": [ForbiddenTestEpisode()]},
                             metadata={"test_used": False, "train_end": "2023-09-01", "val_end": "2023-11-01",
                                       "split_intervals": {"train": ["2021-07-01", "2023-09-01"],
                                                           "val": ["2023-09-01", "2023-11-01"]},
                                       "requested_episode_counts": {"train": 3, "val": 1},
                                       "excluded_counts": {"train": 0, "val": 0},
                                       "monthly_coverage": [], "selections": []})
    monkeypatch.setattr(training.common, "load_model", lambda _: (TinyForecaster(), SimpleNamespace(lookback_window=4), copy.deepcopy(base)))
    monkeypatch.setattr(training, "prepare_training_data", lambda *args: bundle)
    monkeypatch.setattr(training, "code_identity", lambda: {"test_source.py": training.sha256_file(source)})
    monkeypatch.setattr(training, "DifferentiableScenarioMPC", TinyMPC)
    calls, snapshots = [], []
    controls = {"interrupt_at": None, "fail_baseline": False}

    def fake_validation(model, episodes, scales, args, folder, epoch, limit_steps=None):
        assert all(e.split == "val" and e.frame.index.max() < pd.Timestamp("2023-11-01") for e in episodes)
        if controls["fail_baseline"] and epoch == 0:
            raise KeyboardInterrupt("interrupt baseline validation")
        model.eval()
        return dict(steps=3., mse_kW2=float(model.bias.detach() ** 2), economic_CNY=1.,
                    economic_CNY_per_step=1 / 3, critical_ENS_kWh=0., grid_bill_CNY=1.,
                    loss=float(model.bias.detach() ** 2), operational_CNY_per_step=float(model.bias.detach() ** 2), elapsed_seconds=0.)

    def fake_rollout(model, mpc, e, **kwargs):
        assert e.split == "train" and e.frame.index.max() < pd.Timestamp("2023-09-01")
        sample = random.random() + float(np.random.random()) + float(torch.rand(()))
        record = dict(name=e.name, sample=sample, completed=False)
        calls.append(record)
        if len(calls) == controls["interrupt_at"]:
            raise KeyboardInterrupt("interrupt inside episode after consuming RNG")
        record["completed"] = True
        mse = model.bias.square() * (1 + sample)
        cost = model.bias * (.5 + sample)
        return torch.stack((mse, cost, cost * 0., cost, cost * 0., cost * 0., cost * 0., cost * 0.+1., cost * 0.+1., cost * 0.+1.)), model.bias + 100., 3

    original_save = training.atomic_save

    def recording_save(path, payload):
        original_save(path, payload)
        if path.name == "last.pt":
            snapshots.append(copy.deepcopy(payload["state"]))

    monkeypatch.setattr(training, "validation", fake_validation)
    monkeypatch.setattr(training, "differentiable_rollout", fake_rollout)
    monkeypatch.setattr(training, "atomic_save", recording_save)

    def args(name, resume=None):
        values = ["--checkpoint", str(base_path), "--scenario-config", str(scenario_path),
                  "--run-name", name, "--max-epochs", "2", "--patience", "5"]
        if resume:
            values += ["--resume", str(resume)]
        return training.parser().parse_args(values)

    return SimpleNamespace(args=args, calls=calls, snapshots=snapshots, controls=controls, bundle=bundle)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0., atol=0.)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_interrupted_resume_matches_episode_order_rng_optimizer_and_weights(small_training_environment):
    env = small_training_environment
    uninterrupted = training.main(env.args("uninterrupted"))
    expected = training.load_checkpoint(uninterrupted / "last.pt")
    expected_calls = copy.deepcopy(env.calls)
    expected_snapshots = copy.deepcopy(env.snapshots)
    env.calls.clear()
    env.snapshots.clear()
    env.controls["interrupt_at"] = 2
    args = env.args("interrupted")
    with pytest.raises(KeyboardInterrupt):
        training.main(args)
    path = training.ROOT / "experiments/M2/interrupted/last.pt"
    partial = training.load_checkpoint(path)
    assert partial["state"]["epoch"] == 0
    assert partial["state"]["next_episode"] == 1
    assert len(partial["state"]["order"]) == 3
    env.controls["interrupt_at"] = None
    training.main(env.args("interrupted", resume=path))
    resumed = training.load_checkpoint(path)
    assert [x for x in env.calls if x["completed"]] == expected_calls
    assert env.calls[1]["sample"] == env.calls[2]["sample"]  # failed episode replayed
    for key in ("controller_state_dict", "optimizer_state_dict", "rng_state", "state"):
        assert_nested_equal(expected[key], resumed[key])
    # Both epochs checkpoint after every completed episode, including the last
    # episode before validation; no optimizer update is missing from last.pt.
    completed_positions = [(x["epoch"], x["next_episode"]) for x in expected_snapshots if x["order"] and x["next_episode"]]
    assert completed_positions == [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]


def test_resume_rejects_checkpoint_from_m1_even_in_m2_directory(small_training_environment):
    env = small_training_environment
    path = training.main(env.args("wrong_method")) / "last.pt"
    checkpoint = training.load_checkpoint(path)
    checkpoint["training_method"] = "M1"
    training.atomic_save(path, checkpoint)
    with pytest.raises(ValueError, match="Resume rejected"):
        training.main(env.args("wrong_method", resume=path))


def test_changed_training_arguments_reject_resume_without_overwriting_checkpoint(small_training_environment):
    env = small_training_environment
    path = training.main(env.args("changed_loss_weight")) / "last.pt"
    before = path.read_bytes()
    args = env.args("changed_loss_weight", resume=path)
    args.economic_weight = 2.
    with pytest.raises(ValueError, match="Resume rejected"):
        training.main(args)
    assert path.read_bytes() == before


def test_eight_month_checkpoint_cannot_resume_as_multiyear(small_training_environment):
    env = small_training_environment
    path = training.main(env.args("old_eight_month")) / "last.pt"
    saved = training.load_checkpoint(path)
    saved["training_identity"]["data"]["split_intervals"]["train"][0] = "2023-01-01"
    training.atomic_save(path, saved)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="old eight-month run cannot resume"):
        training.main(env.args("old_eight_month", resume=path))
    assert path.read_bytes() == before


def test_baseline_interruption_still_has_a_valid_resume_checkpoint(small_training_environment):
    env = small_training_environment
    env.controls["fail_baseline"] = True
    with pytest.raises(KeyboardInterrupt):
        training.main(env.args("baseline_interruption"))
    folder = training.ROOT / "experiments/M2/baseline_interruption"
    status = json.loads((folder / "status.json").read_text())
    assert Path(status["resume_from"]).is_file()
    saved = training.load_checkpoint(status["resume_from"])
    assert saved["state"]["best_loss"] is None
    assert saved["state"]["next_episode"] == 0
    env.controls["fail_baseline"] = False
    training.main(env.args("baseline_interruption", resume=Path(status["resume_from"])))




def test_penalty_units_and_mse_ablation():
    args = training.parser().parse_args([])
    scales = dict(mse_kW2=2., economic_CNY_per_step=4.)
    sums = [8., 12., 3., 10., 1., 2., 3., 10., 20., 30.]
    b = training.metric_row(sums, 2, scales, args)
    assert b['shedding_penalty_CNY'] == 76.
    assert b['operational_CNY_per_step'] == 44.
    assert b['critical_supply_rate'] == pytest.approx(.9)
    args.prediction_weight = 0.
    a = training.metric_row(sums, 2, scales, args)
    assert b['loss'] - a['loss'] == 2.
    assert b['operational_CNY_per_step'] == a['operational_CNY_per_step']


def test_tier_penalty_backpropagates_without_mse():
    args = training.parser().parse_args(['--prediction-weight', '0'])
    ens = torch.tensor([1., 2., 3.], dtype=torch.float64, requires_grad=True)
    mse = torch.tensor(100., requires_grad=True)
    loss = training.objective(mse, 0., dict(mse_kW2=2., economic_CNY_per_step=4.), args, ens)
    loss.backward()
    torch.testing.assert_close(ens.grad, torch.tensor([62.5, 12.5, 2.5], dtype=torch.float64))
    assert mse.grad == 0


def test_real_executor_ens_gradient_and_chunk_recomputation():
    class ShedMPC(TinyMPC):
        def __call__(self, forecast, future, soc, reserve):
            plan = super().__call__(forecast, future, soc, reserve)
            plan['unmet_cl1'] = .1 * forecast
            plan['unmet_cl2'] = .2 * forecast
            plan['unmet_cl3'] = .3 * forecast
            return plan
    model = TinyForecaster()
    e = episode(steps=3)
    args = training.parser().parse_args(['--prediction-weight','0'])
    results = []
    for chunk in (0, 1):
        model.zero_grad(set_to_none=True)
        sums, _, n = training.differentiable_rollout(model, ShedMPC(), e, chunk_steps=chunk)
        training.penalty(sums[4:7], args).backward()
        assert sums[2].item() == pytest.approx((sums[4]+sums[5]).item())
        assert torch.all(sums[4:7] <= sums[7:10]+1e-9)
        assert model.bias.grad.abs() > 0
        results.append((sums.detach(),model.bias.grad.clone()))
    for a,b in zip(*results): torch.testing.assert_close(a,b)


def test_operational_selection_not_mixed_loss(small_training_environment, monkeypatch):
    env = small_training_environment
    def validate(model, episodes, scales, args, folder, epoch, limit_steps=None):
        return dict(loss=100.+epoch, operational_CNY_per_step=[10.,8.,9.,9.,9.][epoch])
    monkeypatch.setattr(training,'validation',validate)
    args=env.args('selection'); args.max_epochs=7; args.patience=3
    path=training.main(args)
    saved=training.load_checkpoint(path/'best.pt')
    status=json.loads((path/'status.json').read_text())
    assert saved['best_epoch']==1
    assert status['epoch']==4 and status['early_stopped']


def test_resume_rejects_changed_penalty(small_training_environment):
    env=small_training_environment
    path=training.main(env.args('penalty'))/'last.pt'
    before=path.read_bytes()
    args=env.args('penalty',resume=path); args.voll_cl1=51.
    with pytest.raises(ValueError,match='Resume rejected'): training.main(args)
    assert path.read_bytes()==before


def test_actual_mpc_penalty_gradient_and_independent_execution():
    torch.set_num_threads(1)
    case = Scenario('ens-gradient', 'S2', event_hour=0, outage_hours=24,
                    announced_hours=24, warning_hours=24, initial_soc=.1)
    e = episode('ens-gradient', steps=2, case=case)
    # Underforecast with ample actual PV: scheduled shedding cannot be restored.
    # Thus there is a physically meaningful gradient of actual ENS w.r.t. forecast.
    e.frame['pv'] = 5.
    model = TinyForecaster().eval()
    args = training.parser().parse_args(['--prediction-weight', '0'])
    sums, _, _ = training.differentiable_rollout(
        model, training.DifferentiableScenarioMPC(), e, chunk_steps=1)
    training.penalty(sums[4:7], args).backward()
    assert torch.isfinite(model.bias.grad) and model.bias.grad.abs() > 1e-5
    assert sums[6] > .1
    independent, _, _ = training.evaluate_episode(
        model, training.ScenarioMPC(solver_defaults=True, battery_exclusive=False), e)
    np.testing.assert_allclose(sums.detach().numpy()[4:], independent[4:], atol=1e-4, rtol=1e-4)
