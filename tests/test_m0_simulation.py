"""Contracts for continuous simulation: time, information, units and bounded UI."""
import importlib.util
from pathlib import Path
from io import StringIO
import shutil
import sys
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('m0_simulation', ROOT/'tools/run_m0_simulation.py')
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

@pytest.fixture(scope='module')
def bundle():
    return sim.load_model(sim.DEFAULT_CHECKPOINT)


def inputs(bundle, start='2023-11-01', steps=2880, policy='hold-last'):
    return sim.prepare_inputs(bundle[1], start, steps, policy,
        ROOT/'output/frozen/hkust_m0_v1_20260909/dispatch.csv',
        ROOT/'output/frozen/hkust_m0_v1_20260909/load_tiers.csv')


def test_month_never_skips_time_and_fill_is_causal(bundle):
    frame, history, repairs = inputs(bundle)
    assert len(frame) == 2880+95
    assert (frame.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert len(repairs) == 3
    for timestamp in repairs.index:
        assert frame.loc[timestamp,'pv'] == frame.loc[timestamp-pd.Timedelta(minutes=15),'pv']
    assert len(history) == len(frame)+bundle[1].lookback_window


def test_invalid_data_rejected_unless_explicit(bundle):
    with pytest.raises(ValueError, match='invalid PV'):
        inputs(bundle, policy='error')


def test_no_test_boundary_leak_or_off_grid_start(bundle):
    with pytest.raises(ValueError, match='test split'):
        inputs(bundle, start='2023-12-03')
    with pytest.raises(ValueError, match='boundary'):
        inputs(bundle, start='2023-11-01 00:01')


def test_new_input_format_matches_original_forecast_loader(bundle):
    model, dataset, _ = bundle
    frame, histories, _ = inputs(bundle, steps=2, policy='error')
    test = dataset.for_split('test')
    i = test.timestamps.get_loc(frame.index[0])
    original = test[i]['history']
    torch.testing.assert_close(histories[:dataset.lookback_window], original)
    with torch.no_grad():
        a = model.forecast_pv(original.unsqueeze(0))
        b = model.forecast_pv(histories[:dataset.lookback_window].unsqueeze(0))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert a.shape == (1,96,1)
    assert torch.isfinite(a).all()


def test_future_actual_pv_cannot_change_current_model_input(bundle):
    _, histories, _ = inputs(bundle, steps=2)
    lookback = bundle[1].lookback_window
    before = histories[:lookback].clone()
    histories[lookback:] = 999
    assert torch.equal(before, histories[:lookback])


def test_physics_validator_rejects_broken_energy_balance():
    row = dict(pv_used_kwh=1.,battery_discharge_kwh=0.,grid_import_kwh=0.,
               load_served_kwh=1.,battery_charge_kwh=0.,grid_export_kwh=0.,overgeneration_kwh=0.,
               soc_before_kwh=100.,soc_after_kwh=100.)
    assert sim.verify_execution(row) == (0.,0.)
    row['grid_import_kwh'] = 1.
    with pytest.raises(RuntimeError, match='energy/SOC'):
        sim.verify_execution(row)


def test_terminal_reuses_screen_and_latest_snapshot(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(sys, 'stdout', stream)
    monkeypatch.setattr(shutil,'get_terminal_size',lambda fallback: shutil.os.terminal_size((80,24)))
    view = sim.TerminalView(False)
    snapshot = dict(method='model',step=1,total=2880,timestamp='2023-11-01',elapsed=1.,eta=3.,
                    seconds=1.,soc=50.,purchase=1.,bill=2.,plan_purchase=4.,history=[50.,51.],
                    charts=[('SOC',[50.,60.],'%')])
    view.publish(snapshot)
    second = dict(snapshot, step=2)
    view.publish(second)
    assert view.snapshot is second
    view.render()
    output = stream.getvalue()
    assert output.startswith('\033[H') and output.endswith('\033[J')
    assert '2/2880' in output
    assert len(output.splitlines()) <= 23


def test_mpc_night_known_solution_and_constraints():
    mpc = sim.make_mpc()
    future = dict(load=torch.ones(96)*2, import_price=torch.ones(96)*.3,
                  export_price=torch.ones(96)*.2, grid_available=torch.ones(96),
                  critical_load=torch.ones(96))
    arrays, violation = sim.solve_mpc(mpc, torch.zeros(1,96,1), future, sim.HARDWARE.min_soc_kWh)
    assert violation < 1e-4
    np.testing.assert_allclose(arrays['grid_import'], 2., atol=1e-4)
    np.testing.assert_allclose(arrays['charge'], 0., atol=1e-4)
    np.testing.assert_allclose(arrays['discharge'], 0., atol=1e-4)
    np.testing.assert_allclose(arrays['soc'], sim.HARDWARE.min_soc_kWh, atol=1e-4)


def test_s0_night_grid_charge_is_economically_chosen_and_executed():
    mpc,adapter=sim.make_mpc(),sim._build_adapter(sim.HARDWARE)
    adapter.reset(soc_kwh=sim.HARDWARE.min_soc_kWh)
    future=dict(load=torch.ones(96)*2,import_price=torch.ones(96),
                export_price=torch.zeros(96),grid_available=torch.ones(96),critical_load=torch.ones(96))
    future['import_price'][0]=.1
    arrays,violation=sim.solve_mpc(mpc,torch.zeros(1,96,1),future,adapter.get_soc_kwh())
    assert violation<1e-4 and arrays['charge'][0]>1
    row=adapter.step(dict(battery_charge=arrays['charge'][0],battery_discharge=arrays['discharge'][0],
                          grid_import=arrays['grid_import'][0],grid_export=arrays['grid_export'][0]),
                     dict(pv=0.,**{k:float(v[0]) for k,v in future.items()}))
    sim.verify_execution(row)
    assert row['soc_after_kwh']==pytest.approx(arrays['soc'][1],abs=1e-4)
    assert row['grid_import_kw']==pytest.approx(arrays['grid_import'][0],abs=1e-4)


@pytest.mark.parametrize('grid_available,load,expected_charge',[(1.,2.,3.),(.5,2.,.5),(1.,6.,0.),(0.,2.,0.)])
def test_adapter_limits_grid_charge_to_available_import_capacity(grid_available,load,expected_charge):
    config=sim.replace(sim.HARDWARE,grid_import_limit_kW=5.)
    adapter=sim._build_adapter(config)
    row=adapter.step(dict(battery_charge=10.,battery_discharge=0.,grid_import=12.,grid_export=0.),
                     dict(pv=0.,load=load,critical_load=load,import_price=.3,export_price=.2,grid_available=grid_available))
    sim.verify_execution(row)
    assert row['battery_charge_kw']==pytest.approx(expected_charge)
    assert row['grid_import_kw']<=5.*grid_available+1e-6


def test_legacy_pv_only_charging_policy_remains_explicitly_available():
    adapter=sim._build_adapter(sim.replace(sim.HARDWARE,allow_grid_charging=False))
    row=adapter.step(dict(battery_charge=10.,battery_discharge=0.,grid_import=12.,grid_export=0.),
                     dict(pv=0.,load=2.,critical_load=1.,import_price=.3,export_price=.2,grid_available=1.))
    sim.verify_execution(row)
    assert row['battery_charge_kw']==0
    assert row['grid_import_kw']==pytest.approx(2.)


def test_mpc_first_action_matches_physics_when_pv_prediction_is_exact():
    mpc, adapter = sim.make_mpc(), sim._build_adapter(sim.HARDWARE)
    future = dict(load=torch.ones(96)*2, import_price=torch.ones(96)*.3,
                  export_price=torch.ones(96)*.2, grid_available=torch.ones(96),
                  critical_load=torch.ones(96))
    arrays, _ = sim.solve_mpc(mpc, torch.ones(1,96,1)*10, future, adapter.get_soc_kwh())
    plan = dict(battery_charge=arrays['charge'][0], battery_discharge=arrays['discharge'][0],
                grid_import=arrays['grid_import'][0], grid_export=arrays['grid_export'][0])
    row = adapter.step(plan,dict(pv=10.,**{k:float(v[0]) for k,v in future.items()}))
    sim.verify_execution(row)
    assert abs(row['soc_after_kwh'] - arrays['soc'][1]) < 1e-4
    assert abs(row['grid_import_kw'] - arrays['grid_import'][0]) < 1e-4
    assert abs(row['grid_export_kw'] - arrays['grid_export'][0]) < 1e-4


def test_inaccurate_solver_retries_same_problem(monkeypatch):
    mpc = sim.make_mpc()
    solve = mpc._problem.solve
    calls = []
    def inaccurate_first(**kwargs):
        calls.append(kwargs['solver'])
        if len(calls) == 1:
            mpc._problem._status = sim.cp.OPTIMAL_INACCURATE
            return
        return solve(**kwargs)
    monkeypatch.setattr(mpc._problem, 'solve', inaccurate_first)
    future = dict(load=torch.ones(96)*2, import_price=torch.ones(96)*.3,
                  export_price=torch.ones(96)*.2, grid_available=torch.ones(96),
                  critical_load=torch.ones(96))
    arrays, violation = sim.solve_mpc(mpc, torch.zeros(1,96,1), future, sim.HARDWARE.min_soc_kWh)
    assert calls == [sim.cp.ECOS, sim.cp.SCIPY]
    assert mpc.last_solver == 'SCIPY/HiGHS'
    assert mpc.last_solver_attempts == 2
    assert violation < 1e-4
    np.testing.assert_allclose(arrays['grid_import'], 2., atol=1e-4)


def test_all_solver_failures_remain_fatal(monkeypatch):
    mpc = sim.make_mpc()
    def fail(**kwargs):
        raise sim.cp.error.SolverError('test failure')
    monkeypatch.setattr(mpc._problem, 'solve', fail)
    future = dict(load=torch.ones(96)*2, import_price=torch.ones(96)*.3,
                  export_price=torch.ones(96)*.2, grid_available=torch.ones(96),
                  critical_load=torch.ones(96))
    with pytest.raises(RuntimeError, match='failed all solvers'):
        sim.solve_mpc(mpc, torch.zeros(1,96,1), future, sim.HARDWARE.min_soc_kWh)
