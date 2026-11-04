"""Real MILP solves without importing the forecasting/plant runtime.

Only the common module's hardware export is injected; the actual ScenarioMPC,
CVXPY constraints and HiGHS solver are exercised at the full 96-step horizon.
"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import numpy as np
import cvxpy as cp
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'external/ol-ems-hkust'))
    from configurations.hkust_15min import DEFAULT_CONFIG
    common = ModuleType('run_m0_simulation')
    common.HARDWARE = replace(DEFAULT_CONFIG, battery_capacity_kWh=320., allow_grid_charging=True)
    spec = importlib.util.spec_from_file_location('isolated_milp_scenarios', ROOT / 'tools/m0_scenarios.py')
    module = importlib.util.module_from_spec(spec)
    # dataclasses needs its defining module registered during class creation.
    monkeypatch.setitem(sys.modules, spec.name, module)
    with monkeypatch.context() as local:
        local.setitem(sys.modules, 'run_m0_simulation', common)
        spec.loader.exec_module(module)
    monkeypatch.setenv('MPC_MILP_SOLVER', 'highs')
    return module


def inputs(m, grid=1, load=1):
    f = {k: np.full(m.H, float(load)) for k in m.TIERS}
    f.update(grid=np.full(m.H, float(grid)), import_price=np.full(m.H, .3),
             export_price=np.full(m.H, .2))
    return f


def check_mode(m, a, violation):
    assert violation < 1e-4
    assert set(a['battery_charge_mode']).issubset({0, 1})
    assert np.max(np.minimum(a['charge'], a['discharge'])) < 1e-5
    assert np.all(a['charge'] <= 50*a['battery_charge_mode'] + 1e-5)
    assert np.all(a['discharge'] <= 50*(1-a['battery_charge_mode']) + 1e-5)
    np.testing.assert_allclose(np.diff(a['soc']), m.DT*(m.C.charge_efficiency*a['charge']
        - a['discharge']/m.C.discharge_efficiency), atol=1e-5)


def test_binary_modes_support_charging_and_discharging(model_module):
    m = model_module
    f = inputs(m)
    f['import_price'][:48] = .1
    f['import_price'][48:] = 1.
    a, v, solver, stages = m.ScenarioMPC().solve(np.zeros(m.H), f, 160., np.full(m.H+1, 32.))
    check_mode(m, a, v)
    assert a['charge'].max() > 1
    assert a['discharge'].max() > 1
    assert solver == 'HiGHS-MILP' and stages == 1
    assert a['soc'].min() >= 32-1e-5 and a['soc'].max() <= 288+1e-5


def test_blackout_preserves_all_priority_stages(model_module):
    m = model_module
    a, v, solver, stages = m.ScenarioMPC().solve(np.full(m.H, 1.5), inputs(m, grid=0),
        32., np.full(m.H+1, 32.))
    check_mode(m, a, v)
    assert stages == 4
    assert a['grid_import'].max() < 1e-5
    assert m.DT*a['unmet_cl1'].sum() < 1e-4
    assert m.DT*a['unmet_cl2'].sum() == pytest.approx(12, abs=1e-4)
    assert m.DT*a['unmet_cl3'].sum() == pytest.approx(24, abs=1e-4)


def test_reserve_charging_is_exclusive(model_module):
    m = model_module
    a, v, solver, stages = m.ScenarioMPC().solve(np.zeros(m.H), inputs(m),
        32., np.full(m.H+1, 288.))
    check_mode(m, a, v)
    assert stages == 2 and a['charge'][0] > 1
    assert a['soc'][-1] == pytest.approx(288, abs=1e-4)


def test_forced_simultaneous_charge_discharge_is_infeasible(model_module):
    m = model_module
    model = m.ScenarioMPC()
    model.solve(np.zeros(m.H), inputs(m), 160., np.full(m.H+1, 32.))
    problem = cp.Problem(cp.Minimize(0), model.problems[-1].constraints +
        [model.v['charge'][0] == 1, model.v['discharge'][0] == 1])
    problem.solve(solver=cp.SCIPY, scipy_options={'method': 'highs'})
    assert problem.status == cp.INFEASIBLE


def test_differentiable_relaxation_remains_explicitly_continuous(model_module):
    model = model_module.ScenarioMPC(battery_exclusive=False, solver_defaults=True)
    assert 'battery_charge_mode' not in model.v
    assert all(not p.is_mixed_integer() and p.is_dpp() for p in model.problems)
    assert 'continuous' in model.version


def test_full_five_stage_outage_and_reserve(model_module):
    m = model_module
    a, v, solver, stages = m.ScenarioMPC().solve(np.full(m.H, 1.5), inputs(m, grid=0),
        32., np.full(m.H+1, 40.))
    check_mode(m, a, v)
    assert stages == 5
    assert m.DT*a['unmet_cl1'].sum() < 1e-4
    assert m.DT*a['unmet_cl2'].sum() == pytest.approx(12, abs=1e-4)


def test_gurobi_failure_falls_back_to_real_highs(model_module, monkeypatch):
    m = model_module
    monkeypatch.setenv('MPC_MILP_SOLVER', 'auto')
    monkeypatch.setattr(cp, 'installed_solvers', lambda: ['GUROBI', 'SCIPY'])
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    import _gurobi_bounded_lp
    def native_failure(*args, **kwargs):
        calls.append(cp.GUROBI)
        raise Exception('simulated native Gurobi license error')
    monkeypatch.setattr(_gurobi_bounded_lp, 'solve_with_native_bounds', native_failure)
    original = cp.Problem.solve
    calls = []
    def solve(problem, **kwargs):
        calls.append(kwargs['solver'])
        if kwargs['solver'] == cp.GUROBI:
            raise Exception('simulated native Gurobi license error')
        return original(problem, **kwargs)
    monkeypatch.setattr(cp.Problem, 'solve', solve)
    a, v, solver, _ = m.ScenarioMPC().solve(np.zeros(m.H), inputs(m), 160., np.full(m.H+1, 32.))
    check_mode(m, a, v)
    assert calls == [cp.GUROBI, cp.SCIPY]
    assert solver == 'HiGHS-MILP'


def test_default_requires_gurobi_and_never_silently_falls_back(model_module, monkeypatch):
    m = model_module
    monkeypatch.delenv('MPC_MILP_SOLVER')
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    import _gurobi_bounded_lp
    calls = []
    def solve(problem, **kwargs):
        calls.append(kwargs['solver'])
        raise Exception('simulated native Gurobi license error')
    monkeypatch.setattr(cp.Problem, 'solve', solve)
    def native_failure(*args, **kwargs):
        calls.append(cp.GUROBI)
        raise Exception('simulated native Gurobi license error')
    monkeypatch.setattr(_gurobi_bounded_lp, 'solve_with_native_bounds', native_failure)
    with pytest.raises(RuntimeError, match='Gurobi license error'):
        m.ScenarioMPC().solve(np.zeros(m.H), inputs(m), 160., np.full(m.H+1, 32.))
    assert calls == [cp.GUROBI]


@pytest.mark.parametrize('grid,pv', [(1, 0.), (0, 1.5)])
def test_native_gurobi_bounds_match_highs_objectives(model_module, monkeypatch, grid, pv):
    pytest.importorskip('gurobipy')
    m = model_module
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    f = inputs(m, grid=grid)
    forecast = np.full(m.H, pv)
    reserve = np.full(m.H+1, 32.)
    highs = m.ScenarioMPC()
    reference, _, _, _ = highs.solve(forecast, f, 160., reserve)
    expected = [float(loss.value) for loss in highs.losses] + [highs.problems[-1].value]
    monkeypatch.setenv('MPC_MILP_SOLVER', 'gurobi')
    gurobi = m.ScenarioMPC()
    actual, violation, solver, _ = gurobi.solve(forecast, f, 160., reserve)
    check_mode(m, actual, violation)
    assert solver == 'GUROBI'
    observed = [float(loss.value) for loss in gurobi.losses] + [gurobi.problems[-1].value]
    np.testing.assert_allclose(observed, expected, atol=1e-4, rtol=1e-5)
