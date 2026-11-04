"""M1 uses solver defaults while the frozen M0 evaluation policy is preserved.

These contracts substitute solver calls; they do not run optimisation or train
a model. Numerical/gradient comparisons remain in the differentiable MPC tests.
"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "model_variants")]
import m0_scenarios as scenarios
import train_m1 as training


class FakeProblem:
    def __init__(self, *, status=cp.OPTIMAL, residual=0., value=1., fail_ecos=False):
        self.status = status
        self.calls = []
        self.constraints = [SimpleNamespace(violation=lambda: np.array([residual]))]
        self.variable = SimpleNamespace(value=np.array([value]))
        self.fail_ecos = fail_ecos

    def solve(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_ecos and kwargs["solver"] == cp.ECOS:
            raise cp.error.SolverError("ECOS failed")

    def is_mixed_integer(self):
        return False

    def variables(self):
        return [self.variable]


@pytest.mark.parametrize("priority,solver,expected", [
    (False, "ECOS", {"solver": cp.ECOS}),
    (True, "HiGHS", {"solver": cp.SCIPY, "scipy_options": {"method": "highs"}}),
])
def test_m1_validation_solver_receives_only_default_options(priority, solver, expected):
    problem = FakeProblem()
    name, violation = scenarios._solve(problem, priority=priority, solver_defaults=True)
    assert name == solver
    assert violation == 0.
    assert problem.calls == [expected]


def test_m1_validation_highs_fallback_also_uses_defaults():
    problem = FakeProblem(fail_ecos=True)
    name, _ = scenarios._solve(problem, solver_defaults=True)
    assert name == "HiGHS"
    assert problem.calls == [
        {"solver": cp.ECOS},
        {"solver": cp.SCIPY, "scipy_options": {"method": "highs"}},
    ]


def test_m1_inaccurate_status_and_residual_are_kept_in_diagnostics():
    problem = FakeProblem(status=cp.OPTIMAL_INACCURATE, residual=.02)
    diagnostics = {}
    name, residual = scenarios._solve(problem, solver_defaults=True, diagnostics=diagnostics)
    assert name == "ECOS"
    assert residual == pytest.approx(.02)
    assert diagnostics["status"] == cp.OPTIMAL_INACCURATE
    assert diagnostics["solver"] == "ECOS"
    assert diagnostics["constraint_violation"] == pytest.approx(.02)
    assert len(problem.calls) == 1


@pytest.mark.parametrize("status", [cp.INFEASIBLE, cp.UNBOUNDED])
def test_m1_validation_does_not_accept_missing_optimum(status):
    with pytest.raises(RuntimeError, match="Solver attempts failed"):
        scenarios._solve(FakeProblem(status=status), solver_defaults=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_m1_validation_still_rejects_nonfinite_values(value):
    with pytest.raises(RuntimeError, match="nonfinite solution"):
        scenarios._solve(FakeProblem(value=value), solver_defaults=True)


def test_m0_solver_policy_and_precision_options_are_unchanged():
    problem = FakeProblem(fail_ecos=True)
    name, _ = scenarios._solve(problem)
    assert name == "HiGHS"
    assert problem.calls == [
        {"solver": cp.ECOS, "abstol": 1e-8, "reltol": 1e-8,
         "feastol": 1e-8, "max_iters": 1000},
        {"solver": cp.SCIPY, "scipy_options": {
            "method": "highs", "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9}},
    ]


@pytest.mark.parametrize("kwargs", [
    {"status": cp.OPTIMAL_INACCURATE},
    {"residual": .02},
])
def test_m0_retains_its_existing_status_and_feasibility_rejection(kwargs):
    with pytest.raises(RuntimeError, match="Solver attempts failed"):
        scenarios._solve(FakeProblem(**kwargs))


def test_m1_validation_constructs_controller_with_default_solver_policy(monkeypatch, tmp_path):
    constructor_calls = []
    constructor = scenarios.ScenarioMPC

    def capture_constructor(**kwargs):
        constructor_calls.append(kwargs)
        return constructor(**kwargs)

    monkeypatch.setattr(training, "ScenarioMPC", capture_constructor)
    monkeypatch.setattr(training, "evaluate_episode",
                        lambda *args, **kwargs: (np.zeros(4), 0., 1))
    monkeypatch.setattr(training, "append_csv", lambda *args: None)
    model = SimpleNamespace(eval=lambda: None)
    episode = SimpleNamespace(name="validation-only")
    scales = {"mse_kW2": 1., "economic_CNY_per_step": 1.}
    args = SimpleNamespace(prediction_weight=1., economic_weight=1.)
    training.validation(model, [episode], scales, args, tmp_path, epoch=0)
    assert constructor_calls == [{"solver_defaults": True, "battery_exclusive": False}]


def test_solver_diagnostics_preserve_warnings_and_separate_training_from_validation(tmp_path):
    controller = SimpleNamespace(
        version="fake-library-defaults", solver_statistics={"ECOS": 2},
        warning_statistics={"ECOS: Solved/Inaccurate.": 1},
        maximum_observed_constraint_violation=.02)
    training.write_solver_diagnostics(tmp_path, controller, "validation_epoch_0")
    controller.solver_statistics = {"ECOS": 6, "SCS": 1}
    controller.maximum_observed_constraint_violation = .03
    training.write_solver_diagnostics(tmp_path, controller, "training")
    diagnostics = json.loads((tmp_path / "solver_diagnostics.json").read_text())
    validation = diagnostics["validation_epoch_0"]
    assert validation["solver_calls"] == {"ECOS": 2}
    assert validation["warnings"] == {"ECOS: Solved/Inaccurate.": 1}
    assert validation["max_constraint_violation"] == .02
    assert validation["counts_include_backward_recomputation"] is False
    assert diagnostics["training"]["solver_calls"] == {"ECOS": 6, "SCS": 1}
    assert diagnostics["training"]["max_constraint_violation"] == .03
    assert diagnostics["training"]["counts_include_backward_recomputation"] is True
    assert all(row["accuracy_policy"] == "library_defaults" for row in diagnostics.values())
