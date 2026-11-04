"""Continuous relaxation of the three-tier, lexicographic MPC for M1.

The CVXPY problems explicitly omit the binary battery mode: CvxpyLayer
cannot differentiate a MILP. Frozen-model evaluation defaults to exclusive
battery modes; use that same evaluator for both M0 and M1 final comparisons.
Only parameters/solutions used for numerical validation are detached. Reserve
targets and successive lexicographic caps remain part of the autograd graph.
"""
from __future__ import annotations

from pathlib import Path
import sys
import warnings
import logging
from typing import Mapping

import numpy as np
import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from m0_scenarios import C, DT, H, TIERS, Scenario, ScenarioMPC


def _vector(value, size: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.device.type != "cpu":
        raise ValueError("Differentiable ScenarioMPC currently requires CPU tensors")
    tensor = tensor.reshape(-1)
    if tensor.numel() != size or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return tensor


def reserve_profile_tensor(case: Scenario, hour: float, forecast,
                           load) -> torch.Tensor:
    """Exact torch equivalent of ``m0_scenarios.reserve_profile``.

    The known event schedule is discrete metadata. Forecast PV remains a
    differentiable input, including the repeated daily profile for long events.
    """
    pv = _vector(forecast, H, "forecast")
    demand = _vector(load, H, "load")
    _, _, active = case.information(hour)
    minimum = pv.new_tensor(C.min_soc_kWh)
    if not active:
        return minimum.expand(H + 1) + pv.sum() * 0.0
    end = case.event_hour + case.announced_hours
    targets = []
    for j in range(H + 1):
        t = hour + j * DT
        if t >= end:
            targets.append(minimum + pv.sum() * 0.0)
            continue
        first = max(t, case.event_hour)
        offsets = np.rint((np.arange(first, end, DT) - hour) / DT).astype(int) % H
        index = torch.as_tensor(offsets, dtype=torch.long)
        shortage = (case.rho * demand[index] - pv[index]).clamp_min(0.0)
        targets.append(minimum + shortage.sum() * DT / C.discharge_efficiency)
    return torch.stack(targets)


class DifferentiableScenarioMPC(nn.Module):
    """Same continuous physical constraints and priorities, without binary modes.

    One call solves one 96-step horizon. Every solve, including numerical
    retries, goes through diffcp and retains a backward operation. ECOS and SCS
    use their library-default accuracy and iteration settings. Library warnings
    are recorded rather than promoted to errors. Physical residuals remain
    diagnostics; non-finite results and solver exceptions still fail/retry.
    The 1e-6 allowance between priority objectives is part of the optimisation
    model, not a solver setting, and remains identical to the evaluation model.
    """

    version = ScenarioMPC.continuous_version + "-differentiable-v3-library-defaults"

    def __init__(self):
        super().__init__()
        self.reference = ScenarioMPC(solver_defaults=True, battery_exclusive=False)
        self.parameter_names = tuple(self.reference.p)
        self.variable_names = tuple(self.reference.v)
        parameter_ids = {id(value): key for key, value in self.reference.p.items()}
        parameter_ids[id(self.reference.caps)] = "caps"
        # Tariffs do not appear in the four priority-only objectives, so their
        # canonical problems legitimately have fewer parameters than economics.
        self.stage_parameter_names = [
            tuple(parameter_ids[id(value)] for value in problem.parameters())
            for problem in self.reference.problems
        ]
        self.layers = nn.ModuleList([
            CvxpyLayer(problem,
                       parameters=problem.parameters(),
                       variables=list(self.reference.v.values()))
            for problem in self.reference.problems
        ])
        self.last_max_constraint_violation = 0.0
        self.last_stage_indices: tuple[int, ...] = ()
        self.last_stage_caps: dict[int, torch.Tensor] = {}
        self.last_stage_objectives: dict[int, torch.Tensor] = {}
        self.last_stage_solvers: dict[int, str] = {}
        self.last_solver_attempts: dict[int, list[dict[str, object]]] = {}
        self.solver_statistics: dict[str, int] = {}
        self.warning_statistics: dict[str, int] = {}
        self.maximum_observed_constraint_violation = 0.0
        self.last_failure = None

    def _solve_stage(self, parameters, caps, stage):
        """Deterministic, bounded retries of the exact same optimisation.

        The retry order depends only on solver success for the current inputs;
        it never depends on a cached previous choice, random state or gradients.
        Thus checkpoint recomputation follows the same route for equal inputs.
        Diagnostics contain numbers/strings only, never retained solution graphs.
        """
        attempts = (
            ("ECOS", dict(solve_method="ECOS")),
            ("SCS", dict(solve_method="SCS")),
        )
        values = [caps if key == "caps" else parameters[key]
                  for key in self.stage_parameter_names[stage]]
        failed = []
        last_error = None
        for name, options in attempts:
            messages = []
            try:
                with warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter("always")
                    try:
                        output = self.layers[stage](*values, solver_args=options)
                    finally:
                        messages = [str(w.message) for w in notices]
                        for message in messages:
                            key = f"{name}: {message}"
                            self.warning_statistics[key] = self.warning_statistics.get(key, 0) + 1
                            if self.warning_statistics[key] == 1:
                                logging.getLogger(__name__).warning(
                                    "Differentiable MPC stage %d, %s: %s; subsequent occurrences counted",
                                    stage, name, message)
                raw = dict(zip(self.variable_names, output))
                violation = self._validate(parameters, caps, raw, stage)
            except Exception as exc:
                failed.append(dict(solver=name, error=str(exc), warnings=messages))
                last_error = exc
                continue
            self.last_stage_solvers[stage] = name
            self.last_solver_attempts[stage] = failed + [dict(
                solver=name, status="returned_with_warning" if messages else "returned",
                warnings=messages, constraint_violation=violation)]
            self.solver_statistics[name] = self.solver_statistics.get(name, 0) + 1
            self.maximum_observed_constraint_violation = max(
                self.maximum_observed_constraint_violation, violation)
            if failed:
                logging.getLogger(__name__).info(
                    "Differentiable MPC stage %d: %s accepted after %s; constraint %.3g",
                    stage, name, failed, violation)
            return raw, violation
        self.last_solver_attempts[stage] = failed
        self.last_failure = dict(
            controller_version=self.version, stage=stage, attempts=failed,
            solver_options={name: options for name, options in attempts},
            parameters={key: value.detach().tolist() for key, value in parameters.items()},
            caps=caps.detach().tolist(), accuracy_policy='library_defaults')
        raise RuntimeError(f"Differentiable MPC failed all gradient-preserving solvers at stage "
                           f"{stage}: {failed}") from last_error

    def _validate(self, parameters: Mapping[str, torch.Tensor], caps: torch.Tensor,
                  solution: Mapping[str, torch.Tensor], stage: int) -> float:
        # This detached copy is used only to inspect exactly the source
        # CVXPY constraints; it is never returned or used to form a later cap.
        if set(solution) != set(self.variable_names):
            raise RuntimeError(f"Incomplete differentiable MPC output at stage {stage}")
        for key, value in parameters.items():
            self.reference.p[key].value = value.detach().numpy()
        self.reference.caps.value = caps.detach().numpy()
        for key, value in solution.items():
            detached = value.detach().numpy()
            if not np.isfinite(detached).all():
                raise RuntimeError(f"Non-finite differentiable MPC stage {stage}: {key}")
            # Assign without projection: a negative conic-solver residue is a
            # violation to measure, not a reason to alter the returned gradient.
            self.reference.v[key].save_value(detached)
        violation = max(float(np.max(c.violation()))
                        for c in self.reference.problems[stage].constraints)
        # Variable domain constraints are implicit in CVXPY attributes.
        domain_violation = max(max(0.0, -float(value.detach().min()))
                               for key, value in solution.items() if key != "soc")
        violation = max(violation, domain_violation)
        if not np.isfinite(violation):
            raise RuntimeError(f"Non-finite differentiable MPC residual at stage {stage}")
        return violation

    def forward(self, forecast, future: Mapping[str, torch.Tensor], initial,
                reserve) -> dict[str, torch.Tensor]:
        p = {"pv": _vector(forecast, H, "forecast").clamp_min(0.0)}
        for key in TIERS + ("import_price", "export_price", "grid"):
            p[key] = _vector(future[key], H, key)
            if torch.any(p[key] < 0):
                raise ValueError(f"{key} must be nonnegative")
        if torch.any(p["grid"] > 1):
            raise ValueError("grid must lie in [0, 1]")
        p["export_price"] = torch.minimum(
            p["export_price"],
            (p["import_price"] - C.no_arbitrage_margin_CNY_per_kWh).clamp_min(0.0))
        p["initial"] = _vector(initial, 1, "initial").reshape(()).clamp(
            C.min_soc_kWh, C.max_soc_kWh)
        reserve = _vector(reserve, H + 1, "reserve")
        if torch.any(reserve < 0):
            raise ValueError("reserve must be nonnegative")
        bounded = reserve[1:].clamp_max(C.max_soc_kWh)
        p["reserve"] = bounded
        caps = torch.stack([DT * p[key].sum() for key in TIERS] + [
            (bounded - C.min_soc_kWh).clamp_min(0.0).sum() + 1e-4])
        all_grid = bool(torch.all(p["grid"] == 1))
        if all_grid:
            caps = torch.cat((caps[:3] * 0.0, caps[3:]))
        stages = ([] if all_grid else [0, 1, 2])
        if bool(torch.any(bounded > C.min_soc_kWh + 1e-6)):
            stages.append(3)
        else:
            caps = torch.cat((caps[:3], caps[3:] * 0.0))
        stages.append(4)
        self.last_stage_indices = tuple(stages)
        self.last_stage_caps = {}
        self.last_stage_objectives = {}
        self.last_stage_solvers = {}
        self.last_solver_attempts = {}
        self.last_failure = None
        violations = []
        for stage in stages:
            self.last_stage_caps[stage] = caps
            raw, violation = self._solve_stage(p, caps, stage)
            violations.append(violation)
            if stage < 4:
                objective = (DT * raw["unmet_" + TIERS[stage]].sum()
                             if stage < 3 else raw["reserve_shortfall"].sum())
                self.last_stage_objectives[stage] = objective
                # Exact evaluation-controller tolerance; no .item(), float(),
                # NumPy conversion or detach is allowed along this path.
                next_cap = objective.clamp_min(0.0) + 1e-6
                caps = torch.cat((caps[:stage], next_cap.reshape(1), caps[stage + 1:]))
            else:
                self.last_stage_objectives[stage] = DT * (
                    (p["import_price"] * raw["grid_import"]
                     - p["export_price"] * raw["grid_export"]).sum()
                    + C.objective_weights["battery_cycle"]
                    * (raw["charge"] + raw["discharge"]).sum()
                    + C.objective_weights["pv_curtailment"] * raw["pv_curtailment"].sum())
        self.last_max_constraint_violation = max(violations)
        arrays = {key: value.clamp_min(0.0) for key, value in raw.items()}
        net = arrays["grid_import"] - arrays["grid_export"]
        arrays["grid_import"] = net.clamp_min(0.0)
        arrays["grid_export"] = (-net).clamp_min(0.0)
        return arrays

    solve = forward
