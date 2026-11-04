"""Shared PuLP solver selection for MILP/MPC scripts."""

from __future__ import annotations


def build_pulp_solver(
    pulp,
    solver_name: str = "gurobi",
    solver_log: bool = False,
    time_limit_s: int | None = None,
    mip_gap: float | None = None,
):
    solver_name = solver_name.lower()

    if solver_name == "gurobi":
        gurobi_api = getattr(pulp, "GUROBI", None)
        if gurobi_api is not None:
            kwargs = {"msg": solver_log, "timeLimit": time_limit_s}
            if mip_gap is not None:
                kwargs["gapRel"] = mip_gap
            solver = gurobi_api(**kwargs)
            if solver.available():
                return solver

        gurobi_cmd = getattr(pulp, "GUROBI_CMD", None)
        if gurobi_cmd is not None:
            solver = gurobi_cmd(
                msg=solver_log,
                timeLimit=time_limit_s,
                gapRel=mip_gap,
            )
            if solver.available():
                return solver

        raise RuntimeError(
            "Gurobi solver is not available to PuLP. Install/configure Gurobi first "
            "(gurobipy or gurobi_cl plus a valid license), or rerun with --solver cbc."
        )

    if solver_name == "cbc":
        return pulp.PULP_CBC_CMD(
            msg=solver_log,
            timeLimit=time_limit_s,
            gapRel=mip_gap,
        )

    raise ValueError(f"Unsupported solver: {solver_name}. Use 'gurobi' or 'cbc'.")
