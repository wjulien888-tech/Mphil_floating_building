"""Solve CVXPY's linear canonical model with native Gurobi variable bounds.

CVXPY 1.3 emits nonnegative domains and scalar upper limits as matrix rows.
Move one-nonzero inequalities to equivalent variable bounds; preserve every
other matrix row. No horizon, integrality, objective or feasible set is relaxed.
Use CVXPY's own inverse reductions to restore the original named variables.
"""
import numpy as np
import cvxpy as cp


def solve_with_native_bounds(problem, **options):
    import gurobipy as gp
    from cvxpy import settings as s
    data, chain, inverse = problem.get_problem_data(cp.GUROBI)
    if data[s.P].nnz:
        raise ValueError('Native-bound backend supports the linear MPC only')
    q = np.asarray(data[s.Q]).reshape(-1)
    n = len(q)
    lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    types = np.full(n, gp.GRB.CONTINUOUS, dtype='U1')
    for key, kind in ((s.BOOL_IDX, gp.GRB.BINARY), (s.INT_IDX, gp.GRB.INTEGER)):
        for i in data[key]:
            types[i] = kind
            if kind == gp.GRB.BINARY:
                lower[i], upper[i] = 0., 1.
    f = data[s.F].tocsr(copy=True)
    f.eliminate_zeros()
    rhs = np.asarray(data[s.G]).reshape(-1)
    single = np.diff(f.indptr) == 1
    for row in np.flatnonzero(single):
        pos = f.indptr[row]
        column, coefficient = f.indices[pos], f.data[pos]
        bound = rhs[row] / coefficient
        if coefficient > 0:
            upper[column] = min(upper[column], bound)
        else:
            lower[column] = max(lower[column], bound)
    # Keep inconsistent bounds: Gurobi must diagnose infeasibility, not project.
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        for key, value in options.items():
            model.setParam(key, value)
        x = model.addMVar(n, lb=lower, ub=upper, vtype=types)
        a = data[s.A].tocsr()
        if a.shape[0]:
            model.addMConstr(a, x, gp.GRB.EQUAL, np.asarray(data[s.B]).reshape(-1))
        if np.any(~single):
            model.addMConstr(f[~single], x, gp.GRB.LESS_EQUAL, rhs[~single])
        model.setObjective(q @ x, gp.GRB.MINIMIZE)
        model.optimize()
        details = dict(variables=n, equality_rows=a.shape[0],
            inequality_rows=int((~single).sum()), bound_rows_compacted=int(single.sum()),
            gurobi_status=int(model.Status))
        # GUROBI.invert expects the native model and maps its solution through
        # the original CVXPY reduction chain. Domain checks stay in _solve.
        problem.unpack_results({'model': model}, chain, inverse)
    return details
