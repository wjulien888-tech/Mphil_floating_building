"""Versioned S0--S5 events, information boundary and hierarchical test MPC.

Default evaluation is MILP with binary battery modes. M1 explicitly requests
the continuous relaxation for differentiable training and matched validation.
"""
from dataclasses import dataclass, asdict, replace
from itertools import product
import logging
import os
import numpy as np
import pandas as pd
import cvxpy as cp
from run_m0_simulation import HARDWARE as C

TIERS = ('cl1', 'cl2', 'cl3')
DT = .25
H = 96


@dataclass(frozen=True)
class Scenario:
    name: str
    group: str
    event_hour: float = 228.
    outage_hours: float = 0.
    announced_hours: float = 0.
    warning_hours: float = 0.
    initial_soc: float = .5
    rho: float = 1.
    pv_regime: str = 'original'
    selection_hours: float = 0.
    selection_clock: float = 12.

    def validate(self, days):
        if self.group not in ('S0','S1','S2','S3','S4','S5'):
            raise ValueError(f'Unknown group {self.group}')
        if not C.min_soc_fraction <= self.initial_soc <= C.max_soc_fraction:
            raise ValueError('Initial SOC outside battery bounds')
        if not 0 < self.rho <= 1 or self.pv_regime not in ('original','observed_low'):
            raise ValueError('Invalid rho or PV regime; artificial scaling is unsupported')
        for value in (self.event_hour,self.outage_hours,self.announced_hours,self.warning_hours,self.selection_hours,self.selection_clock):
            if not np.isfinite(value) or value < 0 or not float(value*4).is_integer():
                raise ValueError('Event durations/offsets must be nonnegative multiples of 0.25h')
        if self.outage_hours and self.outage_hours != self.announced_hours:
            raise ValueError('This protocol assumes accurate announced duration for actual outages')
        if self.announced_hours and not self.warning_hours:
            raise ValueError('This protocol requires a warning for announced events')
        if self.announced_hours and (self.event_hour+self.announced_hours+24 > days*24):
            raise ValueError('Each event must include at least 24 hours after its announced end')

    def actual_grid(self, hours):
        return np.where((hours >= self.event_hour) & (hours < self.event_hour+self.outage_hours),0.,1.)

    def information(self, hour):
        """No future actual outage flag is passed before warning issuance.

        False warnings and true warnings are identical before t0. At t0 a
        false warning is cancelled using the observed connected grid state.
        During a true outage the previously announced restoration is known.
        """
        future_hours = hour + np.arange(H)*DT
        warning_start = max(0., self.event_hour-self.warning_hours)
        warning = self.announced_hours > 0 and warning_start <= hour < self.event_hour
        islanded = self.outage_hours > 0 and self.event_hour <= hour < self.event_hour+self.outage_hours
        active = warning or islanded
        grid = np.ones(H)
        if active:
            grid[(future_hours >= self.event_hour)&(future_hours < self.event_hour+self.announced_hours)] = 0
        return grid, warning, active


def load_cases(config):
    if config.get('protocol_version') != 'm0-scenarios-v2':
        raise ValueError('Withdrawn/unknown scenario protocol; use m0-scenarios-v2')
    cases = [Scenario(**row) for row in config['cases']]
    grid = config['s5_grid']
    for soc,rho,hour,duration,pv in product(grid['initial_soc'],grid['rho'],grid['event_hour'],grid['outage_hours'],grid['pv_regime']):
        name=f'S5_soc{soc:g}_rho{rho:g}_t{hour:g}_D{duration:g}_{pv}'.replace('.','p')
        cases.append(Scenario(name=name,group='S5',initial_soc=soc,rho=rho,event_hour=hour,
            outage_hours=duration,announced_hours=duration,warning_hours=grid['warning_hours'],
            pv_regime=pv,selection_hours=duration if pv=='observed_low' else 0,selection_clock=hour%24))
    if len({c.name for c in cases}) != len(cases):
        raise ValueError('Duplicate case names')
    for case in cases:
        if not case.name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for ch in case.name):
            raise ValueError('Unsafe case name')
        case.validate(config['days'])
    return cases


def resolve_low_events(cases, frame, days, repairs):
    """Select minimum-energy contiguous observed windows, never scale/copy PV.

    Use only the execution period, fixed local start clock, complete warning
    and 24 h recovery. Include causally repaired PV and record its count in
    each selected window. Ties go earliest.
    Offline event selection is not forecast information supplied to the model.
    """
    work=frame.iloc[:int(days*96)]
    if not work.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError('Event selection requires a continuous 15-minute sequence')
    resolved=[];records=[]
    for c in cases:
        if c.pv_regime!='observed_low':resolved.append(c);continue
        n=int(c.selection_hours/DT)
        if n<1:raise ValueError('Low-PV event needs selection_hours')
        candidates=[]
        for i,t in enumerate(work.index):
            if t.hour+t.minute/60 != c.selection_clock:continue
            if i*DT<max(24,c.warning_hours) or i+n+96>len(work):continue
            block=work.iloc[i:i+n]
            if not np.isfinite(block.pv).all() or (block.pv<0).any():continue
            candidates.append((float(block.pv.sum()*DT),i))
        if not candidates:raise ValueError(f'No valid observed low-PV window for {c.name}')
        energy,i=min(candidates)
        chosen=replace(c,event_hour=i*DT);chosen.validate(days);resolved.append(chosen)
        records.append(dict(case=c.name,start=str(work.index[i]),end_exclusive=str(work.index[i]+pd.Timedelta(hours=c.selection_hours)),
                            hours=c.selection_hours,pv_kwh=energy,load_kwh=float(work.load.iloc[i:i+n].sum()*DT),
                            candidate_count=len(candidates),start_clock=c.selection_clock,
                            repaired_pv_points=len(work.index[i:i+n].intersection(repairs.index)),
                            method='minimum contiguous PV energy; fixed clock; repaired samples included; earliest tie'))
    return resolved,records


def scenario_inputs(frame, histories, dataset, case):
    frame = frame.copy()
    hours = np.arange(len(frame))*DT
    frame['grid_available'] = case.actual_grid(hours)
    return frame, histories.clone()


def reserve_profile(case, hour, forecast, load):
    _,warning,active = case.information(hour)
    targets = np.full(H+1,C.min_soc_kWh)
    if not active:
        return targets
    # The network predicts only 24h. Longer reserve estimates repeat this
    # predicted daily profile, NOT future actual PV. This is a declared proxy.
    end = case.event_hour + case.announced_hours
    for j in range(H+1):
        t = hour+j*DT
        if t >= end:
            continue
        first = max(t,case.event_hour)
        offsets = np.rint((np.arange(first,end,DT)-hour)/DT).astype(int) % H
        need = np.maximum(case.rho*load[offsets]-forecast[offsets],0).sum()*DT/C.discharge_efficiency
        targets[j] += need
    return targets


def _solve(problem, priority=False, *, solver_defaults=False, diagnostics=None):
    errors=[]
    if problem.is_mixed_integer():
        requested = os.environ.get('MPC_MILP_SOLVER', 'gurobi').lower()
        if requested not in ('auto', 'gurobi', 'highs'):
            raise ValueError('MPC_MILP_SOLVER must be auto, gurobi or highs')
        gurobi = ('GUROBI', dict(solver=cp.GUROBI, MIPGap=0.0,
            MIPGapAbs=0.0, IntFeasTol=1e-9, FeasibilityTol=1e-9))
        highs = ('HiGHS-MILP', dict(solver=cp.SCIPY,
            scipy_options={'method': 'highs', 'mip_rel_gap': 0.0}))
        methods = ([gurobi] if requested == 'gurobi' else [highs]
                   if requested == 'highs' else
                   ([gurobi] if 'GUROBI' in cp.installed_solvers() else []) + [highs])
    elif solver_defaults:
        # M1 explicitly opts into library accuracy/iteration defaults. Keep the
        # existing solver selection; no custom precision or residual gate.
        highs=('HiGHS',dict(solver=cp.SCIPY,scipy_options={'method':'highs'}))
        methods=[highs] if priority else [('ECOS',dict(solver=cp.ECOS)),highs]
    else:
        # Retain the numerical policy used by the existing M0 experiments.
        highs=('HiGHS',dict(solver=cp.SCIPY,scipy_options={'method':'highs',
            'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}))
        methods=[highs] if priority else [
            ('ECOS',dict(solver=cp.ECOS,abstol=1e-8,reltol=1e-8,feastol=1e-8,max_iters=1000)),highs]
    for name,kwargs in methods:
        try:
            compact_details = None
            if name == 'GUROBI':
                from _gurobi_bounded_lp import solve_with_native_bounds
                compact_details = solve_with_native_bounds(problem, **{k: val for k, val in kwargs.items() if k != 'solver'})
            else:
                problem.solve(**kwargs)
            accepted=(cp.OPTIMAL,cp.OPTIMAL_INACCURATE) if solver_defaults and not problem.is_mixed_integer() else (cp.OPTIMAL,)
            if problem.status not in accepted:
                raise RuntimeError(problem.status)
            if not all(v.value is not None and np.isfinite(v.value).all() for v in problem.variables()):
                raise RuntimeError('nonfinite solution')
            violation=max(float(np.max(c.violation())) for c in problem.constraints)
            if problem.is_mixed_integer():
                # Check the raw Boolean solution, before returning any arrays.
                for var in problem.variables():
                    if var.attributes['boolean']:
                        raw = np.asarray(var.value)
                        if np.max(np.abs(raw - np.rint(raw))) > 1e-6:
                            raise RuntimeError('nonintegral battery mode')
            if not np.isfinite(violation) or ((not solver_defaults or problem.is_mixed_integer()) and violation > 1e-4):
                raise RuntimeError(f'constraint violation {violation}')
            if diagnostics is not None:
                diagnostics.update(solver=name,status=problem.status,
                                   constraint_violation=violation,solver_options=kwargs,
                                   native_bound_formulation=compact_details)
            return name,violation
        except Exception as exc:
            # Gurobi may raise its native GurobiError for license failures.
            # Non-solver errors from other backends must remain visible.
            if name != 'GUROBI' and not isinstance(exc, (cp.error.SolverError, RuntimeError)):
                raise
            errors.append(f'{name}: {exc}')
    raise RuntimeError('Solver attempts failed: '+ '; '.join(errors))


class ScenarioMPC:
    """L0 constraints, lexicographic CL1/CL2/CL3, reserve, then economics.

    All S0--S5 cases share this controller version; old S0 weighted-objective
    results must not be silently mixed into this version's matrix.
    """
    protocol_version='m0-scenarios-v2'
    version='m0-scenarios-v3-battery-milp'
    continuous_version='m0-scenarios-v2-continuous'
    def __init__(self, *, solver_defaults=False, battery_exclusive=True):
        self.solver_defaults=solver_defaults
        self.battery_exclusive=battery_exclusive
        self.version=(type(self).version if battery_exclusive else self.continuous_version)
        if solver_defaults:
            self.version+='-library-defaults'
        self.solver_statistics={}
        self.warning_statistics={}
        self.maximum_observed_constraint_violation=0.
        self.last_stage_results={}
        self.p={k:cp.Parameter(H,nonneg=True,name=k) for k in ('pv','cl1','cl2','cl3','import_price','export_price','grid')}
        self.p['initial']=cp.Parameter(nonneg=True)
        self.p['reserve']=cp.Parameter(H,nonneg=True)
        self.caps=cp.Parameter(4,nonneg=True)
        self.v={k:cp.Variable(H,nonneg=True,name=k) for k in ('charge','discharge','grid_import','grid_export','pv_used','pv_curtailment','unmet_cl1','unmet_cl2','unmet_cl3','reserve_shortfall')}
        self.v['soc']=cp.Variable(H+1,name='soc')
        if battery_exclusive:
            self.v['battery_charge_mode']=cp.Variable(H,boolean=True,name='battery_charge_mode')
        v,p=self.v,self.p
        served=sum(p[k]-v['unmet_'+k] for k in TIERS)
        con=[v['soc'][0]==p['initial'],v['soc']>=C.min_soc_kWh,v['soc']<=C.max_soc_kWh,
             v['soc'][1:]==v['soc'][:-1]+DT*C.charge_efficiency*v['charge']-DT/C.discharge_efficiency*v['discharge'],
             v['charge']<=C.battery_power_kW,v['discharge']<=C.battery_power_kW,
             v['grid_import']<=C.grid_import_limit_kW*p['grid'],v['grid_export']<=C.grid_export_limit_kW*p['grid'],
             v['pv_used']+v['pv_curtailment']==p['pv'],
             v['pv_used']+v['discharge']+v['grid_import']==served+v['charge']+v['grid_export'],
             v['reserve_shortfall']>=p['reserve']-v['soc'][1:]]
        if battery_exclusive:
            con += [v['charge']<=C.battery_power_kW*v['battery_charge_mode'],
                    v['discharge']<=C.battery_power_kW*(1-v['battery_charge_mode'])]
        if not C.allow_grid_charging:
            con += [v['charge']<=v['pv_used']]
        self.losses=[DT*cp.sum(v['unmet_'+k]) for k in TIERS]+[cp.sum(v['reserve_shortfall'])]
        for j,k in enumerate(TIERS):
            con += [v['unmet_'+k]<=p[k], self.losses[j]<=self.caps[j]]
        con += [self.losses[3]<=self.caps[3]]
        economic=DT*(cp.sum(cp.multiply(p['import_price'],v['grid_import'])-cp.multiply(p['export_price'],v['grid_export']))+
            C.objective_weights['battery_cycle']*cp.sum(v['charge']+v['discharge'])+
            C.objective_weights['pv_curtailment']*cp.sum(v['pv_curtailment']))
        self.problems=[cp.Problem(cp.Minimize(obj),con) for obj in self.losses+[economic]]

    def solve(self, forecast, future, initial, reserve):
        p,v=self.p,self.v
        p['pv'].value=np.maximum(np.asarray(forecast),0)
        for k in TIERS+('import_price',):p[k].value=np.asarray(future[k])
        p['export_price'].value=np.minimum(future['export_price'],np.maximum(future['import_price']-C.no_arbitrage_margin_CNY_per_kWh,0))
        p['grid'].value=future['grid']
        p['initial'].value=np.clip(initial,C.min_soc_kWh,C.max_soc_kWh)
        # Optimize attainable reserve; separately report uncapped hardware gap.
        bounded=np.minimum(reserve[1:],C.max_soc_kWh)
        p['reserve'].value=bounded
        caps=np.array([DT*np.sum(future[k]) for k in TIERS]+[np.maximum(bounded-C.min_soc_kWh,0).sum()+1e-4])
        solvers=[];violations=[]
        all_grid=np.all(np.asarray(future['grid'])==1)
        if all_grid:
            caps[:3]=0 # ample grid capacity, enforce full supply
        stages=([] if all_grid else [0,1,2])+([3] if np.any(bounded>C.min_soc_kWh+1e-6) else [])+[4]
        if 3 not in stages:caps[3]=0
        self.last_stage_results={}
        for stage in stages:
            self.caps.value=caps
            result={}
            solver,violation=_solve(self.problems[stage],priority=stage<4,
                solver_defaults=self.solver_defaults,diagnostics=result)
            self.last_stage_results[stage]=result
            self.solver_statistics[solver]=self.solver_statistics.get(solver,0)+1
            self.maximum_observed_constraint_violation=max(self.maximum_observed_constraint_violation,violation)
            if result['status']==cp.OPTIMAL_INACCURATE:
                key=solver+': '+result['status']
                self.warning_statistics[key]=self.warning_statistics.get(key,0)+1
                if self.warning_statistics[key]==1:
                    logging.getLogger(__name__).warning('MPC stage %d: %s; constraint residual %.6g',stage,key,violation)
            solvers.append(solver);violations.append(violation)
            if stage<4:caps[stage]=max(0.,float(self.losses[stage].value))+1e-6
        arrays={k:np.maximum(np.asarray(var.value).reshape(-1),0) for k,var in v.items()}
        if self.battery_exclusive:
            # Never round an infeasible schedule into a purported valid one.
            if np.max(np.minimum(arrays['charge'],arrays['discharge'])) > 1e-4:
                raise RuntimeError('MILP returned simultaneous battery charging/discharging')
            arrays['battery_charge_mode']=np.rint(arrays['battery_charge_mode']).astype(int)
        self.last_economic_solver=solvers[-1]
        net=arrays['grid_import']-arrays['grid_export']
        arrays['grid_import'],arrays['grid_export']=np.maximum(net,0),np.maximum(-net,0)
        return arrays,max(violations),'+'.join(sorted(set(solvers))),len(stages)


def execute(adapter, arrays, actual, tiers):
    """Apply scheduled tier shedding, then use real PV in python-microgrid.

    The adapter sees only loads scheduled for service, so it does not undo
    intentional shedding to charge for higher priority future loads. Real
    shortages in the admitted load are allocated CL3 first, then CL2, then CL1.
    """
    desired=np.array([tiers[k] for k in TIERS],dtype=float)
    shed=np.clip([arrays['unmet_'+k][0] for k in TIERS],0,desired)
    admitted=desired-shed
    actual_for_plant=dict(actual,load=float(admitted.sum()),critical_load=float(admitted[:2].sum()))
    plan={key:float(arrays[value][0]) for key,value in dict(battery_charge='charge',battery_discharge='discharge',grid_import='grid_import',grid_export='grid_export').items()}
    row=adapter.step(plan,actual_for_plant)
    available=row['load_served_kwh']/DT
    served=[]
    for demand in admitted:
        supply=min(demand,max(0.,available));served.append(supply);available-=supply
    unmet=np.maximum(desired-np.array(served),0)
    for j,k in enumerate(TIERS):
        row[k+'_load_kwh']=desired[j]*DT
        row[k+'_served_kwh']=served[j]*DT
        row[k+'_ens_kwh']=unmet[j]*DT
        row[k+'_scheduled_shed_kw']=shed[j]
    row.update(load_kw=float(desired.sum()),load_kwh=float(desired.sum()*DT),
        critical_load_kw=float(desired[:2].sum()),critical_load_kwh=float(desired[:2].sum()*DT),
        noncritical_load_kw=float(desired[2]),noncritical_load_kwh=float(desired[2]*DT),
        loss_load_kw=float(unmet.sum()),loss_load_kwh=float(unmet.sum()*DT),
        critical_loss_load_kw=float(unmet[:2].sum()),critical_loss_load_kwh=float(unmet[:2].sum()*DT),
        noncritical_loss_load_kw=float(unmet[2]),noncritical_loss_load_kwh=float(unmet[2]*DT))
    row['critical_loss_load_cost']=row['critical_loss_load_kwh']*adapter.critical_loss_load_cost_per_kwh
    row['noncritical_loss_load_cost']=row['noncritical_loss_load_kwh']*adapter.noncritical_loss_load_cost_per_kwh
    row['loss_load_cost']=row['critical_loss_load_cost']+row['noncritical_loss_load_cost']
    row['component_cost_sum']=sum(row[k] for k in ('grid_cost','battery_cost','loss_load_cost','pv_curtailment_cost','overgeneration_cost'))
    row['cost']=row['component_cost_sum'];row['reward']=-row['cost']
    return row
