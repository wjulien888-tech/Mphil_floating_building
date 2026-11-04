#!/usr/bin/env python3
"""Run a frozen M0 through a versioned S0--S5 matrix; one runner, many cases."""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime
import csv
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
import numpy as np
import pandas as pd
import torch

import run_m0_simulation as common
from m0_scenarios import ScenarioMPC, load_cases, resolve_low_events, scenario_inputs, reserve_profile, execute, TIERS, DT, H

ROOT=common.ROOT
DEFAULT_CONFIG=ROOT/'configs/m0_s0_s5_v2.json'


def dump(path, obj):
    # Atomic progress updates survive interruption during a write.
    tmp=path.with_suffix(path.suffix+'.tmp')
    common.write_json(tmp,obj)
    tmp.replace(path)


def fingerprint(args, config):
    files=[Path(__file__),ROOT/'tools/m0_scenarios.py',ROOT/'tools/run_m0_simulation.py',ROOT/'tools/_hardware_profile.py',ROOT/'tools/_gurobi_bounded_lp.py',
           ROOT/'tools/export_m0_report_metrics.py',
           ROOT/'external/ol-ems-hkust/main.py',ROOT/'external/ol-ems-hkust/configurations/hkust_15min.py',
           ROOT/'external/ol-ems-hkust/src/plants/pymgrid_adapter.py',
           ROOT/'external/ol-ems-hkust/src/models/controller.py',
           ROOT/'model_variants/gru_2layer_m0.py',ROOT/'model_variants/lstm_2layer_m0.py']
    # Canonical JSON types keep the identity equal after manifest reload.
    hardware=json.loads(json.dumps(asdict(common.HARDWARE),default=str))
    return dict(config=config,hardware=hardware,controller_version=ScenarioMPC.version,
        milp_solver=os.environ.get('MPC_MILP_SOLVER','gurobi'),checkpoint_sha256=common.sha256_file(args.checkpoint),
        dispatch_sha256=common.sha256_file(args.dispatch),tiers_sha256=common.sha256_file(args.load_tiers),
        code_sha256={str(p.relative_to(ROOT)):common.sha256_file(p) for p in files},
        invalid_pv=args.invalid_pv,smoke_steps=args.smoke_steps)


def prepare(args, config):
    steps=int(config['days']*96)
    model,dataset,checkpoint=common.load_model(args.checkpoint,evaluation_start=config['start'],evaluation_steps=steps)
    frame,histories,repairs=common.prepare_inputs(dataset,config['start'],steps,args.invalid_pv,args.dispatch,args.load_tiers)
    tiers=pd.read_csv(args.load_tiers,parse_dates=['datetime']).set_index('datetime').reindex(frame.index)
    for k in TIERS:
        frame[k]=tiers['minimum_operation_'+k.upper()+'_kW']
    if not np.isfinite(frame[list(TIERS)].to_numpy()).all() or (frame[list(TIERS)]<0).any().any():
        raise ValueError('Invalid tier data')
    if not np.allclose(frame[list(TIERS)].sum(axis=1),frame.load,atol=1e-6,rtol=0):
        raise ValueError('CL1+CL2+CL3 must equal total load at every timestamp')
    return model,dataset,checkpoint,frame,histories,repairs


def run_case(args,case,model,dataset,base,base_history,repairs,folder,stop,case_number,total_cases,initial_soc_kwh=None):
    folder.mkdir(parents=True,exist_ok=False)
    frame,histories=scenario_inputs(base,base_history,dataset,case)
    requested=int(args.days*96)
    steps=min(requested,args.smoke_steps) if args.smoke_steps else requested
    logging.basicConfig(filename=folder/'run.log',level=logging.INFO,force=True)
    logging.captureWarnings(True)
    repairs.to_csv(folder/'source_pv_replacements.csv')
    common.write_json(folder/'case.json',dict(case=asdict(case),controller_version=ScenarioMPC.version,
        allow_grid_charging=common.HARDWARE.allow_grid_charging,
        actual_start=str(frame.index[0]),requested_steps=requested,executed_step_limit=steps,
        incomplete_protocol=bool(args.smoke_steps),
        reserve_proxy='For duration >24h, repeat current 96-step predicted PV and load profile. Never use future actual PV.',
        reserve_priority='CL1 ENS, CL2 ENS, CL3 ENS, attainable reserve shortage, economic cost; lexicographic tolerance 1e-6 per stage.',
        low_pv='Unmodified source PV and history. Low-PV outage timestamps selected offline from observed contiguous windows.',
        initial_soc_meaning=('Carried from preceding month; reset only at annual trajectory start.' if initial_soc_kwh is not None
                             else f'SOC at start of the {args.days:g}-day run; outage-start SOC is a result, never reset.')))
    totals={k:0. for k in ('grid_import_kwh','grid_export_kwh','grid_cost','component_cost_sum','battery_charge_kwh',
        'battery_discharge_kwh','pv_curtailed_kwh','loss_load_kwh','cl1_ens_kwh','cl2_ens_kwh','cl3_ens_kwh')}
    outage_ens={k:0. for k in TIERS};outage_load={k:0. for k in TIERS}
    first_loss={k:None for k in TIERS};longest={k:0 for k in TIERS};streak={k:0 for k in TIERS}
    completed=0;status='running';error=None;outage_soc=None;fallback=0;max_violation=0.
    recent=deque(maxlen=args.history_steps)
    began=time.perf_counter();adapter=common._build_adapter(common.HARDWARE)
    start_energy = case.initial_soc*common.HARDWARE.battery_capacity_kWh if initial_soc_kwh is None else initial_soc_kwh
    adapter.reset(soc_kwh=start_energy)
    max_reserve_gap=0.;max_hardware_gap=0.;warning_gap_steps=0
    try:
        mpc=ScenarioMPC();aw=pw=None
        with (folder/'actual.csv').open('w',newline='') as af, gzip.open(folder/'plans.csv.gz','wt',newline='') as pf, common.TerminalView(not args.no_live) as view:
            for i in range(steps):
                if stop.is_set():break
                tick=time.perf_counter();hour=i*DT
                known_grid,warning,active=case.information(hour)
                with torch.no_grad():
                    forecast=model.forecast_pv(histories[i:i+dataset.lookback_window].unsqueeze(0)).numpy().reshape(-1)
                prediction_seconds=time.perf_counter()-tick
                future={k:frame[k].iloc[i:i+H].to_numpy() for k in TIERS+('import_price','export_price')}
                future['grid']=known_grid
                reserve=reserve_profile(case,hour,forecast,frame.load.iloc[i:i+H].to_numpy())
                arrays,violation,solver,stages=mpc.solve(forecast,future,adapter.get_soc_kwh(),reserve)
                solved=time.perf_counter()
                actual={k:float(frame.iloc[i][k]) for k in ('pv',)+common.FUTURE}
                row=execute(adapter,arrays,actual,{k:float(frame.iloc[i][k]) for k in TIERS})
                row['battery_charge_mode']=int(arrays['battery_charge_mode'][0])
                balance,soc_error=common.verify_execution(row)
                if abs(row['load_kwh']-row['load_served_kwh']-row['loss_load_kwh'])>1e-5:
                    raise RuntimeError('Tier energy ledger does not reconcile')
                if actual['grid_available']==0 and (row['grid_import_kwh']+row['grid_export_kwh'])>1e-6:
                    raise RuntimeError('Grid exchanged energy during outage')
                reserve_gap=max(0.,reserve[0]-row['soc_before_kwh']) if active else 0.
                hardware_gap=max(0.,reserve[0]-common.HARDWARE.max_soc_kWh) if active else 0.
                max_reserve_gap=max(max_reserve_gap,reserve_gap);max_hardware_gap=max(max_hardware_gap,hardware_gap)
                warning_gap_steps+=int(warning and reserve_gap>1e-4)
                row.update(timestamp=str(frame.index[i]),decision_step=i,forecast_pv_kw=float(forecast[0]),actual_pv_kw=actual['pv'],
                    warning_active=warning,reserve_active=active,reserve_target_kwh=float(reserve[0]),reserve_gap_kwh=reserve_gap,
                    reserve_hardware_gap_kwh=hardware_gap,mpc_solver=solver,mpc_economic_solver=mpc.last_economic_solver,mpc_stages=stages,mpc_max_constraint_violation=violation,
                    balance_residual_kwh=balance,soc_error_kwh=soc_error,prediction_seconds=prediction_seconds,
                    mpc_seconds=solved-tick-prediction_seconds,step_seconds=time.perf_counter()-tick)
                for quality in ('pv_reference_valid','pv_imputed','load_imputed'):
                    if quality in frame: row[quality] = int(frame.iloc[i][quality])
                if aw is None:aw=csv.DictWriter(af,fieldnames=list(row));aw.writeheader()
                aw.writerow(row);af.flush()
                # Avoid keeping a month of horizon plans in RAM.
                plan=pd.DataFrame({k:val[:H] for k,val in arrays.items()})
                plan['soc_end']=arrays['soc'][1:];plan['pv_forecast_kw']=forecast
                plan['reserve_target_kwh']=reserve[1:]
                plan['known_grid_available']=known_grid
                plan['decision_step']=i;plan['decision_timestamp']=str(frame.index[i]);plan['horizon_timestamp']=frame.index[i:i+H].astype(str)
                for k in TIERS+('import_price','export_price'):plan[k]=future[k]
                if pw is None:pw=csv.writer(pf);pw.writerow(plan.columns)
                pw.writerows(plan.itertuples(index=False,name=None));pf.flush()
                completed=i+1
                for k in totals:totals[k]+=row[k]
                if actual['grid_available']==0:
                    if outage_soc is None:outage_soc=row['soc_before_kwh']
                    for k in TIERS:
                        outage_ens[k]+=row[k+'_ens_kwh'];outage_load[k]+=row[k+'_load_kwh']
                        lost=row[k+'_ens_kwh']>1e-5
                        if lost and first_loss[k] is None:first_loss[k]=hour-case.event_hour
                        streak[k]=streak[k]+1 if lost else 0;longest[k]=max(longest[k],streak[k])
                else:
                    for k in TIERS:streak[k]=0
                fallback+=int(mpc.last_economic_solver=='HiGHS');max_violation=max(max_violation,violation)
                recent.append(100*row['soc_after_fraction']);elapsed=time.perf_counter()-began
                view.publish(dict(method='GRU/LSTM',scenario=f'{case.group} {case_number}/{total_cases}',step=completed,total=steps,
                    timestamp=str(frame.index[i]),elapsed=elapsed,eta=elapsed/completed*(steps-completed),seconds=time.perf_counter()-tick,
                    soc=recent[-1],purchase=totals['grid_import_kwh'],bill=totals['grid_cost'],
                    event_status=f'{case.name} | grid={int(actual["grid_available"])} warn={int(warning)} | critical ENS={totals["cl1_ens_kwh"]+totals["cl2_ens_kwh"]:.2f} kWh',
                    plan_purchase=float(arrays['grid_import'].sum()*DT),history=list(recent),charts=[
                    ('Charge',arrays['charge'].tolist(),'kW'),('Discharge',arrays['discharge'].tolist(),'kW'),
                    ('SOC',(arrays['soc']/common.HARDWARE.battery_capacity_kWh*100).tolist(),'%'),
                    ('Purchase',(arrays['grid_import']*DT).tolist(),'kWh / 15min')]))
                if not view.enabled and (completed==1 or completed%384==0):
                    print(f'{case.name} {completed}/{steps} | {elapsed:.1f}s | SOC {recent[-1]:.1f}% | critical ENS {outage_ens["cl1"]+outage_ens["cl2"]:.3f}',flush=True)
                if completed%96==0:dump(folder/'status.json',dict(status='running',completed_steps=completed,**totals))
        status=('smoke_completed' if args.smoke_steps else 'completed') if completed==steps else 'interrupted'
    except Exception:
        status='failed';error=traceback.format_exc();logging.exception('Case failed')
    summary=dict(status=status,case=case.name,group=case.group,completed_steps=completed,requested_steps=requested,
        allow_grid_charging=common.HARDWARE.allow_grid_charging,
        elapsed_seconds=time.perf_counter()-began,**totals,initial_soc_kwh=start_energy,
        final_soc_kwh=adapter.get_soc_kwh(),outage_start_soc_kwh=outage_soc,max_reserve_gap_kwh=max_reserve_gap,
        max_reserve_hardware_gap_kwh=max_hardware_gap,warning_reserve_gap_steps=warning_gap_steps,
        fallback_steps=fallback,max_constraint_violation=max_violation,error=error)
    for k in TIERS:
        summary['outage_'+k+'_ens_kwh']=outage_ens[k]
        summary['outage_'+k+'_load_kwh']=outage_load[k]
        summary['outage_'+k+'_supply_fraction']=1-outage_ens[k]/outage_load[k] if outage_load[k]>0 else None
        summary['outage_'+k+'_first_loss_hours']=first_loss[k]
        summary['outage_'+k+'_longest_loss_hours']=longest[k]*DT
    dump(folder/'summary.json',summary);dump(folder/'status.json',summary)
    print(f'{case.name}: {status} {completed}/{requested} -> {folder}',flush=True)
    return summary


def main(args):
    config=json.loads(args.config.read_text())
    if config.get('protocol_version')!=ScenarioMPC.protocol_version:raise ValueError('Unknown protocol version')
    cases=load_cases(config)
    torch.set_num_threads(1)
    model,dataset,checkpoint,frame,histories,repairs=prepare(args,config)
    cases,event_records=resolve_low_events(cases,frame,config['days'],repairs)
    selected=[c for c in cases if (args.scenarios==['all'] or c.group in args.scenarios) and (not args.case or c.name in args.case)]
    if not selected:raise ValueError('No cases selected')
    if args.case and set(args.case)-{c.name for c in selected}:raise ValueError('Unknown/unselected case')
    if args.history_steps<1 or (args.smoke_steps is not None and args.smoke_steps<1):raise ValueError('Invalid step count')
    args.days=config['days']
    print(f'{len(selected)} selected cases; {config["days"]} days / case; controller {ScenarioMPC.version}',flush=True)
    print(f'Grid charging: {"MPC decides" if common.HARDWARE.allow_grid_charging else "disabled"}',flush=True)
    for case in selected:
        event_time=frame.index[0]+pd.Timedelta(hours=case.event_hour)
        print(f'  {case.name}: outage={case.outage_hours:g}h warning={case.warning_hours:g}h SOC0={case.initial_soc:g} rho={case.rho:g} PV={case.pv_regime} event={event_time}',flush=True)
    if args.list:return
    if args.check_only:
        print(f'Preflight passed: all selected event windows fit; {len(repairs)} source PV replacements.');return
    identity=fingerprint(args,config)
    identity['resolved_cases']=[asdict(c) for c in cases]
    identity['event_selection']=event_records
    if args.resume:
        output=args.resume.resolve();stored=json.loads((output/'manifest.json').read_text())
        if stored['identity']!=identity:raise ValueError('Resume rejected: code/config/data/checkpoint changed. Create a new output directory.')
        state=json.loads((output/'suite_status.json').read_text())
    else:
        output=(args.output or ROOT/'experiments'/('m0_s0_s5_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
        output.mkdir(parents=True,exist_ok=False)
        dump(output/'manifest.json',dict(identity=identity,all_cases=[asdict(c) for c in cases],pv_identity=checkpoint['dataset_identity'],
            controller_version=ScenarioMPC.version,python=sys.version,torch=torch.__version__,
            allow_grid_charging=common.HARDWARE.allow_grid_charging,
            statistics='Deterministic scenario sweep; no probability weights, no population EENS or CVaR claim.'))
        state=dict(status='created',cases={c.name:dict(status='pending') for c in cases})
        dump(output/'suite_status.json',state)
        dump(output/'event_selection.json',event_records)
    print(f'Suite output: {output}',flush=True)
    stop=threading.Event();previous=signal.signal(signal.SIGINT,lambda *_:stop.set())
    try:
        state['status']='running';dump(output/'suite_status.json',state)
        for number,case in enumerate(selected,1):
            if stop.is_set():break
            record=state['cases'][case.name]
            if record['status']=='completed':
                saved=json.loads((Path(record['path'])/'summary.json').read_text())
                if saved['status']!='completed' or saved['completed_steps']!=int(config['days']*96):raise ValueError('Completed case summary is inconsistent')
                print(f'Skip completed: {case.name}',flush=True);continue
            attempts=output/case.name;attempts.mkdir(exist_ok=True)
            folder=attempts/('attempt_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            state['cases'][case.name]=dict(status='running',path=str(folder));dump(output/'suite_status.json',state)
            result=run_case(args,case,model,dataset,frame,histories,repairs,folder,stop,number,len(selected))
            state['cases'][case.name]=dict(status=result['status'],path=str(folder))
            dump(output/'suite_status.json',state)
            if result['status'] not in ('completed','smoke_completed'):
                if result['error']:raise RuntimeError(result['error'])
                break
        good={'completed','smoke_completed'}
        state['status']='selected_completed' if all(state['cases'][c.name]['status'] in good for c in selected) else 'interrupted'
        state['full_matrix_completed']=all(r['status']=='completed' for r in state['cases'].values())
    except Exception:
        state['status']='failed';raise
    finally:
        signal.signal(signal.SIGINT,previous);dump(output/'suite_status.json',state)
        summaries=[]
        for name,record in state['cases'].items():
            if record.get('path') and (Path(record['path'])/'summary.json').exists():
                summary=json.loads((Path(record['path'])/'summary.json').read_text());summary['result_path']=record['path'];summaries.append(summary)
        if summaries:pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
        print(f'Suite {state["status"]}: {output}',flush=True)
    if state.get('full_matrix_completed'):
        from export_m0_report_metrics import export_report_metrics
        metrics = export_report_metrics(output)
        print(f'Report metrics: {output / "report_data"} ({metrics["case_count"]} cases)', flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--scenarios',nargs='+',choices=['all','S0','S1','S2','S3','S4','S5'],default=['all'])
    p.add_argument('--case',action='append',help='Exact case name from --list; repeat to select several')
    p.add_argument('--checkpoint',type=Path,default=common.DEFAULT_CHECKPOINT)
    p.add_argument('--dispatch',type=Path,default=ROOT/'output/frozen/hkust_m0_v1_20260909/dispatch.csv')
    p.add_argument('--load-tiers',type=Path,default=ROOT/'output/frozen/hkust_m0_v1_20260909/load_tiers.csv')
    p.add_argument('--invalid-pv',choices=['error','hold-last'],default='error')
    p.add_argument('--history-steps',type=int,default=96)
    g=p.add_mutually_exclusive_group();g.add_argument('--output',type=Path);g.add_argument('--resume',type=Path)
    p.add_argument('--no-live',action='store_true');p.add_argument('--list',action='store_true');p.add_argument('--check-only',action='store_true')
    p.add_argument('--smoke-steps',type=int,help='Validation only: truncated cases are NEVER marked formally completed')
    return p

if __name__=='__main__':main(parser().parse_args())
