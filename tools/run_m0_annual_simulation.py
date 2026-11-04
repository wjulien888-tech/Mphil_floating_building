"""One manual launch: 47 continuous annual trajectories, monthly M0 updates.

--check-only validates data/config/checkpoint files without inference or solves.
--resume reuses completed month/case artifacts and epoch-boundary checkpoints.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import gzip
import json
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import pandas as pd
import run_m0_annual_forecast as annual
import run_m0_scenarios as scenarios
import run_m0_simulation as common
from m0_scenarios import load_cases, resolve_low_events
from prepare_annual_simulation_inputs import DATA, OUTPUT as IMPUTED, verify, read, digest
from export_m0_report_metrics import export_report_metrics

CONFIG = ROOT/'configs/m0_annual_continuous_2023_2024.json'
RESULTS = ROOT/'experiments/annual_18_2_12_m0'


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    scenarios.dump(path,obj)


def preflight(args):
    imputation = verify(args.simulation_inputs,args.data)
    c = json.loads(args.config.read_text())
    if c['start'] != '2023-03-01 00:00:00' or c['days'] != 366:
        raise ValueError('Expected the agreed full test year')
    cases = load_cases(c)
    if len(cases) != 47: raise ValueError('Expected 47 annual trajectories')
    pv=read(args.simulation_inputs/'pv.csv')
    d=read(args.simulation_inputs/'dispatch.csv')
    tiers=read(args.simulation_inputs/'load_tiers.csv')
    needed=pd.date_range('2023-02-28', '2024-03-01 23:30:00',freq='15min')
    columns=['minimum_operation_CL1_kW','minimum_operation_CL2_kW','minimum_operation_CL3_kW']
    if (not np.isfinite(pv.reindex(needed).pv_kW).all()
            or not np.isfinite(d.reindex(needed)[['load_kW','import_price','export_price']]).all().all()
            or not d.reindex(needed).load_scenario_valid.eq(1).all()
            or not d.reindex(needed).grid_available.eq(1).all()
            or not np.allclose(tiers.reindex(needed)[columns].sum(axis=1),d.reindex(needed).load_kW,atol=1e-6,rtol=0)):
        raise ValueError('Annual simulation input preflight failed')
    frame=pd.DataFrame({'pv':pv.pv_kW,'load':d.load_kW}).loc['2023-03-01':]
    repairs=pv.loc[pv.pv_imputed.eq(1)]
    cases,events=resolve_low_events(cases,frame,366,repairs)
    _, identity, _, _ = annual.inputs(args.data,args.output)
    first=args.output/'2023-03/training/best.pt'
    checkpoint=annual.load_checkpoint(first)  # deserialize only, no model instantiated
    review=json.loads((first.parent.parent/'training_review.json').read_text())
    if checkpoint['dataset_identity']!=identity or digest(first)!=review['checkpoint_sha256']:
        raise ValueError('Initial model/data differ from reviewed training')
    if checkpoint['annual_fold']!=annual.select_fold('2023-03'):
        raise ValueError('Wrong initial checkpoint month')
    code={name:digest(ROOT/'tools'/name) for name in ['run_m0_annual_simulation.py',
        'run_m0_annual_forecast.py','run_m0_scenarios.py','run_m0_simulation.py',
        'm0_scenarios.py','prepare_annual_simulation_inputs.py','export_m0_report_metrics.py']}
    settings=scenarios.parser().parse_args(['--checkpoint',str(first),
        '--dispatch',str(args.simulation_inputs/'dispatch.csv'),'--load-tiers',str(args.simulation_inputs/'load_tiers.csv')])
    signature=scenarios.fingerprint(settings,c)
    signature.update(annual_code=code,simulation_manifest_sha256=digest(args.simulation_inputs/'manifest.json'),
        resolved_cases=[asdict(x) for x in cases],monthly_learning_rate=1e-4,monthly_max_epochs=20)
    return c,cases,events,signature,imputation


def file_hashes(folder):
    return {name:digest(folder/name) for name in ['actual.csv','plans.csv.gz','summary.json']}


def verify_case(record, initial_energy, steps):
    folder=Path(record['path'])
    if file_hashes(folder)!=record['hashes']:raise ValueError(f'Saved case changed: {folder}')
    s=json.loads((folder/'summary.json').read_text())
    if s['status']!='completed' or s['completed_steps']!=steps or not np.isclose(s['initial_soc_kwh'],initial_energy,atol=1e-7,rtol=0):
        raise ValueError(f'Case completion/SOC mismatch: {folder}')
    return s


def score_month(args,fold,checkpoint_path):
    settings,identity,config,base=annual.inputs(args.data,args.output)
    views,counts=annual.fold_views(base,fold)
    for arm,path in [('updated',checkpoint_path),('fixed_initial',args.output/'2023-03/training/best.pt')]:
        destination=args.output/fold['month']/arm
        stamp=destination/'annual_evaluation_completed.json'
        tag=dict(checkpoint_sha256=digest(path),fold=fold,dataset_identity=identity)
        if stamp.exists():
            saved=json.loads(stamp.read_text())
            if saved['identity']!=tag or saved['files']!={n:digest(destination/n) for n in saved['files']}:
                raise ValueError('Saved evaluation identity/artifacts changed')
            continue
        if destination.exists():
            # Keep partial data; new inference writes to a clean directory.
            destination.rename(destination.with_name(arm+'_partial_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        destination.mkdir()
        model=annual._build_controller(config,base,settings,annual.PVGRUController)
        model.load_state_dict(annual.load_checkpoint(path)['controller_state_dict'])
        annual.evaluate_m0(model,views['test'],destination,settings,
            dict(annual_fold=fold,arm=arm,sample_counts=counts,checkpoint=str(path),
                 checkpoint_sha256=digest(path),note='Original-valid samples only; imputation not scored or used as training labels'))
        save(stamp,dict(identity=tag,files={p.name:digest(p) for p in destination.iterdir() if p.is_file()}))


def run(args):
    c,cases,events,signature,imputation=preflight(args)
    report=dict(status='preflight_passed_not_run',trajectory_count=47,days=366,
        steps_per_trajectory=366*96,total_environment_steps=47*366*96,
        initial_checkpoint_sha256=signature['checkpoint_sha256'],imputation=imputation,
        resolved_cases=[asdict(x) for x in cases],event_selection=events,
        state_policy='SOC continuous per trajectory; 11 monthly model updates; no scenario reset at month boundary')
    if args.check_only:
        save(ROOT/'deliverables/M0全年连续仿真启动检查_20261006.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('imputation','resolved_cases','event_selection')},ensure_ascii=False,indent=2))
        return
    annual.torch.set_num_threads(1)
    year=args.output/'continuous_year'
    if year.exists():
        if not args.resume:raise FileExistsError('Annual run exists; use --resume after inspecting progress')
        manifest=json.loads((year/'manifest.json').read_text())
        if manifest['identity']!=signature:raise ValueError('Annual code/config/data changed since start')
    else:
        year.mkdir(parents=True)
        save(year/'manifest.json',dict(identity=signature,all_cases=[asdict(x) for x in cases],
            allow_grid_charging=True,controller_version=scenarios.ScenarioMPC.version,
            period_scope='47 continuous annual trajectories, monthly model switching',imputation=imputation))
        save(year/'event_selection.json',events)
    protocol=annual.prepare_protocol(args.data,args.output)
    protocol.update(execution='One user launch; 47 continuous annual trajectories; automatic monthly training/review/testing',
        state_policy='Carry each trajectory SOC across month boundaries; shared PV-history-driven model updates',
        closed_loop_missing_policy='Verified causal imputation overlay; original learning/scoring validity unchanged',
        annual_scenario_config_sha256=digest(args.config),simulation_manifest_sha256=digest(args.simulation_inputs/'manifest.json'))
    save(year/'protocol.json',protocol)
    stop=threading.Event()
    previous_signal=signal.signal(signal.SIGINT,lambda *_:stop.set())
    model_index=[]
    energies={case.name:case.initial_soc*common.HARDWARE.battery_capacity_kWh for case in cases}
    try:
        for fold in annual.month_folds():
            if stop.is_set():raise InterruptedError('Stopped before next month')
            month=fold['month']; folder=args.output/month
            training=folder/'training'; cp=training/'best.pt'
            metadata=json.loads((training/'run_metadata.json').read_text()) if (training/'run_metadata.json').exists() else {}
            if metadata.get('status')!='completed':
                # Training has epoch checkpoints, but does not consume the
                # simulation stop event. Let Ctrl-C interrupt it normally.
                signal.signal(signal.SIGINT,signal.default_int_handler)
                try:
                    annual.train_month(args.data,args.output,protocol,month,resume=args.resume and training.exists())
                finally:
                    signal.signal(signal.SIGINT,lambda *_:stop.set())
            review=annual.review_month(args.output,month)
            key=dict(month=month,checkpoint_sha256=review['checkpoint_sha256'],annual_signature=signature)
            statepath=folder/'continuous_month_state.json'
            state=json.loads(statepath.read_text()) if statepath.exists() else dict(identity=key,cases={})
            if state['identity']!=key:raise ValueError('Monthly checkpoint/protocol changed')
            model_index.append(dict(month=month,checkpoint=str(cp),sha256=digest(cp),best_epoch=review['best_epoch']))
            start=pd.Timestamp(fold['test_origin_start']); end=pd.Timestamp(fold['test_origin_end_exclusive'])
            steps=int((end-start)/pd.Timedelta(minutes=15)); offset=int((start-annual.TEST_START)/pd.Timedelta(minutes=15))
            remaining=[case for case in cases if case.name not in state['cases']]
            if remaining:
                model,dataset,_=common.load_model(cp,evaluation_start=start,evaluation_steps=steps)
                frame,history,repairs=common.prepare_inputs(dataset,start,steps,'error',
                    args.simulation_inputs/'dispatch.csv',args.simulation_inputs/'load_tiers.csv',args.simulation_inputs)
                tiers=read(args.simulation_inputs/'load_tiers.csv').reindex(frame.index)
                for i in (1,2,3):frame[f'cl{i}']=tiers[f'minimum_operation_CL{i}_kW']
            print(f'YEAR / {month}: carry 47 SOC states; model epoch {review["best_epoch"]}',flush=True)
            for number,case in enumerate(cases,1):
                if case.name in state['cases']:
                    s=verify_case(state['cases'][case.name],energies[case.name],steps)
                else:
                    if stop.is_set():raise InterruptedError('Stopped before next trajectory')
                    attempt=folder/'continuous_cases'/case.name/('attempt_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                    attempt.parent.mkdir(parents=True,exist_ok=True)
                    local=replace(case,event_hour=case.event_hour-offset*.25)
                    options=argparse.Namespace(days=steps/96,smoke_steps=None,history_steps=96,no_live=args.no_live)
                    s=scenarios.run_case(options,local,model,dataset,frame,history,repairs,attempt,stop,number,47,
                        initial_soc_kwh=energies[case.name])
                    if s['status']!='completed':raise RuntimeError(s.get('error') or 'Month/case interrupted; resume from its month-start SOC')
                    state['cases'][case.name]=dict(path=str(attempt),hashes=file_hashes(attempt))
                    save(statepath,state)
                energies[case.name]=s['final_soc_kwh']
            if stop.is_set():raise InterruptedError('Stopped before forecast evaluation')
            score_month(args,fold,cp)
            save(folder/'coupled_test_completed.json',dict(status='completed',month=month,
                checkpoint_sha256=digest(cp),config_sha256=digest(args.config),scope='continuous_annual_47',
                month_state_sha256=digest(statepath),final_soc_kwh=energies.copy()))
            save(year/'annual_status.json',dict(status='running',last_completed_month=month,models=model_index,final_soc_kwh=energies.copy()))
        consolidate(args,year,cases,model_index)
        annual.pool_results(args.output,[f['month'] for f in annual.month_folds()])
        save(year/'annual_status.json',dict(status='completed',models=model_index,final_soc_kwh=energies,trajectory_count=47))
    except BaseException as exc:
        save(year/'annual_status.json',dict(status='interrupted' if stop.is_set() or isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'failed',
            error=str(exc),models=model_index,completed_months=[f['month'] for f in annual.month_folds()
                if (args.output/f['month']/'coupled_test_completed.json').exists()]))
        raise
    finally:
        signal.signal(signal.SIGINT,previous_signal)


def consolidate(args,year,cases,model_index):
    """Annual ledgers from completed monthly runs; no repeated model execution."""
    summaries=[]; index=[]
    model_map=pd.DataFrame(model_index)
    model_map.to_csv(year/'monthly_models.csv',index=False)
    manifest=json.loads((year/'manifest.json').read_text())
    manifest['monthly_models']=model_index
    manifest['monthly_models_sha256']=digest(year/'monthly_models.csv')
    save(year/'manifest.json',manifest)
    total_keys=['grid_import_kwh','grid_export_kwh','grid_cost','component_cost_sum','battery_charge_kwh',
        'battery_discharge_kwh','pv_curtailed_kwh','loss_load_kwh','cl1_ens_kwh','cl2_ens_kwh','cl3_ens_kwh']
    for case in cases:
        blocks=[]; months=[]; dest=year/case.name;dest.mkdir(exist_ok=True)
        previous_energy=case.initial_soc*common.HARDWARE.battery_capacity_kWh
        for fold in annual.month_folds():
            state=json.loads((args.output/fold['month']/'continuous_month_state.json').read_text())
            record=state['cases'][case.name]; source=Path(record['path'])
            a=pd.read_csv(source/'actual.csv',parse_dates=['timestamp'])
            s=verify_case(record,previous_energy,len(a))
            if not np.isclose(a.soc_before_kwh.iloc[0],previous_energy,atol=1e-7,rtol=0):raise ValueError('SOC discontinuity')
            previous_energy=float(a.soc_after_kwh.iloc[-1]);months.append(s)
            offset=int((pd.Timestamp(fold['test_origin_start'])-annual.TEST_START)/pd.Timedelta(minutes=15))
            a['decision_step']+=offset;a['model_month']=fold['month'];blocks.append(a)
            index.append(dict(case=case.name,month=fold['month'],path=str(source),**record['hashes']))
        a=pd.concat(blocks,ignore_index=True)
        if not pd.DatetimeIndex(a.timestamp).equals(pd.date_range(annual.TEST_START,annual.TEST_END,freq='15min',inclusive='left')):
            raise ValueError('Annual time gap')
        a.to_csv(dest/'actual.csv',index=False)
        o=a.loc[a.grid_connected.eq(False)]
        s=dict(status='completed',case=case.name,group=case.group,completed_steps=len(a),requested_steps=len(a),
            allow_grid_charging=True,elapsed_seconds=sum(m['elapsed_seconds'] for m in months),
            **{k:float(a[k].sum()) for k in total_keys},initial_soc_kwh=float(a.soc_before_kwh.iloc[0]),
            final_soc_kwh=float(a.soc_after_kwh.iloc[-1]),outage_start_soc_kwh=float(o.soc_before_kwh.iloc[0]) if len(o) else None,
            max_reserve_gap_kwh=float(a.reserve_gap_kwh.max()),max_reserve_hardware_gap_kwh=float(a.reserve_hardware_gap_kwh.max()),
            warning_reserve_gap_steps=sum(m['warning_reserve_gap_steps'] for m in months),fallback_steps=sum(m['fallback_steps'] for m in months),
            max_constraint_violation=max(m['max_constraint_violation'] for m in months),error=None,result_path=str(dest))
        for tier in ('cl1','cl2','cl3'):
            ens=float(o[tier+'_ens_kwh'].sum());load=float(o[tier+'_load_kwh'].sum())
            lost=o[tier+'_ens_kwh'].gt(1e-5).to_numpy();streak=longest=0
            for loss in lost:streak=streak+1 if loss else 0;longest=max(longest,streak)
            first=float((o.loc[lost,'timestamp'].iloc[0]-(annual.TEST_START+pd.Timedelta(hours=case.event_hour)))/pd.Timedelta(hours=1)) if lost.any() else None
            s.update({f'outage_{tier}_ens_kwh':ens,f'outage_{tier}_load_kwh':load,
                f'outage_{tier}_supply_fraction':1-ens/load if load else None,
                f'outage_{tier}_first_loss_hours':first,f'outage_{tier}_longest_loss_hours':longest*.25})
        save(dest/'summary.json',s);summaries.append(s)
    # Shared PV forecasts are indexed once; complete per-case MPC plans stay
    # in their original monthly files rather than being duplicated 47 times.
    with gzip.open(year/'S0_grid/plans.csv.gz','wt') as out:
        first=True
        for fold in annual.month_folds():
            source=Path(next(x['path'] for x in index if x['case']=='S0_grid' and x['month']==fold['month']))
            offset=int((pd.Timestamp(fold['test_origin_start'])-annual.TEST_START)/pd.Timedelta(minutes=15))
            for block in pd.read_csv(source/'plans.csv.gz',usecols=['decision_step','decision_timestamp','horizon_timestamp','pv_forecast_kw'],chunksize=96000):
                block['decision_step']+=offset;block.to_csv(out,index=False,header=first);first=False
    pd.DataFrame(index).to_csv(year/'full_plans_index.csv',index=False)
    pd.DataFrame(summaries).to_csv(year/'summary.csv',index=False)
    save(year/'suite_status.json',dict(full_matrix_completed=True,status='completed'))
    export_report_metrics(year)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--simulation-inputs',type=Path,default=IMPUTED)
    p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--output',type=Path,default=RESULTS)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-live',action='store_true')
    return p


if __name__=='__main__':run(parser().parse_args())
