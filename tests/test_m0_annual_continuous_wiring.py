"""Software wiring checks only: training, prediction, MPC and execution mocked."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_m0_annual_simulation as runner
from m0_scenarios import load_cases
from prepare_annual_simulation_inputs import donors


def test_year_chains_11_updates_and_carries_all_47_soc_states(tmp_path):
    args=runner.parser().parse_args(['--output',str(tmp_path)])
    config=json.loads(runner.CONFIG.read_text())
    cases=load_cases(config)
    calls=[];updates=[];previous={c.name:c.initial_soc*215 for c in cases}
    def completed_training(month):
        folder=tmp_path/month/'training';folder.mkdir(parents=True)
        (folder/'run_metadata.json').write_text('{"status":"completed"}')
        (folder/'best.pt').write_text('fake checkpoint '+month)
    completed_training('2023-03')
    def train(data,output,protocol,month,resume=False):
        prior=(pd.Timestamp(month+'-01')-pd.offsets.MonthBegin()).strftime('%Y-%m')
        assert (tmp_path/prior/'coupled_test_completed.json').exists()
        updates.append(month);completed_training(month)
    def review(output,month):
        return dict(checkpoint_sha256=runner.digest(tmp_path/month/'training/best.pt'),best_epoch=14)
    def case_run(options,case,model,dataset,frame,history,repairs,folder,stop,number,total,initial_soc_kwh):
        assert initial_soc_kwh==previous[case.name]
        previous[case.name]-=1
        calls.append((case.name,initial_soc_kwh))
        folder.mkdir()
        s=dict(status='completed',completed_steps=int(options.days*96),initial_soc_kwh=initial_soc_kwh,
               final_soc_kwh=previous[case.name])
        (folder/'summary.json').write_text(json.dumps(s))
        (folder/'actual.csv').write_text('stub,not simulation\n')
        (folder/'plans.csv.gz').write_bytes(b'stub,not predictions')
        return s
    frame=pd.DataFrame(index=pd.date_range('2023-03-01',periods=1,freq='15min'))
    tiers=pd.DataFrame({f'minimum_operation_CL{i}_kW':[1.] for i in (1,2,3)},index=frame.index)
    with patch.object(runner,'preflight',return_value=(config,cases,[],{'checkpoint_sha256':'stub-no-experiment'},{})), \
         patch.object(runner.annual,'prepare_protocol',return_value={}), \
         patch.object(runner.annual,'train_month',side_effect=train), \
         patch.object(runner.annual,'review_month',side_effect=review), \
         patch.object(runner.common,'load_model',return_value=(None,None,None)) as load_model, \
         patch.object(runner.common,'prepare_inputs',return_value=(frame,None,None)), \
         patch.object(runner,'read',return_value=tiers), \
         patch.object(runner.scenarios,'run_case',side_effect=case_run), \
         patch.object(runner,'score_month'),patch.object(runner,'consolidate'), \
         patch.object(runner.annual,'pool_results'),redirect_stdout(io.StringIO()):
        runner.run(args)
        assert len(calls)==47*12 and len(updates)==11 and load_model.call_count==12
        assert updates==[f['month'] for f in runner.annual.month_folds()[1:]]
        state=json.loads((tmp_path/'continuous_year/annual_status.json').read_text())
        for c in cases:assert state['final_soc_kwh'][c.name]==c.initial_soc*215-12
        args.resume=True
        runner.run(args)
        assert len(calls)==47*12 and len(updates)==11 and load_model.call_count==12


def test_month_offset_preserves_outage_warning_and_future_grid_information():
    cases=load_cases(json.loads(runner.CONFIG.read_text()))
    # Include a synthetic outage crossing a month boundary, independent of
    # where the real low-PV selection happens to fall this year.
    cross=replace(cases[2],event_hour=(pd.Timestamp('2023-03-31 18:00')-runner.annual.TEST_START)/pd.Timedelta(hours=1),
                  outage_hours=48,announced_hours=48,warning_hours=24)
    for start in pd.date_range('2023-03-01','2024-02-01',freq='MS'):
        offset=(start-runner.annual.TEST_START)/pd.Timedelta(hours=1)
        for original in cases+[cross]:
            local=replace(original,event_hour=original.event_hour-offset)
            hours=np.arange(0,24*start.days_in_month,.25)
            np.testing.assert_array_equal(local.actual_grid(hours),original.actual_grid(hours+offset))
            for h in (0,.25,12,24,24*start.days_in_month-.25):
                a=local.information(h);b=original.information(h+offset)
                np.testing.assert_array_equal(a[0],b[0]);assert a[1:]==b[1:]


def test_imputation_donors_are_original_valid_and_strictly_past():
    axis=pd.date_range('2023-03-01','2023-04-02',freq='15min')
    f=pd.DataFrame({'value':np.arange(len(axis))},index=axis)
    valid=pd.Series(True,index=axis);t=pd.Timestamp('2023-04-01 12:00')
    valid.loc[t-pd.Timedelta(days=1)]=False
    found=donors(f,valid,t)
    assert all(s<t and s.hour==12 and s.minute==0 and valid.loc[s] for s in found)
    assert t-pd.Timedelta(days=1) not in found
