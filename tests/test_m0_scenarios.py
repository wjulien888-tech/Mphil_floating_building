import json
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_m0_simulation as common
from m0_scenarios import Scenario,ScenarioMPC,load_cases,reserve_profile,execute,TIERS,H,DT


def future(grid=0,load=1):
    f={k:np.ones(H)*load for k in TIERS}
    f.update(grid=np.ones(H)*grid,import_price=np.ones(H)*.3,export_price=np.ones(H)*.2)
    return f


def test_matrix_has_all_groups_and_47_unique_cases():
    cases=load_cases(json.loads((ROOT/'configs/m0_s0_s5_v2.json').read_text()))
    assert len(cases)==47 and len({c.name for c in cases})==47
    assert {c.group for c in cases}=={'S0','S1','S2','S3','S4','S5'}
    assert len([c for c in cases if c.group=='S5'])==32


def test_suite_identity_records_grid_charging_and_survives_manifest_reload(tmp_path):
    from argparse import Namespace
    from run_m0_scenarios import fingerprint
    data=tmp_path/'input';data.write_bytes(b'fixture')
    args=Namespace(checkpoint=data,dispatch=data,load_tiers=data,invalid_pv='error',smoke_steps=1)
    identity=fingerprint(args,{'protocol_version':'m0-scenarios-v2'})
    assert identity['hardware']['allow_grid_charging'] is True
    assert json.loads(json.dumps(identity))==identity


def test_false_and_true_warning_are_indistinguishable_until_event():
    false=Scenario('false','S1',event_hour=48,announced_hours=24,warning_hours=24)
    true=Scenario('true','S2',event_hour=48,outage_hours=24,announced_hours=24,warning_hours=24)
    for hour in [0,23.75,24,36,47.75]:
        a=false.information(hour);b=true.information(hour)
        np.testing.assert_array_equal(a[0],b[0]);assert a[1:]==b[1:]
    assert np.all(true.information(0)[0]==1) # no leaked future outage
    assert np.all(false.information(48)[0]==1)
    assert true.information(48)[0][0]==0
    assert true.information(72)[0][0]==1


def test_reserve_long_event_uses_prediction_not_future_actual():
    case=Scenario('long','S3',event_hour=24,outage_hours=168,announced_hours=168,warning_hours=24)
    low=reserve_profile(case,0,np.zeros(H),np.ones(H)*2)
    high=reserve_profile(case,0,np.ones(H)*2,np.ones(H)*2)
    assert low[0]>common.HARDWARE.max_soc_kWh
    assert high[0]==common.HARDWARE.min_soc_kWh
    assert low[0]==pytest.approx(common.HARDWARE.min_soc_kWh+168*2/common.HARDWARE.discharge_efficiency)


def test_empty_battery_blackout_records_all_demand_as_unserved():
    mpc=ScenarioMPC();f=future()
    arrays,err,_,_=mpc.solve(np.zeros(H),f,common.HARDWARE.min_soc_kWh,np.ones(H+1)*common.HARDWARE.min_soc_kWh)
    assert err<1e-4
    assert arrays['charge'].max()<1e-5
    assert arrays['grid_import'].max()<1e-5
    for k in TIERS:np.testing.assert_allclose(arrays['unmet_'+k],1.,atol=1e-4)
    adapter=common._build_adapter(common.HARDWARE);adapter.reset(soc_kwh=common.HARDWARE.min_soc_kWh)
    row=execute(adapter,arrays,dict(pv=0,load=3,critical_load=2,import_price=.3,export_price=.2,grid_available=0),dict(cl1=1,cl2=1,cl3=1))
    common.verify_execution(row)
    assert row['loss_load_kwh']==pytest.approx(.75,abs=1e-5)
    assert row['cl1_ens_kwh']==pytest.approx(.25,abs=1e-5)
    assert row['grid_import_kwh']==0


def test_island_priority_and_exact_prediction_execution():
    # Enough PV for CL1, some CL2, no CL3. Empty battery cannot borrow energy.
    mpc=ScenarioMPC();f=future()
    arrays,err,_,_=mpc.solve(np.ones(H)*1.5,f,common.HARDWARE.min_soc_kWh,np.ones(H+1)*common.HARDWARE.min_soc_kWh)
    assert arrays['unmet_cl1'].sum()*DT<1e-4
    assert arrays['unmet_cl2'].sum()*DT==pytest.approx(12,abs=1e-4)
    assert arrays['unmet_cl3'].sum()*DT==pytest.approx(24,abs=1e-4)
    adapter=common._build_adapter(common.HARDWARE);adapter.reset(soc_kwh=common.HARDWARE.min_soc_kWh)
    row=execute(adapter,arrays,dict(pv=1.5,load=3,critical_load=2,import_price=.3,export_price=.2,grid_available=0),dict(cl1=1,cl2=1,cl3=1))
    common.verify_execution(row)
    assert row['soc_after_kwh']==pytest.approx(arrays['soc'][1],abs=1e-4)
    for k in TIERS:assert row[k+'_ens_kwh']==pytest.approx(arrays['unmet_'+k][0]*DT,abs=1e-4)


def test_scheduled_shedding_does_not_get_undone_to_feed_load_before_charging():
    # Intentional CL3 shedding releases PV to charge for future priority loads.
    arrays=dict(charge=np.array([1.]),discharge=np.array([0.]),grid_import=np.array([0.]),grid_export=np.array([0.]),
                unmet_cl1=np.array([0.]),unmet_cl2=np.array([0.]),unmet_cl3=np.array([2.]))
    adapter=common._build_adapter(common.HARDWARE)
    row=execute(adapter,arrays,dict(pv=3,load=4,critical_load=2,import_price=.3,export_price=.2,grid_available=0),dict(cl1=1,cl2=1,cl3=2))
    common.verify_execution(row)
    assert row['battery_charge_kwh']==pytest.approx(.25)
    assert row['load_served_kwh']==pytest.approx(.5)
    assert row['cl3_ens_kwh']==pytest.approx(.5)


def test_reserve_can_charge_from_grid_without_pv_within_physical_limits():
    mpc=ScenarioMPC();f=future(grid=1)
    arrays,err,_,_=mpc.solve(np.zeros(H),f,common.HARDWARE.min_soc_kWh,np.ones(H+1)*500)
    assert err<1e-4
    assert arrays['charge'][0]>1
    assert arrays['charge'].max()<=common.HARDWARE.battery_power_kW+1e-5
    assert arrays['grid_import'].max()<=common.HARDWARE.grid_import_limit_kW+1e-5
    assert arrays['soc'].max()<=common.HARDWARE.max_soc_kWh+1e-5
    assert arrays['soc'][-1]==pytest.approx(common.HARDWARE.max_soc_kWh,abs=1e-4)
    assert arrays['reserve_shortfall'][0]>0
    assert arrays['reserve_shortfall'][-1]<1e-4
    for k in TIERS:assert arrays['unmet_'+k].max()<1e-5
    adapter=common._build_adapter(common.HARDWARE)
    adapter.reset(soc_kwh=common.HARDWARE.min_soc_kWh)
    row=execute(adapter,arrays,dict(pv=0,load=3,critical_load=2,import_price=.3,export_price=.2,grid_available=1),dict(cl1=1,cl2=1,cl3=1))
    common.verify_execution(row)
    assert row['soc_after_kwh']==pytest.approx(arrays['soc'][1],abs=1e-4)
    assert row['grid_import_kw']==pytest.approx(arrays['grid_import'][0],abs=1e-4)


@pytest.mark.parametrize('later_price,should_charge',[(1.,True),(.31,False)])
def test_night_grid_charging_accounts_for_price_and_roundtrip_losses(later_price,should_charge):
    # Buying at .30 to avoid .31 is unprofitable after the 0.92 round trip.
    # Buying at .30 to avoid 1.00 should charge, despite zero PV throughout.
    mpc=ScenarioMPC();f=future(grid=1)
    f['import_price'][:]=later_price;f['import_price'][0]=.3
    f['export_price'][:]=0
    arrays,err,_,_=mpc.solve(np.zeros(H),f,common.HARDWARE.min_soc_kWh,
                           np.full(H+1,common.HARDWARE.min_soc_kWh))
    assert err<1e-4
    if should_charge:
        assert arrays['charge'][0]>1
        assert arrays['discharge'][1:].sum()>1
    else:
        assert arrays['charge'].max()<1e-4


def test_grid_charge_before_announced_outage_serves_load_and_stops_importing():
    case=Scenario('night','S2',event_hour=1,outage_hours=6,announced_hours=6,warning_hours=1)
    f=future(grid=1,load=2);f['grid']=case.information(0)[0]
    reserve=reserve_profile(case,0,np.zeros(H),sum(f[k] for k in TIERS))
    arrays,err,_,_=ScenarioMPC().solve(np.zeros(H),f,common.HARDWARE.min_soc_kWh,reserve)
    assert err<1e-4
    assert arrays['charge'][:4].sum()>1
    outage=f['grid']==0
    np.testing.assert_allclose(arrays['grid_import'][outage],0,atol=1e-5)
    np.testing.assert_allclose(arrays['charge'][outage],0,atol=1e-5)
    for k in TIERS:assert arrays['unmet_'+k].sum()*DT<1e-4
    # Execute the entire warning and outage against the same PV/load inputs.
    adapter=common._build_adapter(common.HARDWARE)
    adapter.reset(soc_kwh=common.HARDWARE.min_soc_kWh)
    for j in range(28):
        step={k:values[j:j+1] for k,values in arrays.items()}
        row=execute(adapter,step,dict(pv=0,load=6,critical_load=4,import_price=.3,export_price=.2,
                                    grid_available=float(f['grid'][j])),dict(cl1=2,cl2=2,cl3=2))
        common.verify_execution(row)
        assert row['soc_after_kwh']==pytest.approx(arrays['soc'][j+1],abs=1e-4)
        if outage[j]:assert row['grid_import_kwh']==0


def test_low_pv_does_not_rewrite_past_or_reveal_future_to_model():
    import pandas as pd
    import torch
    from types import SimpleNamespace
    from m0_scenarios import scenario_inputs
    ds=SimpleNamespace(lookback_window=96,pv_min=0.,pv_max=100.)
    frame=pd.DataFrame({'pv':np.ones(200)*20,'grid_available':np.ones(200)})
    histories=torch.ones(296,1)*.2
    case=Scenario('low','S4',pv_regime='observed_low',selection_hours=24)
    altered,history=scenario_inputs(frame,histories,ds,case)
    torch.testing.assert_close(history,histories,rtol=0,atol=0)
    np.testing.assert_array_equal(altered.pv,frame.pv)


def test_low_pv_lexicographic_regression_at_previous_failure(monkeypatch):
    # This fixture guards the HiGHS regression, independent of local solver defaults.
    monkeypatch.setenv("MPC_MILP_SOLVER", "highs")
    saved=json.loads((ROOT/'tests/fixtures/m0_lowpv_lexicographic_regression.json').read_text())
    mpc=ScenarioMPC()
    arrays,violation,solver,stages=mpc.solve(np.array(saved['forecast']),
        {k:np.array(v) for k,v in saved['future'].items()},saved['initial_soc'],np.array(saved['reserve']))
    assert violation<1e-4 and stages==5
    assert 'HiGHS' in solver
    # The final economic solution must retain all three prior tier optima.
    for j,k in enumerate(TIERS):
        assert arrays['unmet_'+k].sum()*DT<=mpc.caps.value[j]+1e-5


def test_event_selection_includes_repaired_minimum_and_preserves_source():
    import pandas as pd
    from m0_scenarios import resolve_low_events
    index=pd.date_range('2023-11-01',periods=10*96,freq='15min')
    f=pd.DataFrame(dict(pv=np.ones(len(index))*10,load=np.ones(len(index))),index=index)
    # The lowest-energy block remains eligible even with a repaired PV point.
    f.loc['2023-11-03 12:00':'2023-11-03 17:45','pv']=0
    f.loc['2023-11-05 12:00':'2023-11-05 17:45','pv']=1
    original=f.copy(deep=True)
    c=Scenario('low','S4',outage_hours=6,announced_hours=6,warning_hours=24,
               pv_regime='observed_low',selection_hours=6)
    repairs=pd.DataFrame(index=pd.to_datetime(['2023-11-01 13:00','2023-11-03 13:00']))
    resolved,records=resolve_low_events([c],f,10,repairs)
    assert resolved[0].event_hour==60
    assert records[0]['pv_kwh']==0
    assert records[0]['repaired_pv_points']==1
    assert records[0]['candidate_count']==8
    pd.testing.assert_frame_equal(f,original)


def test_old_scaled_protocol_rejected():
    with pytest.raises(ValueError,match='Withdrawn'):
        load_cases({'protocol_version':'m0-scenarios-v1'})
    with pytest.raises(TypeError):
        Scenario('bad','S4',pv_factor=.25)
