"""Stage relocation must preserve physical settings and actual data identity."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import train_m1 as training

@pytest.mark.parametrize('mutation', [None, 'capacity', 'filename', 'data_hash', 'code_hash'])
def test_stage_relocation_is_narrow(monkeypatch, tmp_path, mutation):
    old_root, new_root = tmp_path / 'old', tmp_path / 'new'
    old = dict(scales={}, data={'sources': {'dispatch': {'path': str(old_root/'data.csv'), 'sha256':'same'}}},
               initial_checkpoint_sha256='initial', runtime={},
               training={'hardware': {'battery_capacity_kWh':215., 'data_path':str(old_root/'frozen/data.csv'),
                                      'load_tiers_path':str(old_root/'frozen/tiers.csv')},
                         'prediction_weight':1., 'economic_weight':1., 'selection_metric':'loss'},
               code_sha256={'tools/train_m1.py':'old', 'controller.py':'same'})
    identity = copy.deepcopy(old)
    identity['data']['sources']['dispatch']['path']=str(new_root/'data.csv')
    identity['training']['hardware']['data_path']=str(new_root/'frozen/data.csv')
    identity['training']['hardware']['load_tiers_path']=str(new_root/'frozen/tiers.csv')
    identity['training'].update(economic_weight=2., selection_metric='economic_CNY_per_step')
    identity['code_sha256']['tools/train_m1.py']='new'
    if mutation=='capacity': identity['training']['hardware']['battery_capacity_kWh']=216.
    if mutation=='filename': identity['training']['hardware']['data_path']=str(new_root/'other.csv')
    if mutation=='data_hash': identity['data']['sources']['dispatch']['sha256']='changed'
    if mutation=='code_hash': identity['code_sha256']['controller.py']='changed'
    saved=dict(training_method='M1', state={'order':[], 'next_episode':0}, training_identity=old)
    monkeypatch.setattr(training,'ROOT',new_root)
    monkeypatch.setattr(training,'load_checkpoint',lambda _:saved)
    checkpoint=old_root/'experiments/M1/run/last.pt'
    if mutation:
        with pytest.raises(ValueError):training.stage_checkpoint(checkpoint,identity)
    else:
        assert training.stage_checkpoint(checkpoint,identity) is saved
