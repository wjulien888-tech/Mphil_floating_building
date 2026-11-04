"""Verify the adapter preserves official outputs, units and differentiability."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_variants"))
from open_source_m0 import PVPatchTST, PVDLinear
import open_source_m0


@pytest.mark.parametrize("cls", [PVPatchTST, PVDLinear])
def test_adapter_units_gradients_and_roundtrip(cls):
    torch.set_num_threads(1)
    dataset = SimpleNamespace(pv_min=0., pv_max=80., lookback_window=96,
                              prediction_horizon=96, input_dim=1)
    model = cls(None, dataset)
    history = torch.rand(4, 96, 1, requires_grad=True)
    model.eval()
    with torch.no_grad():
        expected = (80 * model.backbone(history)).clamp_min(0)
        actual = model.forecast_pv(history)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (4, 96, 1)
    assert (actual >= 0).all()
    model.train()
    model.forecast_pv(history).square().mean().backward()
    assert torch.isfinite(history.grad).all() and history.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    clone = cls(None, dataset)
    clone.load_state_dict(model.state_dict())
    model.eval()
    clone.eval()
    torch.testing.assert_close(model.forecast_pv(history), clone.forecast_pv(history))


def test_cli_keeps_mode_distinct_from_model_type(monkeypatch):
    captured = {}
    monkeypatch.setattr(sys, "argv", ["open_source_m0.py", "--model_type", "dlinear",
                                     "--mode", "train", "--prediction_horizon", "96"])
    monkeypatch.setattr(open_source_m0, "main", lambda args, cls: captured.update(args= args, cls=cls))
    open_source_m0.run()
    assert captured["cls"] is PVDLinear
    assert captured["args"].mode == "train"
    assert captured["args"].forecast_architecture["model_type"] == "dlinear"
