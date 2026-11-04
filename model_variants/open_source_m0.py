"""Official PatchTST/DLinear adapters for the frozen PV-only M0 protocol."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external"))

from forecast_baselines.patchtst import Model as PatchTST
from forecast_baselines.dlinear import Model as DLinear
from _m0_variant_runner import build_parser, main


PATCH_CONFIG = dict(e_layers=2, n_heads=4, d_model=64, d_ff=128,
                    dropout=0.2, fc_dropout=0.0, head_dropout=0.0,
                    individual=False, patch_len=16, stride=8,
                    padding_patch="end", revin=True, affine=True,
                    subtract_last=False, decomposition=False, kernel_size=25)


class PVOpenSource(nn.Module):
    def __init__(self, optimisation_layer, dataset, **unused):
        super().__init__()
        self.register_buffer("pv_min", torch.tensor(dataset.pv_min))
        self.register_buffer("pv_range", torch.tensor(dataset.pv_max - dataset.pv_min))
        config = dict(seq_len=dataset.lookback_window,
                      pred_len=dataset.prediction_horizon, enc_in=dataset.input_dim,
                      individual=False)
        if self.recurrent_type == "PatchTST":
            config.update(PATCH_CONFIG)
            self.backbone = PatchTST(SimpleNamespace(**config))
        else:
            self.backbone = DLinear(SimpleNamespace(**config))

    def forecast_pv(self, history):
        prediction = self.pv_min + self.pv_range * self.backbone(history)
        # Keep native linear outputs and gradients for training. Enforce physical
        # nonnegativity for evaluation; no future observations are used.
        return prediction if self.training else prediction.clamp_min(0)


class PVPatchTST(PVOpenSource):
    recurrent_type = "PatchTST"


class PVDLinear(PVOpenSource):
    recurrent_type = "DLinear"


def run():
    import argparse
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--model_type", choices=("patchtst", "dlinear"), required=True)
    selected, remaining = selector.parse_known_args()
    cls = PVPatchTST if selected.model_type == "patchtst" else PVDLinear
    args = build_parser(cls.recurrent_type).parse_args(remaining)
    args.forecast_architecture = dict(
        architecture_id=f"official_{selected.model_type}_pv_v1",
        model_type=selected.model_type,
        backbone_config=PATCH_CONFIG if selected.model_type == "patchtst" else
            dict(individual=False, kernel_size=25),
        output_activation="linear_train_nonnegative_eval",
        upstream_commit="204c21efe0b39603ad6e2ca640ef5896646ab1a9" if
            selected.model_type == "patchtst" else "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6")
    main(args, cls)


if __name__ == "__main__":
    run()
