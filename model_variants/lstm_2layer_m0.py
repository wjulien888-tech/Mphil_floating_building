"""双层 LSTM M0：可配置历史PV窗口 → 直接多步PV预测。

两层64维循环网络、0.2层间dropout；旧默认64→5→1输出头保持兼容。
未来24小时配置使用96/192/288步历史、64→64→96输出头。
沿用本项目15分钟/kW数据、验证集选模、best/last及独立测试协议。
默认读取2021—2023年PV-only冻结数据；本脚本不生成标签，也不使用气象输入。
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

OL_ROOT = Path(__file__).resolve().parents[1] / "external/ol-ems-hkust"
if str(OL_ROOT) not in sys.path:
    sys.path.insert(0, str(OL_ROOT))

import torch
from torch import nn
from src.models.controller import Controller as OriginalController


class PVLSTMController(OriginalController):
    """双层循环网络，预测头一次性输出全部未来时间步。"""

    recurrent_type = "LSTM"

    def __init__(
        self,
        optimisation_layer: nn.Module,
        dataset: Any = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        input_dim: int | None = None,
        pv_min: float | None = None,
        pv_max: float | None = None,
        dropout: float = 0.2,
        fc_hidden_size: int = 5,
    ) -> None:
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if fc_hidden_size < 1:
            raise ValueError("fc_hidden_size must be positive")
        super().__init__(optimisation_layer=optimisation_layer, dataset=dataset,
                         hidden_dim=hidden_dim, num_layers=num_layers,
                         input_dim=input_dim, pv_min=pv_min, pv_max=pv_max)
        # 移除原单层模型的预测模块，不保留闲置的可训练参数。
        del self._lstm
        del self._pv_head
        self._lstm = nn.LSTM(
            input_size=self._input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, self._prediction_horizon)

    def forecast_pv(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self._input_dim:
            raise ValueError(
                f"history must have shape [batch, lookback, {self._input_dim}]; "
                f"got {tuple(history.shape)}"
            )
        if history.shape[1] == 0:
            raise ValueError("history lookback dimension must not be empty")
        encoded, _ = self._lstm(history.to(dtype=self.fc1.weight.dtype))
        normalised_pv = torch.sigmoid(self.fc2(self.relu(self.fc1(encoded[:, -1, :]))))
        return (self._pv_min + self._pv_range * normalised_pv).unsqueeze(-1)


if __name__ == "__main__":
    from _m0_variant_runner import build_parser, main
    main(build_parser("LSTM").parse_args(), PVLSTMController)
