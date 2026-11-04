# Vendored forecasting baselines

PatchTST: https://github.com/yuqinie98/PatchTST
Commit: 204c21efe0b39603ad6e2ca640ef5896646ab1a9
Files: PatchTST_supervised/models/PatchTST.py and layers/{PatchTST_backbone,PatchTST_layers,RevIN}.py.
License: Apache-2.0 (LICENSE_PatchTST).

DLinear: https://github.com/cure-lab/LTSF-Linear
Commit: 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6
File: models/DLinear.py. License: Apache-2.0 (LICENSE_DLinear).

Only internal import paths were changed to avoid the generic `layers` namespace.
PV adapters use training-split min/max scaling, native linear model outputs,
and a nonnegative clamp during evaluation only. No pretrained weights used.
