#!/bin/bash
set -euo pipefail
cd /data/user/jwei484/projects/m1_original_20261013
PY=/data/user/jwei484/envs/m1-original-py311/bin/python
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$PY" -m pip install --progress-bar off pip==23.3.2 setuptools==59.5.0 wheel==0.42.0
"$PY" -m pip install --progress-bar off -r external/ol-ems-hkust/requirements.txt
"$PY" -m pip install --progress-bar off --no-build-isolation --no-deps -e external/python-microgrid-hkust
"$PY" -m pip check
"$PY" -m pip freeze > installed_packages.txt
