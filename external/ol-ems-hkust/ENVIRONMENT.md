# Python environment

The HKUST integration uses one Python 3.11 virtual environment shared by the two
vendored repositories. The existing virtual environment at the parent-project
root is independent and must not be reused or replaced.

```bash
brew install python@3.11
cd external/ol-ems-hkust
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade \
  "pip==23.3.2" "setuptools==59.5.0" "wheel==0.42.0"
.venv/bin/python -m pip install -r requirements-macos-arm64.txt
.venv/bin/python -m pip install \
  --no-deps --no-build-isolation -e ../python-microgrid-hkust
```

Do not install the upstream `requirements.txt` on macOS. It contains Linux CUDA
and Triton packages from the original development machine. Lightning 1.5.10
requires setuptools 59.5.0; this is intentional. `--no-build-isolation` makes the
editable install use that already-installed build tool instead of downloading a
temporary copy.

Verify the environment with:

```bash
.venv/bin/python -m pip check
.venv/bin/python -c \
  "import torch, pytorch_lightning, cvxpy, cvxpylayers, pymgrid"
.venv/bin/python main.py --help
```

The environment was verified on Apple Silicon with Python 3.11.16. The editable
`pymgrid` import must resolve to `../python-microgrid-hkust/src/pymgrid`, so edits
to that vendored source tree are immediately visible without reinstalling it.
