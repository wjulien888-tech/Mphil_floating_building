# Source update — 2026-11-04

This snapshot brings in annual PV preparation and simulation tools, multi-horizon forecasters, M0 scenario evaluation, differentiable MPC/execution, M1/M2 training utilities, weather-event scenario tools, configurations and regression tests. The forecasting baselines in `external/forecast_baselines` include upstream provenance and licenses.

These files are research implementations. A checked-in configuration is not proof of a completed experiment. Battery sizing and critical-load duration require closed-loop verification; lower cost alone does not establish reliability. Compare forecasting error, operating cost, unmet energy and solver diagnostics under matching conditions. Future actual PV is valid only for an explicitly labelled perfect-forecast baseline.

Raw datasets, checkpoints, generated experiment runs, reports, slide workspaces and local environments are retained locally. Configurations and launchers may contain workstation/HPC-specific paths; supply the corresponding local inputs before running. The existing WORKLOG records historical work, not a fresh validation of all results.

## Validation environment

Use Python 3.11 and the dependencies in `external/ol-ems-hkust/requirements.txt`. Make the vendored microgrid package available with an editable install or PYTHONPATH:

```sh
python -m pip install --no-build-isolation --no-deps -e external/python-microgrid-hkust
python -m pytest tests -q
# Alternatively, without installing the vendored package:
PYTHONPATH=external/python-microgrid-hkust/src python -m pytest tests -q
```

Solver-dependent tests additionally require their configured solvers and, where applicable, a valid local license. No new formal training run is part of this source update.

## Source-update checks

All selected Python sources parsed successfully; no credential patterns were found in the selected text files. The first full test run passed 262 tests and exposed one solver-selection assumption in the HiGHS regression test. That test now explicitly selects HiGHS; the final validation result is recorded below after rerunning.

Final full-suite rerun: **263 passed** (14 dependency warnings), using the command above. The HiGHS regression test also passed independently. Upstream baseline whitespace is preserved; project-owned staged files pass `git diff --cached --check`.
