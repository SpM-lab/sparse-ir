# Repository Rules

These are `sparse-ir`-specific rules. Apply them on top of the shared
SpM-lab rules in [`spm-agent-rules`](https://github.com/SpM-lab/spm-agent-rules).
Everything below is verified against this repository's actual files and CI
configuration as of this writing; treat any drift you notice between this
document and the code as a bug to fix in the same PR that touches the
affected area.

## Project Overview

`sparse-ir` is a pure-Python wrapper around the `libsparseir` C library. It
does not vendor or build the C library itself; the native binding layer is
supplied by the separate `pylibsparseir` package (a `pyproject.toml`
dependency, currently pinned as `pylibsparseir>=0.8.3,<0.10.0`).

## Package Layout

```text
src/sparse_ir/
  __init__.py    public API surface and __all__
  abstract.py    AbstractBasis and other ABCs
  basis.py       FiniteTempBasis, finite_temp_bases
  basis_set.py   FiniteTempBasisSet
  sampling.py    TauSampling, MatsubaraSampling
  sve.py         SVEResult, compute, compute_sve
  dlr.py         DiscreteLehmannRepresentation
  kernel.py      LogisticKernel, RegularizedBoseKernel
  augment.py     AugmentedBasis and augmentation helpers
  poly.py        polynomial interpolation utilities
  _gauss.py      Gauss quadrature helpers
  _util.py       internal utilities
```

`tests/` holds one `test_<module>.py` per area above, plus
`test_advanced_features.py`, `test_sve_advanced.py`, and
`test_sampling_advanced.py` for cross-cutting and edge-case coverage.
`tests/conftest.py` defines session-scoped fixtures (`sve_logistic`,
`sve_reg_bose`, `test_bases`, `rng`) shared across the suite.

## Native Library Loading (the FFI boundary)

This repository never calls `ctypes.CDLL` directly. Every module that talks
to the C API imports the already-loaded handle and typed wrappers from
`pylibsparseir`, for example (`src/sparse_ir/kernel.py`):

```python
import ctypes
from ctypes import c_int, c_double, byref

from pylibsparseir.core import _lib
from pylibsparseir.core import logistic_kernel_new, reg_bose_kernel_new
from pylibsparseir.constants import COMPUTATION_SUCCESS
```

The same pattern (`from pylibsparseir.core import _lib`, plus
`pylibsparseir.constants` for status codes) appears in `augment.py`,
`basis.py`, `dlr.py`, `poly.py`, `sampling.py`, and `sve.py`. Consequences
for changes in this repository:

- Locating, downloading, or building the native `libsparseir` shared object
  is `pylibsparseir`'s responsibility, not this repository's. Do not add a
  local `dlopen`/`CDLL` search path here.
- Status-code handling must use `pylibsparseir.constants` (e.g.
  `COMPUTATION_SUCCESS`), not ad hoc integer literals.
- Bumping the supported `libsparseir`/`pylibsparseir` version means updating
  the dependency range in **both** `pyproject.toml` and `.conda/meta.yaml`
  (see Version Consistency below) — they are checked for consistency, not
  merged automatically.

## Version Consistency Check

`check_libsparseir_version_consistency.py` (repository root) verifies that
the `pylibsparseir` version range in `pyproject.toml` matches the one in
`.conda/meta.yaml`. Run it after touching either file:

```bash
python check_libsparseir_version_consistency.py
```

It is also wired up as a local pre-commit hook (`.pre-commit-config.yaml`,
id `version-consistency-check`) and as a standalone `version-check` job in
CI (`.github/workflows/CI.yml`).

## Installing Dev Dependencies

The project uses `uv`. From a clean clone:

```bash
uv sync              # installs the package plus the default dependency groups
uv sync --group dev  # dev group: pytest, sphinx, sphinx-rtd-theme, matplotlib, ipykernel, jupytext
```

`pyproject.toml` defines two dependency groups: `dev` (testing plus doc
tooling) and `doc` (documentation build only: sphinx, sphinx-rtd-theme,
matplotlib).

## Running Tests

Tests are pytest-based; `pyproject.toml` configures `testpaths = ["tests"]`.

```bash
uv run pytest                       # full suite
uv run pytest -v                    # verbose
uv run pytest tests/test_basis.py   # a single file
uv run pytest --cov=sparse_ir       # with coverage (pytest-cov must be installed separately; not a default dev dependency)
```

## CI Entry Points

- `.github/workflows/CI.yml` — on push/PR to `mainline`: a `version-check`
  job (`python check_libsparseir_version_consistency.py`) and a `test`
  matrix over `{ubuntu-latest, macos-latest} x {3.10, 3.11, 3.12, 3.13}`
  that runs `uv sync` then `uv run pytest`.
- `.github/workflows/wheel.yml` — on `v*` tag push or manual dispatch: builds
  the sdist/wheel with `uv build` and uploads to PyPI via
  `pypa/gh-action-pypi-publish`.
- `.github/workflows/conda.yml` — on `v*` tag push or manual dispatch: builds
  and uploads the conda package from `.conda/` to the `spm-lab` Anaconda
  channel.

The default branch is `mainline`, not `main`; CI triggers are scoped to it.

## Documentation Build

Docs live under `doc/` (Sphinx, `doc/conf.py`) and are built via the `doc`
dependency group:

```bash
uv sync --group doc
```

`.readthedocs.yaml` drives the hosted Read the Docs build.

## Precedence

Where this file and the shared `spm-agent-rules` content disagree, the more
specific rule here applies to this repository; note the override in the
pull request description when it affects review.
