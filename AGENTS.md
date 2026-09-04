# AGENTS.md

This file provides guidance to AI agents (Claude Code and others) working with
code in this repository.

## Project Overview

**sparse-ir** is the Python wrapper for the sparse intermediate representation
(IR) of many-body Green's functions. It exposes a pure-Python API
(`src/sparse_ir/`: `basis.py`, `sampling.py`, `dlr.py`, `sve.py`, `kernel.py`,
`poly.py`, `augment.py`, `basis_set.py`) built on top of the `libsparseir` C
library, called through `ctypes` via the `pylibsparseir` package.

## Shared Rules

Read the SpM-lab shared agent rules before making changes:
<https://github.com/SpM-lab/spm-agent-rules> — start at `rules/index.md` and
load only the rule files the current task needs.

For this repository, that is normally:

- `rules/common.md` — cross-language repository policy
- `rules/ffi-boundary.md` — the C boundary: dtype, contiguity, status codes
- `rules/numerical-conventions.md` — the physics contracts every wrapper must
  state
- `rules/testing.md` — what the test suite must actually cover
- `rules/python.md` — `sparse-ir`- and `ctypes`-specific guidance

If network access is unavailable, look for a sibling checkout at
`../spm-agent-rules`.

## Repository-Specific Rules

See [`REPOSITORY_RULES.md`](REPOSITORY_RULES.md) for this repository's
verified layout, native-library loading, dev setup, and test/CI commands.

## Precedence

Repository-local rules in `REPOSITORY_RULES.md` override the shared rules
when they are more specific. Say so in the pull request when the override
affects a review.
