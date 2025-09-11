# Fortran API documentation

## How to update `_cbinding.inc`

We assume we are using M-series macOS.
Install `llvm` via homebrew:

```sh
% brew install llvm
```

Then, run the following commands

```zsh
% export DYLD_LIBRARY_PATH=/opt/homebrew/opt/llvm/lib:$DYLD_LIBRARY_PATH
% uv run generate_c_binding.py ../include/sparseir/sparseir.h
```

