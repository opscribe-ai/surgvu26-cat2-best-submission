"""Root conftest so `surgvu` is importable without an installed package.

The Condor pytest job transfers `src`, `scripts`, `tests`, `config`, and this
file into job scratch — not `pyproject.toml` — so pytest's `pythonpath` ini
option never takes effect there. conftest.py is always collected by pytest
regardless of how it was invoked, so it is the one place this path insertion
is guaranteed to run in both the login-node and container environments.
"""
import sys
from pathlib import Path

_SRC = str(Path(__file__).parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
