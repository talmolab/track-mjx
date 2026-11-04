"""Loaders shared by the entry-script tests.

The DMPO entry scripts live in ``scripts/``, which is not a package, so the
tests execute them with ``runpy.run_path`` under a name other than
``"__main__"``: the module body runs (imports, helper definitions, the hydra
decorator) and ``main()`` does not.

Each entry also imports vnl-playground modules that the pinned vnl-playground
release predates. A missing module skips the test visibly instead of failing
collection; any other import error still fails.
"""

import os
import pathlib
import runpy

import pytest

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts"

# entry id -> (script file name, vnl-playground modules it imports at load time
# that the pinned vnl-playground release does not ship).
ENTRIES = {
    "kl_anchor": (
        "train_dmpo_highlvl_kl_anchor.py",
        (
            "vnl_playground.naccdmax_patch",
            "vnl_playground.tasks.wrappers_kl_anchor",
            "vnl_playground.tasks.wrappers_truncation",
        ),
    ),
    "highlvl": (
        "train_dmpo_highlvl.py",
        (
            "vnl_playground.naccdmax_patch",
            "vnl_playground.tasks.wrappers_truncation",
        ),
    ),
    "imitation": (
        "train_dmpo.py",
        ("vnl_playground.tasks.wrappers_truncation",),
    ),
}

# Variables the entries set with os.environ.setdefault at import time. They
# are restored after each test so that loading an entry does not change the
# environment the rest of the suite runs in.
_ENTRY_ENV_VARS = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "XLA_PYTHON_CLIENT_PREALLOCATE")


def load_entry(entry: str, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Execute an entry script without running ``main()``.

    Args:
        entry: A key of ``ENTRIES``.
        monkeypatch: Used to restore the environment variables the entry sets.

    Returns:
        The script's module globals.
    """
    script, required_modules = ENTRIES[entry]
    for module in required_modules:
        pytest.importorskip(
            module,
            reason=f"{script} needs {module}, which the pinned vnl-playground lacks",
        )
    for var in _ENTRY_ENV_VARS:
        # Record the current state (set or unset) so teardown restores it.
        if var in os.environ:
            monkeypatch.setenv(var, os.environ[var])
        else:
            monkeypatch.delenv(var, raising=False)
    return runpy.run_path(str(SCRIPTS_DIR / script), run_name="entry_under_test")
