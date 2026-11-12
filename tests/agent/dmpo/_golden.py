"""Comparison against the pre-alignment goldens, exact where it can be.

The goldens are CPU float32 results of a few learner steps. They reproduce
bit for bit on the CPU model and jax/jaxlib versions that generated them,
which are recorded in ``golden/pre_alignment.platform.json``. On any other CPU,
XLA's code generation for that host can change the order of float
reductions, and the values then agree only to rounding that grows over the
steps. There the comparison falls back to a tolerance, and
``test_legacy_equivalence`` checks that the tolerance still separates the
legacy preset from the aligned defaults on that platform.
"""

import json
import os
import pathlib

import jax
import jaxlib
import numpy as np

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"
GOLDEN_PATH = GOLDEN_DIR / "pre_alignment.npz"
PLATFORM_PATH = GOLDEN_DIR / "pre_alignment.platform.json"

# Cross-platform tolerance. On an emulated older instruction set the largest
# difference was 2.2e-4 absolute (a KL statistic after two steps); the aligned
# defaults move dozens of arrays by far more.
RTOL = 5e-2
ATOL = 1e-3


def _cpu_model() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def on_golden_platform() -> bool:
    """Whether this host matches the one that generated the goldens.

    ``DMPO_GOLDEN_EXACT=0`` forces the tolerance path, to exercise it here.
    """
    if os.environ.get("DMPO_GOLDEN_EXACT") == "0":
        return False
    recorded = json.loads(PLATFORM_PATH.read_text())
    return (
        jax.default_backend() == "cpu"
        and _cpu_model() == recorded["cpu_model"]
        and jax.__version__ == recorded["jax"]
        and jaxlib.__version__ == recorded["jaxlib"]
    )


def mismatched(got: dict, want: dict, exact: bool | None = None) -> list[str]:
    """Keys of ``want`` that ``got`` does not reproduce.

    Args:
        got: Recomputed arrays.
        want: Golden arrays, same keys.
        exact: Compare bit for bit. Defaults to ``on_golden_platform()``.

    Returns:
        Sorted keys whose dtype or shape differs, or whose values differ:
        bitwise when exact, else beyond ``RTOL``/``ATOL`` (integer and
        boolean arrays are always compared exactly).
    """
    if exact is None:
        exact = on_golden_platform()
    bad = []
    for k in sorted(want):
        g, w = np.asarray(got[k]), np.asarray(want[k])
        if g.dtype != w.dtype or g.shape != w.shape:
            bad.append(k)
        elif exact or w.dtype.kind not in "fc":
            if not np.array_equal(g, w):
                bad.append(k)
        elif not np.allclose(g, w, rtol=RTOL, atol=ATOL, equal_nan=True):
            bad.append(k)
    return bad
