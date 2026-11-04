"""Legacy preset reproduces the pre-alignment learner and loss bit for bit.

``golden/pre_alignment.npz`` was recorded by ``golden/make_pre_alignment_goldens.py``
on the code as it stood BEFORE any acme-alignment switch changed behaviour. Here
the same four computations are rerun under ``LEGACY_ALIGNMENT`` and compared
with ``np.array_equal`` (CPU, same seeds, same jitted functions). This is the
regression guard for every later change behind the alignment switches: if one
of them leaks into the legacy path, an array here stops matching.

The comparison is bitwise, so it is only meaningful on CPU; it is skipped when
the default backend is not CPU.
"""

import importlib.util
import pathlib

import jax
import numpy as np
import pytest

from track_mjx.agent.dmpo.config import LEGACY_ALIGNMENT

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"
GOLDEN_PATH = GOLDEN_DIR / "pre_alignment.npz"
MAKER_PATH = GOLDEN_DIR / "make_pre_alignment_goldens.py"

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="bitwise goldens are CPU-only"
)


def _load_maker():
    spec = importlib.util.spec_from_file_location(
        "make_pre_alignment_goldens", MAKER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def maker():
    return _load_maker()


@pytest.fixture(scope="module")
def golden():
    with np.load(GOLDEN_PATH) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.parametrize("group", ["ff", "rnn", "tick", "mpo"])
def test_legacy_preset_matches_pre_alignment_golden(group, maker, golden):
    got = maker.compute_group(group, dict(LEGACY_ALIGNMENT))
    want = {k: v for k, v in golden.items() if k.startswith(f"{group}/")}
    assert want, f"no golden arrays for group {group!r}"
    assert set(got) == set(want), (
        f"array names differ for {group!r}: "
        f"missing={sorted(set(want) - set(got))[:5]} "
        f"extra={sorted(set(got) - set(want))[:5]}"
    )
    mismatched = [
        k
        for k in sorted(want)
        if got[k].dtype != want[k].dtype
        or got[k].shape != want[k].shape
        or not np.array_equal(got[k], want[k])
    ]
    assert not mismatched, f"{len(mismatched)} arrays differ, first: {mismatched[:5]}"


def test_goldens_cover_every_group(golden, maker):
    for group in maker.GROUPS:
        assert any(k.startswith(f"{group}/") for k in golden), group
    assert GOLDEN_PATH.stat().st_size <= 300 * 1024
