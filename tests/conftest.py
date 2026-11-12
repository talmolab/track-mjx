"""Repo-wide pytest defaults.

This file is imported by pytest before any test module or nested conftest
(pytest walks conftest.py files parent-first), so setting environment
defaults here — as long as it happens before anything imports jax — is
enough to make them the effective default for the whole tree, including
``tests/agent/dmpo/conftest.py`` and every test module under it.
"""

import os

# Project practice is to run the test suite on CPU: a stray GPU-targeted
# test can OOM-kill a live training job sharing the same card. GPU runs opt
# in explicitly by exporting JAX_PLATFORMS (or CUDA_VISIBLE_DEVICES) before
# invoking pytest; `setdefault` leaves that override alone.
#
# Two inherited DMPO tests are known to fail on GPU only (cause not
# established) and are deliberately left as-is rather than "fixed" to pass
# under a backend they were never written for:
#   tests/agent/dmpo/test_learner_rnn.py::test_unroll_resets_hidden_after_done_and_matches_hand_rollout
#   tests/agent/dmpo/test_temperature_floor.py::test_flat_q_gives_a_constant_positive_temperature_gradient
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# NOTE on MUJOCO_GL: deliberately left unset here. mujoco reads MUJOCO_GL at
# `import mujoco` time (mujoco/rendering/classic/gl_context.py:24, imported
# from mujoco/__init__.py:76), but that whole import is wrapped in a bare
# `try/except ImportError: pass` (mujoco/__init__.py:75-78), and every
# failure mode below that point (missing libEGL, missing GLFW, etc.) is
# itself converted to ImportError before it gets there — see
# OpenGL/platform/egl.py's `EGL` lazy property, which explicitly does
# `except OSError as err: raise ImportError(...)` when the EGL shared
# library can't be loaded. So `import mujoco` cannot crash on a GPU-less,
# EGL-less runner regardless of MUJOCO_GL's value; only *creating* a
# GLContext/Renderer would fail. The ~7 test files that need a real (or
# best-effort) render backend already do their own
# `os.environ.setdefault("MUJOCO_GL", "egl")` at module level. CI sets
# MUJOCO_GL=disable for the test job (see .github/workflows/ci.yml): with
# MUJOCO_GL=egl, importing the vnl-playground rodent tasks also loads
# dm_control's EGL renderer, which does raise when libEGL is missing.
