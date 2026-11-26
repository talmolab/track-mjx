"""Regression test pinning ``make_checkpointer``'s keep-every-checkpoint default.

Project rule: training checkpoints are never pruned. ``make_checkpointer`` in
``track_mjx.agent.dmpo.checkpoint`` defaults ``max_to_keep=None`` precisely
for this reason -- a diverged run must not be able to silently overwrite the
last healthy checkpoint before anyone notices. No existing test pinned that
default before this file. Without it, a future edit that "tidies" the
signature back to a small finite integer (the old, dangerous default, under which a
diverged run came within two saves of overwriting its best checkpoint)
would pass silently.

This test saves more checkpoints than any plausible old finite default (6)
through a DEFAULT ``make_checkpointer(tmp_path)`` call -- no ``max_to_keep``
argument -- and asserts every one of them is still on disk afterward.
"""

import numpy as np
import orbax.checkpoint as ocp

from track_mjx.agent.dmpo.checkpoint import make_checkpointer

NUM_CHECKPOINTS = 6  # bigger than any plausible old finite max_to_keep (e.g. 3)


def test_default_max_to_keep_retains_every_checkpoint(tmp_path):
    # No max_to_keep argument: this is the exact call site pattern used by the
    # DMPO training entry point, and it must keep everything.
    mgr = make_checkpointer(str(tmp_path))

    steps = list(range(NUM_CHECKPOINTS))
    for step in steps:
        tiny_pytree = {"w": np.array([float(step)], dtype=np.float32)}
        mgr.save(step, args=ocp.args.StandardSave(tiny_pytree))
    mgr.wait_until_finished()

    assert sorted(mgr.all_steps()) == steps, (
        f"all_steps() dropped checkpoints: expected {steps}, got "
        f"{sorted(mgr.all_steps())}. If max_to_keep was reintroduced as a "
        "finite default, this is exactly the regression this test exists "
        "to catch -- see the checkpoint-retention guardrail."
    )

    for step in steps:
        step_dir = tmp_path / f"DMPONetwork_{step}"
        assert step_dir.exists() and step_dir.is_dir(), (
            f"expected checkpoint directory {step_dir} to still exist; "
            "a checkpoint was pruned even though max_to_keep was not set."
        )
