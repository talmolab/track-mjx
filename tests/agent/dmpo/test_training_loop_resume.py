"""Resuming a run must continue the env-step counter, not restart it at zero.

Checkpoints are written at ``step=total_env_steps``
(``ckpt_save_callback(state, total_env_steps)``), so a checkpoint directory named
``DMPONetwork_95232000`` means 95.232M env steps. Before ``start_env_steps``
existed, ``training_loop.run`` hard-coded ``total_env_steps = 0``, so a resumed
run:

  * trained ``num_timesteps`` MORE steps instead of stopping at that total,
  * re-saved checkpoints under step numbers it had already used, and
  * reported a wandb x-axis that jumped backwards mid-run.

Two things must hold, or the fix is not safe to land:
  1. ``start_env_steps=0`` (the default) reproduces the previous behaviour
     exactly, so no completed experiment becomes incomparable.
  2. ``start_env_steps=S`` runs ``num_timesteps - S`` further steps and reports
     absolute env-step counts, not session-relative ones.
"""

import jax

from track_mjx.agent.dmpo.training_loop import run as train_loop_run
from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step

from tests.agent.dmpo.test_train_dmpo_fused import _setup


def _run(start_env_steps, num_timesteps):
    """Run the loop and return every env-step value it reported to wandb."""
    s = _setup()
    cfg = s["cfg"]
    K, iters = 1, 1
    rng = jax.random.PRNGKey(0)

    # Pre-fill replay so the warm-up branch is skipped and every iteration is a
    # full chunk of exactly env_steps_per_chunk steps.
    fused = make_fused_train_step(s["env"], s["nets"], s["optimizers"], s["rb"], cfg, K=K)
    state, env_state, rb_state, _ = fused(s["state"], None, s["rb_state"], rng)

    seen = []
    train_loop_run(
        env=s["env"], nets=s["nets"], optimizers=s["optimizers"], rb=s["rb"],
        cfg=cfg, K=K, iters_per_chunk=iters, rng=rng,
        state=state, env_state=env_state, rb_state=rb_state,
        warmup_done=True,
        ckpt_mgr=None, eval_callback=None,
        # eval_every_steps=1 makes the callback fire on every chunk, so `seen`
        # is the full trace of the counter rather than a subsample.
        wandb_log_callback=lambda payload, step: seen.append(int(step)),
        eval_every_steps=1,
        log_every_steps=1,
        num_timesteps=num_timesteps,
        start_env_steps=start_env_steps,
    )
    return seen, cfg.num_envs * cfg.unroll_length * iters


def test_default_is_unchanged():
    """start_env_steps defaults to 0 -> identical to the pre-fix behaviour."""
    seen, per_chunk = _run(start_env_steps=0, num_timesteps=3 * 32)
    assert seen[0] == per_chunk, f"first report should be one chunk in, got {seen[0]}"
    assert seen == [per_chunk * (i + 1) for i in range(len(seen))]
    assert seen[-1] >= 3 * 32


def test_resume_continues_the_counter():
    """A resumed run reports ABSOLUTE env steps starting from the checkpoint."""
    start = 10 * 32
    seen, per_chunk = _run(start_env_steps=start, num_timesteps=start + 3 * 32)
    assert seen[0] == start + per_chunk, (
        f"resumed run must report absolute steps: expected {start + per_chunk}, "
        f"got {seen[0]} (counter restarted at zero?)"
    )
    assert all(b > a for a, b in zip(seen, seen[1:])), "counter must be monotone"


def test_resume_does_not_retrain_the_whole_budget():
    """The budget is a TOTAL, not an increment.

    This is the bug that motivated the fix: resuming a 95.2M checkpoint with
    num_timesteps=300M must run ~204.8M more steps, not another 300M.
    """
    per_chunk = 32
    total = 20 * per_chunk
    start = 15 * per_chunk

    fresh, _ = _run(start_env_steps=0, num_timesteps=total)
    resumed, _ = _run(start_env_steps=start, num_timesteps=total)

    assert len(fresh) == 20, f"fresh run should take 20 chunks, took {len(fresh)}"
    assert len(resumed) == 5, (
        f"resumed run should take only the remaining 5 chunks, took {len(resumed)} "
        "-- the counter restarted and it retrained the full budget"
    )
    assert fresh[-1] == resumed[-1] == total, "both must stop at the same TOTAL"
