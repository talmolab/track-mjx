"""Verify that the extracted training_loop produces the same state as
calling train_chunk + warmup + log/eval bookkeeping inline (the path
currently in train_dmpo.py)."""

import jax
import jax.numpy as jnp

# Reuse the toy env + setup helper from the fused-step test fixture.
from tests.agent.dmpo.test_train_dmpo_fused import _setup
from track_mjx.agent.dmpo.train_dmpo_chunk import make_train_chunk
from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step
from track_mjx.agent.dmpo.training_loop import run as train_loop_run


def test_training_loop_one_chunk_matches_inline():
    """One chunk through training_loop.run == one direct train_chunk call,
    given the same RNG, same starting state, and warmup_done already True
    (we pre-fill the replay)."""
    s = _setup()
    cfg = s["cfg"]
    K = 2
    rng = jax.random.key(0)

    # --- Path A: inline (what train_dmpo.py used to do) ---
    fused = make_fused_train_step(
        s["env"], s["nets"], s["optimizers"], s["rb"], cfg, K=K
    )
    chunk = make_train_chunk(fused, n_iters=2)
    state_a = s["state"]
    rb_state_a = s["rb_state"]
    # Pre-fill with one fused step so warmup is over.
    state_a, env_state_a, rb_state_a, _ = fused(state_a, None, rb_state_a, rng)
    rng_a, k_chunk_a = jax.random.split(rng)
    state_a, env_state_a, rb_state_a, metrics_a = chunk(
        state_a,
        env_state_a,
        rb_state_a,
        k_chunk_a,
    )

    # --- Path B: training_loop.run with num_timesteps that triggers exactly
    # one chunk after warmup. ---
    s2 = _setup()
    cfg2 = s2["cfg"]
    # Pre-fill replay the same way Path A did.
    fused2 = make_fused_train_step(
        s2["env"], s2["nets"], s2["optimizers"], s2["rb"], cfg2, K=K
    )
    state_b, env_state_b, rb_state_b, _ = fused2(s2["state"], None, s2["rb_state"], rng)
    state_b, env_state_b, rb_state_b, metrics_b = train_loop_run(
        env=s2["env"],
        nets=s2["nets"],
        optimizers=s2["optimizers"],
        rb=s2["rb"],
        cfg=cfg2,
        K=K,
        iters_per_chunk=2,
        rng=rng,  # run() does the same split(rng) as Path A
        # Path A: rng_a, k_chunk_a = jax.random.split(rng); chunk(..., k_chunk_a)
        # Path B run() does: rng, k_step = jax.random.split(rng); chunk(..., k_step)
        # so passing the original rng (not rng_a) lines them up bit-equal.
        state=state_b,
        env_state=env_state_b,
        rb_state=rb_state_b,
        warmup_done=True,  # pre-warmed
        max_chunks=1,  # exactly one chunk
        ckpt_mgr=None,  # no checkpoints in test
        eval_callback=None,  # no eval in test
        wandb_log_callback=None,  # no wandb in test
    )

    # Assert state pytree leaves are bit-equal (or close). RNG split layout
    # is shared between fused_step.test and training_loop, so the result
    # should match exactly.
    leaves_a = jax.tree.leaves(state_a)
    leaves_b = jax.tree.leaves(state_b)
    assert len(leaves_a) == len(leaves_b)
    for la, lb in zip(leaves_a, leaves_b):
        assert jnp.allclose(la, lb, rtol=1e-5, atol=1e-6), (
            f"state leaf mismatch: {la} vs {lb}"
        )


def _run_with_ckpt_callback(
    num_timesteps, eval_every_steps, start_env_steps=0, with_callback=True
):
    """Run the loop for 32-env-step chunks; return every checkpoint save step."""
    s = _setup()
    cfg = s["cfg"]
    K, iters = 1, 1
    rng = jax.random.key(0)

    # Pre-fill replay so every loop iteration is one full chunk of
    # num_envs * unroll_length = 32 env steps.
    fused = make_fused_train_step(
        s["env"], s["nets"], s["optimizers"], s["rb"], cfg, K=K
    )
    state, env_state, rb_state, _ = fused(s["state"], None, s["rb_state"], rng)

    saved = []
    train_loop_run(
        env=s["env"],
        nets=s["nets"],
        optimizers=s["optimizers"],
        rb=s["rb"],
        cfg=cfg,
        K=K,
        iters_per_chunk=iters,
        rng=rng,
        state=state,
        env_state=env_state,
        rb_state=rb_state,
        warmup_done=True,
        ckpt_save_callback=(
            (lambda st, step: saved.append(int(step))) if with_callback else None
        ),
        eval_every_steps=eval_every_steps,
        num_timesteps=num_timesteps,
        start_env_steps=start_env_steps,
    )
    return saved


def test_final_checkpoint_written_after_loop():
    """A budget that is not a multiple of the eval cadence still ends saved.

    Evals (and so periodic saves) fire at 64; the loop then runs one more chunk
    to 96 without reaching the next cadence boundary. The last
    ``ckpt_save_callback`` must be at the final env-step count, not at 64.
    """
    saved = _run_with_ckpt_callback(num_timesteps=3 * 32, eval_every_steps=64)
    assert saved == [64, 96], f"expected a final save at loop exit, got {saved}"


def test_final_checkpoint_not_double_saved_when_cadence_aligned():
    """When the last periodic save already sits at the final step, don't repeat it."""
    saved = _run_with_ckpt_callback(num_timesteps=4 * 32, eval_every_steps=64)
    assert saved == [64, 128], f"final step saved twice or missed: {saved}"


def test_final_checkpoint_not_rewritten_when_resumed_at_budget():
    """Resuming a run already at its budget must not re-save the restored step."""
    saved = _run_with_ckpt_callback(
        num_timesteps=3 * 32, eval_every_steps=64, start_env_steps=3 * 32
    )
    assert saved == [], f"re-saved an already-existing checkpoint step: {saved}"


def test_final_checkpoint_skipped_without_save_callback():
    """No checkpointing configured -> the loop exit saves nothing and still runs."""
    saved = _run_with_ckpt_callback(
        num_timesteps=3 * 32, eval_every_steps=64, with_callback=False
    )
    assert saved == []
