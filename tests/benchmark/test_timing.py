import jax
import jax.numpy as jnp
from track_mjx.analysis.benchmark import timing


def test_time_op_returns_summary_keys():
    x = jnp.ones((256, 256))
    fn = jax.jit(lambda a: a @ a)
    out = timing.time_op(fn, x, n_warmup=2, n_reps=10)
    for k in ("median_ms", "mean_ms", "std_ms", "min_ms", "p25_ms", "p75_ms", "n_reps"):
        assert k in out
    assert out["median_ms"] > 0
    assert out["n_reps"] == 10


def test_scan_throughput_positive():
    init = jnp.zeros((128,))
    step_fn = lambda c: c + 1.0  # carry -> carry
    out = timing.scan_throughput(step_fn, init, n_steps=50, n_warmup=1, n_reps=3)
    assert out["per_step_ms"] > 0
    assert out["n_steps"] == 50


def test_collect_metadata_has_core_keys():
    md = timing.collect_metadata()
    for k in ("gpu", "cpu", "jax", "jaxlib", "mujoco", "brax", "vnl_playground", "jax_devices"):
        assert k in md
