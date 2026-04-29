import jax.numpy as jnp
from track_mjx.agent.dmpo.action_utils import bind, unbind


def test_bind_clips_to_open_unit_interval():
    raw = jnp.array([-100.0, 0.0, 100.0])
    bound = bind(raw)
    assert jnp.all(bound > -1.0)
    assert jnp.all(bound < 1.0)
    assert jnp.isclose(bound[1], 0.0)


def test_unbind_inverts_bind():
    raw = jnp.linspace(-3.0, 3.0, 11)
    round_trip = unbind(bind(raw))
    assert jnp.allclose(round_trip, raw, atol=1e-5)


def test_unbind_handles_boundary():
    boundary = jnp.array([-1.0, 1.0])
    result = unbind(boundary)
    assert jnp.all(jnp.isfinite(result))


def test_bind_unbind_jittable():
    import jax
    f = jax.jit(lambda x: unbind(bind(x)))
    x = jnp.array([0.5, -0.5, 0.0])
    y = f(x)
    assert jnp.allclose(y, x, atol=1e-5)


def test_bind_unbind_vmappable():
    import jax
    raw = jnp.zeros((4, 3))
    out = jax.vmap(lambda r: unbind(bind(r)))(raw)
    assert out.shape == (4, 3)
