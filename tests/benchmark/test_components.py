import jax
from omegaconf import OmegaConf
from track_mjx.config.utils import prepare_config
from track_mjx.analysis.benchmark import components, policy_factory

CONFIG = "track_mjx/config/rodent-full-clips.yaml"


def test_build_timed_callables_run():
    cfg, _, _ = prepare_config(OmegaConf.load(CONFIG))
    env, state = components.build_env_and_state(cfg, num_envs=2, seed=0)
    fn = policy_factory.build_inference_fn(cfg, components.unwrap(env), state, seed=0)
    callables = components.build_timed_callables(cfg, env, state, fn)
    assert set(callables) == {"policy", "mujoco", "rl_env", "full_step"}
    for name, (call, args) in callables.items():
        out = call(*args)
        jax.block_until_ready(out)  # must not raise


def test_control_step_advances_state():
    cfg, _, _ = prepare_config(OmegaConf.load(CONFIG))
    env, state = components.build_env_and_state(cfg, num_envs=2, seed=0)
    fn = policy_factory.build_inference_fn(cfg, components.unwrap(env), state, seed=0)
    step = components.build_control_step(env, fn, jax.random.PRNGKey(0))
    next_state = jax.jit(step)(state)
    jax.block_until_ready(next_state)
    assert next_state.data.qpos.shape[0] == 2
