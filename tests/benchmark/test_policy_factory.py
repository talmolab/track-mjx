import jax
from omegaconf import OmegaConf
from track_mjx.config.utils import prepare_config
from track_mjx.analysis import rollout
from track_mjx.analysis.benchmark import policy_factory, components

CONFIG = "track_mjx/config/rodent-full-clips.yaml"


def _cfg_and_state(num_envs=2):
    cfg, _, _ = prepare_config(OmegaConf.load(CONFIG))
    env, state = components.build_env_and_state(cfg, num_envs, seed=0)
    return cfg, env, state


def test_inference_fn_produces_action_of_correct_shape():
    cfg, env, state = _cfg_and_state(num_envs=2)
    base = components.unwrap(env)
    fn = policy_factory.build_inference_fn(cfg, base, state, seed=0)
    action, extras = fn(state.obs, jax.random.PRNGKey(0))
    assert action.shape == (2, base.action_size)
    assert jax.numpy.isfinite(action).all()
