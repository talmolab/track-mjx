# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brax training acting functions."""

import time
from typing import Callable, Sequence, Tuple, Union

from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
from brax.v1 import envs as envs_v1
import jax
import numpy as np

State = Union[envs.State, envs_v1.State]
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
import jax.numpy as jnp
from flax import nnx
from track_mjx.agent.nnx_ppo_network import PPOImitationNetworks


def actor_step(
    env: Env, env_state: State, ppo_network: PPOImitationNetworks, key: PRNGKey, extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
    """Collect data."""
    actions, policy_extras = ppo_network.policy(env_state.obs, key)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )


def generate_unroll(
    env: Env,
    env_state: State,
    ppo_network: PPOImitationNetworks,  # need to check this
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        env_state, graph_def, state, current_key = carry
        ppo_network = nnx.merge(graph_def, state)
        current_key, next_key = jax.random.split(current_key)
        nstate, transition = actor_step(env, env_state, ppo_network, key, extra_fields=extra_fields)
        return (nstate, graph_def, state, next_key), transition

    graph_def, state = nnx.split(ppo_network)
    (final_state,_, _, _), data = jax.lax.scan(f, (env_state, graph_def, state, key), (), length=unroll_length)
    return final_state, data


# TODO: Consider moving this to its own file.
# TODO: Scott: Make this evaluator into nnx form
class Evaluator:
    """Class to run evaluations."""

    def __init__(
        self,
        eval_env: envs.Env,
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: PRNGKey,
    ):
        """Init.

        Args:
          eval_env: Batched environment to run evals on.
          eval_policy_fn: Function returning the policy from the policy parameters.
          num_eval_envs: Each env will run 1 episode in parallel for each eval.
          episode_length: Maximum length of an episode.
          action_repeat: Number of physics steps per env step.
          key: RNG key.
        """
        self._key = key
        self._eval_walltime = 0.0

        eval_env = envs.training.EvalWrapper(eval_env)

        def generate_eval_unroll(ppo_network: PPOImitationNetworks, key: PRNGKey) -> State:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_unroll(
                eval_env,
                eval_first_state,
                ppo_network,
                key,
                unroll_length=episode_length // action_repeat,
            )[0]

        self._generate_eval_unroll = nnx.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self, ppo_network: PPOImitationNetworks, training_metrics: Metrics, aggregate_episodes: bool = True
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(ppo_network, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        for fn in [np.mean, np.std]:
            suffix = "_std" if fn == np.std else ""
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (fn(value) if aggregate_episodes else value)
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )
        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray
