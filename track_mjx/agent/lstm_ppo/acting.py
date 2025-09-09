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

"""Brax training lstm acting functions."""

import time
from typing import Callable, Sequence, Tuple, Union

from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn

State = envs.State
Env = envs.Env


def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    hidden_state: tuple[jnp.ndarray, jnp.ndarray],
    extra_fields: Sequence[str] = (),
) -> tuple[State, Transition, jnp.ndarray]:
    """Collect data and update LSTM hidden state."""

    actions, policy_extras, new_hidden_state = policy(env_state.obs, key, hidden_state)

    info_hidden = env_state.info["hidden_state"]

    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    done = nstate.done[:, None]  # done flags (batch_size, num_envs)
    done = done.reshape((done.shape[0], 1, 1))  # (128, 1, 1)

    new_hidden_state = jax.tree_util.tree_map(
        lambda info_h, h: jnp.where(done, info_h, h), info_hidden, new_hidden_state
    )
    new_hidden_state = jax.tree_map(jax.lax.stop_gradient, new_hidden_state)

    return (
        nstate,
        Transition(
            observation=env_state.obs,  # start with first obs, nstate as next obs will feed in and update next iter
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={
                "policy_extras": policy_extras,
                "state_extras": state_extras,
                "hidden_state": hidden_state[
                    0
                ],  # lag one, or else first hidden_state lost, should be hidden before policy
                "cell_state": hidden_state[1],
            },
        ),
        new_hidden_state,
    )  # use for forward


def generate_unroll(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    hidden_state: tuple[jnp.ndarray, jnp.ndarray],
    unroll_length: int,
    extra_fields: Sequence[str] = (),
) -> tuple[State, Transition, jnp.ndarray]:
    """Collect trajectories of given unroll_length while tracking LSTM state."""

    @jax.jit
    def f(carry, unused_t):
        state, current_key, hidden_state = carry  # track hidden state
        current_key, next_key = jax.random.split(current_key)

        nstate, transition, new_hidden_state = actor_step(
            env, state, policy, current_key, hidden_state, extra_fields=extra_fields
        )  # updated hidden state

        # both here and in forward, provide unroll_length here
        # return the final hidden in carry for carry alignment (forward), return stacked hidden in transition extra field
        return (nstate, next_key, new_hidden_state), transition

    # carry shape need to match, so out hidden would not have unroll length if in carry
    (final_state, _, forward_hidden_state), data = jax.lax.scan(
        f, (env_state, key, hidden_state), (), length=unroll_length
    )

    return final_state, data, forward_hidden_state


class Evaluator:
    """Class to run evaluations with LSTM state handling."""

    def __init__(
        self,
        eval_env: envs.Env,
        eval_policy_fn: Callable[[PolicyParams], Policy],
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

        def generate_eval_unroll(
            policy_params: PolicyParams, key: PRNGKey
        ) -> tuple[State, tuple[jnp.ndarray, jnp.ndarray]]:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)

            dummy_hidden_state = eval_first_state.info["hidden_state"]

            # unstack one here
            final_state, _, final_hidden_state = generate_unroll(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params),
                key,
                dummy_hidden_state,
                unroll_length=episode_length // action_repeat,
            )
            return final_state, final_hidden_state

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation with LSTM tracking."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()

        eval_state, hidden_state = self._generate_eval_unroll(policy_params, unroll_key)

        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}

        for fn in [np.mean, np.std]:
            suffix = "_std" if fn == np.std else ""
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (
                        fn(value) if aggregate_episodes else value
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray
