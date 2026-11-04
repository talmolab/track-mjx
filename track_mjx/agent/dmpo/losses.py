# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
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

"""Continuous MPO loss with dual variables.

Vendored from acme.jax.losses.mpo (Apache 2.0). See LICENSE-ACME for the
original copyright notice.

Modifications:
  1. Selectable action-penalization formula (`action_penalty_formula`):
     Acme's `-norm(actions - clip(actions, -1, 1))` ("acme", the default) or
     vnl-ray's `-norm(actions)` ("vnl_ray", the algorithm vnl-ray trained
     with on downstream tasks). See `cost_out_of_bound` in __call__.
  2. Optional per-state `batch_weights` in `MPO.__call__` (and the helpers it
     calls): every mean over the batch becomes `sum(w*x)/max(sum(w), 1)`, so
     states can be excluded. `None` keeps the unweighted mean exactly.

The MPO loss uses MPOParams, which can be initialized using init_params,
to track the temperature and the dual variables.

Tensor shapes are annotated, where helpful, as follow:
  B: batch size,
  N: number of sampled actions, see MPO paper for more details,
  D: dimensionality of the action space.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

_MPO_FLOAT_EPSILON = 1e-8
_PENALTY_FORMULAS = ("acme", "vnl_ray")
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0

Shape = tuple[int]
DType = type(jnp.float32)  # _ScalarMeta, a private type.


def _reciprocal_mass(weights: jnp.ndarray) -> jnp.ndarray:
    """``1 / max(sum(w), 1)``, the factor of the weighted mean.

    Applied as a multiplication rather than a division on purpose: XLA lowers
    the divide-by-constant-batch-size inside ``jnp.mean`` to a multiplication by
    the f32 reciprocal, which is not bitwise equal to a true division for some
    inputs. Multiplying by the runtime reciprocal reproduces ``jnp.mean`` exactly
    when every weight is 1.
    """
    return 1.0 / jnp.maximum(jnp.sum(weights), 1.0)


def _batch_mean(x: jnp.ndarray, weights: jnp.ndarray | None) -> jnp.ndarray:
    """Mean over the leading batch axis; weighted when ``weights`` is given.

    ``weights`` is ``[B]`` and nonnegative. The weighted mean is
    ``sum(w * x) / max(sum(w), 1)``: a 0/1 mask averages over the kept rows, and
    a batch whose weights sum to zero gives exactly 0 (not NaN). ``None``
    is the plain ``jnp.mean(x, axis=0)``, chosen in Python so the unweighted
    path is op for op what it has always been; all-ones weights match it bitwise.
    """
    if weights is None:
        return jnp.mean(x, axis=0)
    w = weights.reshape((-1,) + (1,) * (x.ndim - 1))
    return jnp.sum(w * x, axis=0) * _reciprocal_mass(weights)


def _mean_over_samples_and_batch(
    x: jnp.ndarray, weights: jnp.ndarray | None
) -> jnp.ndarray:
    """Mean of an ``[N, B]`` array over both axes, weighting the B axis."""
    if weights is None:
        return jnp.mean(x)
    return jnp.sum(weights * x) * (
        1.0 / (x.shape[0] * jnp.maximum(jnp.sum(weights), 1.0))
    )


def _scale_constant(c: float | jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """The weighted batch mean of the constant ``c``: ``c * sum(w) / max(sum(w), 1)``.

    Constants inside a batch mean (the epsilon and log N terms of the dual
    losses) go through this so a fully excluded batch contributes nothing. When
    the weights sum to at least 1 it is ``c`` itself, selected rather than
    multiplied by 1, so a batch of ones leaves the arithmetic untouched.
    """
    total = jnp.sum(weights)
    return jnp.where(total >= 1.0, c, total * c)


class MPOParams(NamedTuple):
    """NamedTuple to store trainable loss parameters."""

    log_temperature: jnp.ndarray
    log_alpha_mean: jnp.ndarray
    log_alpha_stddev: jnp.ndarray
    log_penalty_temperature: jnp.ndarray | None = None


class MPOStats(NamedTuple):
    """NamedTuple to store loss statistics."""

    dual_alpha_mean: float | jnp.ndarray
    dual_alpha_stddev: float | jnp.ndarray
    dual_temperature: float | jnp.ndarray

    loss_policy: float | jnp.ndarray
    loss_alpha: float | jnp.ndarray
    loss_temperature: float | jnp.ndarray
    kl_q_rel: float | jnp.ndarray

    kl_mean_rel: float | jnp.ndarray
    kl_stddev_rel: float | jnp.ndarray

    q_min: float | jnp.ndarray
    q_max: float | jnp.ndarray

    pi_stddev_min: float | jnp.ndarray
    pi_stddev_max: float | jnp.ndarray
    pi_stddev_cond: float | jnp.ndarray

    # Mean L2 norm of pre-tanh actions sampled from the target policy. Useful
    # for tuning epsilon_penalty (action_penalization budget): the MO-MPO penalty
    # biases the policy toward small ‖a‖, so this is the headline number to watch
    # when assessing whether the L2-action regulariser is over-tightening under
    # torque actuation.
    mean_action_norm: float | jnp.ndarray = 0.0
    max_action_norm: float | jnp.ndarray = 0.0

    penalty_kl_q_rel: float | None = None

    # KL-anchor stats (populated only when kl_anchor_alpha > 0; otherwise 0).
    anchor_kl_mean: float | jnp.ndarray = 0.0
    anchor_reward_mean: float | jnp.ndarray = 0.0
    anchor_loss_term: float | jnp.ndarray = 0.0
    # Current value of `w` after applying the linear-decay schedule (or the
    # static `kl_anchor_w` if decay is disabled / anchor branch inactive).
    anchor_w_now: float | jnp.ndarray = 0.0


class MPO:
    """MPO loss with decoupled KL constraints as in (Abdolmaleki et al., 2018).

    This implementation of the MPO loss includes the following features, as
    options:
    - Satisfying the KL-constraint on a per-dimension basis (on by default);
    - Penalizing actions that fall outside of [-1, 1] (on by default) as a
        special case of multi-objective MPO (MO-MPO; Abdolmaleki et al., 2020).
    For best results on the control suite, keep both of these on.

    (Abdolmaleki et al., 2018): https://arxiv.org/pdf/1812.02256.pdf
    (Abdolmaleki et al., 2020): https://arxiv.org/pdf/2005.07513.pdf
    """

    def __init__(
        self,
        epsilon: float,
        epsilon_mean: float,
        epsilon_stddev: float,
        init_log_temperature: float,
        init_log_alpha_mean: float,
        init_log_alpha_stddev: float,
        per_dim_constraining: bool = True,
        action_penalization: bool = True,
        epsilon_penalty: float = 0.001,
        action_penalty_formula: str = "acme",
    ):
        """Initialize and configure the MPO loss.

        Args:
          epsilon: KL constraint on the non-parametric auxiliary policy, the one
            associated with the dual variable called temperature.
          epsilon_mean: KL constraint on the mean of the Gaussian policy, the one
            associated with the dual variable called alpha_mean.
          epsilon_stddev: KL constraint on the stddev of the Gaussian policy, the
            one associated with the dual variable called alpha_mean.
          init_log_temperature: initial value for the temperature in log-space, note
            a softplus (rather than an exp) will be used to transform this.
          init_log_alpha_mean: initial value for the alpha_mean in log-space, note a
            softplus (rather than an exp) will be used to transform this.
          init_log_alpha_stddev: initial value for the alpha_stddev in log-space,
            note a softplus (rather than an exp) will be used to transform this.
          per_dim_constraining: whether to enforce the KL constraint on each
            dimension independently; this is the default. Otherwise the overall KL
            is constrained, which allows some dimensions to change more at the
            expense of others staying put.
          action_penalization: whether to use a KL constraint to penalize actions
            via the MO-MPO algorithm.
          epsilon_penalty: KL constraint on the probability of violating the action
            constraint.
          action_penalty_formula: the out-of-bound cost of a sampled action.
            "acme": `-norm(a - clip(a, -1, 1))`, only the part outside [-1, 1]
            (applied to the pre-tanh samples, it penalizes |a| > 1, i.e.
            |tanh(a)| > 0.76). "vnl_ray": `-norm(a)`, every action magnitude.
        """
        if action_penalty_formula not in _PENALTY_FORMULAS:
            raise ValueError(
                f"unknown action_penalty_formula={action_penalty_formula!r}; "
                "valid values: " + ", ".join(repr(f) for f in _PENALTY_FORMULAS)
            )

        # MPO constrain thresholds.
        self._epsilon = epsilon
        self._epsilon_mean = epsilon_mean
        self._epsilon_stddev = epsilon_stddev

        # Initial values for the constraints' dual variables.
        self._init_log_temperature = init_log_temperature
        self._init_log_alpha_mean = init_log_alpha_mean
        self._init_log_alpha_stddev = init_log_alpha_stddev

        # Whether to penalize out-of-bound actions via MO-MPO and its corresponding
        # constraint threshold.
        self._action_penalization = action_penalization
        self._epsilon_penalty = epsilon_penalty
        self._action_penalty_formula = action_penalty_formula

        # Whether to ensure per-dimension KL constraint satisfication.
        self._per_dim_constraining = per_dim_constraining

    @property
    def per_dim_constraining(self):
        return self._per_dim_constraining

    def init_params(self, action_dim: int, dtype: DType = jnp.float32):
        """Creates an initial set of parameters."""

        dual_variable_shape = [action_dim] if self._per_dim_constraining else [1]

        log_temperature = jnp.full([1], self._init_log_temperature, dtype=dtype)

        log_alpha_mean = jnp.full(
            dual_variable_shape, self._init_log_alpha_mean, dtype=dtype
        )

        log_alpha_stddev = jnp.full(
            dual_variable_shape, self._init_log_alpha_stddev, dtype=dtype
        )

        if self._action_penalization:
            log_penalty_temperature = jnp.full(
                [1], self._init_log_temperature, dtype=dtype
            )
        else:
            log_penalty_temperature = None

        return MPOParams(
            log_temperature=log_temperature,
            log_alpha_mean=log_alpha_mean,
            log_alpha_stddev=log_alpha_stddev,
            log_penalty_temperature=log_penalty_temperature,
        )

    def __call__(
        self,
        params: MPOParams,
        online_action_distribution: tfd.MultivariateNormalDiag | tfd.Independent,
        target_action_distribution: tfd.MultivariateNormalDiag | tfd.Independent,
        actions: jnp.ndarray,  # Shape [N, B, D].
        q_values: jnp.ndarray,  # Shape [N, B].
        batch_weights: jnp.ndarray | None = None,  # Shape [B].
    ) -> tuple[jnp.ndarray, MPOStats]:
        """Computes the decoupled MPO loss.

        Args:
          params: parameters tracking the temperature and the dual variables.
          online_action_distribution: online distribution returned by the online
            policy network; expects batch_dims of [B] and event_dims of [D].
          target_action_distribution: target distribution returned by the target
            policy network; expects same shapes as online distribution.
          actions: actions sampled from the target policy; expects shape [N, B, D].
          q_values: Q-values associated with each action; expects shape [N, B].
          batch_weights: optional nonnegative per-state weights, shape [B]. Every
            mean over the batch (losses and statistics alike) becomes
            ``sum(w * x) / max(sum(w), 1)``, so a zero-weight state is excluded
            and an all-zero batch gives a finite loss with zero gradient. ``None``
            (default) is the plain unweighted mean, bit-identical to before.

        Returns:
          Loss, combining the policy loss, KL penalty, and dual losses required to
            adapt the dual variables.
          Stats, for diagnostics and tracking performance.
        """

        # Cast `MultivariateNormalDiag`s to Independent Normals.
        # The latter allows us to satisfy KL constraints per-dimension.
        if isinstance(target_action_distribution, tfd.MultivariateNormalDiag):
            target_action_distribution = tfd.Independent(
                tfd.Normal(
                    target_action_distribution.mean(),
                    target_action_distribution.stddev(),
                )
            )
            online_action_distribution = tfd.Independent(
                tfd.Normal(
                    online_action_distribution.mean(),
                    online_action_distribution.stddev(),
                )
            )

        # Transform dual variables from log-space.
        # Note: using softplus instead of exponential for numerical stability.
        temperature = jax.nn.softplus(params.log_temperature) + _MPO_FLOAT_EPSILON
        alpha_mean = jax.nn.softplus(params.log_alpha_mean) + _MPO_FLOAT_EPSILON
        alpha_stddev = jax.nn.softplus(params.log_alpha_stddev) + _MPO_FLOAT_EPSILON

        # Get online and target means and stddevs in preparation for decomposition.
        online_mean = online_action_distribution.distribution.mean()
        online_scale = online_action_distribution.distribution.stddev()
        target_mean = target_action_distribution.distribution.mean()
        target_scale = target_action_distribution.distribution.stddev()

        # Compute normalized importance weights, used to compute expectations with
        # respect to the non-parametric policy; and the temperature loss, used to
        # adapt the tempering of Q-values.
        normalized_weights, loss_temperature = compute_weights_and_temperature_loss(
            q_values, self._epsilon, temperature, batch_weights
        )

        # Only needed for diagnostics: Compute estimated actualized KL between the
        # non-parametric and current target policies.
        kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
            normalized_weights
        )

        if self._action_penalization:
            # Transform action penalization temperature.
            penalty_temperature = (
                jax.nn.softplus(params.log_penalty_temperature) + _MPO_FLOAT_EPSILON
            )

            # Compute action penalization cost.
            if self._action_penalty_formula == "acme":
                # Acme: only the out-of-bound portion of each action.
                diff_out_of_bound = actions - jnp.clip(actions, -1.0, 1.0)
                cost_out_of_bound = -jnp.linalg.norm(diff_out_of_bound, axis=-1)
            else:
                # vnl-ray's formulation penalizes ALL action magnitudes, the
                # algorithm vnl-ray trained with on downstream tasks (its
                # vnl_ray/agents/losses_mpo.py keeps Acme's form as comments).
                cost_out_of_bound = -jnp.linalg.norm(actions, axis=-1)

            penalty_normalized_weights, loss_penalty_temperature = (
                compute_weights_and_temperature_loss(
                    cost_out_of_bound,
                    self._epsilon_penalty,
                    penalty_temperature,
                    batch_weights,
                )
            )

            # Only needed for diagnostics: Compute estimated actualized KL between the
            # non-parametric and current target policies.
            penalty_kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
                penalty_normalized_weights
            )

            # Combine normalized weights.
            normalized_weights += penalty_normalized_weights
            loss_temperature += loss_penalty_temperature

        # Decompose the online policy into fixed-mean & fixed-stddev distributions.
        # This has been documented as having better performance in bandit settings,
        # see e.g. https://arxiv.org/pdf/1812.02256.pdf.
        fixed_stddev_distribution = tfd.Independent(
            tfd.Normal(loc=online_mean, scale=target_scale)
        )
        fixed_mean_distribution = tfd.Independent(
            tfd.Normal(loc=target_mean, scale=online_scale)
        )

        # Compute the decomposed policy losses.
        loss_policy_mean = compute_cross_entropy_loss(
            actions, normalized_weights, fixed_stddev_distribution, batch_weights
        )
        loss_policy_stddev = compute_cross_entropy_loss(
            actions, normalized_weights, fixed_mean_distribution, batch_weights
        )

        # Compute the decomposed KL between the target and online policies.
        if self._per_dim_constraining:
            kl_mean = target_action_distribution.distribution.kl_divergence(
                fixed_stddev_distribution.distribution
            )  # Shape [B, D].
            kl_stddev = target_action_distribution.distribution.kl_divergence(
                fixed_mean_distribution.distribution
            )  # Shape [B, D].
        else:
            kl_mean = target_action_distribution.kl_divergence(
                fixed_stddev_distribution
            )  # Shape [B].
            kl_stddev = target_action_distribution.kl_divergence(
                fixed_mean_distribution
            )  # Shape [B].

        # Compute the alpha-weighted KL-penalty and dual losses to adapt the alphas.
        loss_kl_mean, loss_alpha_mean = compute_parametric_kl_penalty_and_dual_loss(
            kl_mean, alpha_mean, self._epsilon_mean, batch_weights
        )
        loss_kl_stddev, loss_alpha_stddev = compute_parametric_kl_penalty_and_dual_loss(
            kl_stddev, alpha_stddev, self._epsilon_stddev, batch_weights
        )

        # Combine losses.
        loss_policy = loss_policy_mean + loss_policy_stddev
        loss_kl_penalty = loss_kl_mean + loss_kl_stddev
        loss_dual = loss_alpha_mean + loss_alpha_stddev + loss_temperature
        loss = loss_policy + loss_kl_penalty + loss_dual

        # Create statistics.
        pi_stddev = online_action_distribution.distribution.stddev()
        action_norms = jnp.linalg.norm(actions, axis=-1)  # Shape [N, B].
        stats = MPOStats(
            # Dual Variables.
            dual_alpha_mean=jnp.mean(alpha_mean),
            dual_alpha_stddev=jnp.mean(alpha_stddev),
            dual_temperature=jnp.mean(temperature),
            # Losses.
            loss_policy=jnp.mean(loss),
            loss_alpha=jnp.mean(loss_alpha_mean + loss_alpha_stddev),
            loss_temperature=jnp.mean(loss_temperature),
            # KL measurements.
            kl_q_rel=_batch_mean(kl_nonparametric, batch_weights) / self._epsilon,
            penalty_kl_q_rel=(
                (
                    _batch_mean(penalty_kl_nonparametric, batch_weights)
                    / self._epsilon_penalty
                )
                if self._action_penalization
                else None
            ),
            kl_mean_rel=_batch_mean(kl_mean, batch_weights) / self._epsilon_mean,
            kl_stddev_rel=_batch_mean(kl_stddev, batch_weights) / self._epsilon_stddev,
            # Q measurements.
            q_min=_batch_mean(jnp.min(q_values, axis=0), batch_weights),
            q_max=_batch_mean(jnp.max(q_values, axis=0), batch_weights),
            # If the policy has stddev, log summary stats for this as well.
            pi_stddev_min=_batch_mean(jnp.min(pi_stddev, axis=-1), batch_weights),
            pi_stddev_max=_batch_mean(jnp.max(pi_stddev, axis=-1), batch_weights),
            # Condition number of the diagonal covariance (actually, stddev) matrix.
            pi_stddev_cond=_batch_mean(
                jnp.max(pi_stddev, axis=-1) / jnp.min(pi_stddev, axis=-1),
                batch_weights,
            ),
            # Action-norm diagnostics (see MPOStats docstring).
            mean_action_norm=_mean_over_samples_and_batch(action_norms, batch_weights),
            max_action_norm=_batch_mean(jnp.max(action_norms, axis=0), batch_weights),
        )

        return loss, stats


def compute_weights_and_temperature_loss(
    q_values: jnp.ndarray,
    epsilon: float,
    temperature: jnp.ndarray,
    batch_weights: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes normalized importance weights for the policy optimization.

    Args:
      q_values: Q-values associated with the actions sampled from the target
        policy; expected shape [N, B].
      epsilon: Desired constraint on the KL between the target and non-parametric
        policies.
      temperature: Scalar used to temper the Q-values before computing normalized
        importance weights from them. This is really the Lagrange dual variable in
        the constrained optimization problem, the solution of which is the
        non-parametric policy targeted by the policy loss.
      batch_weights: optional nonnegative per-state weights, shape [B]. The
        batch mean in the temperature loss is then weighted, including its
        constant epsilon and log N terms, so an all-zero batch gives 0.

    Returns:
      Normalized importance weights, used for policy optimization.
      Temperature loss, used to adapt the temperature.
    """

    # Temper the given Q-values using the current temperature.
    tempered_q_values = jax.lax.stop_gradient(q_values) / temperature

    # Compute the normalized importance weights used to compute expectations with
    # respect to the non-parametric policy.
    normalized_weights = jax.nn.softmax(tempered_q_values, axis=0)
    normalized_weights = jax.lax.stop_gradient(normalized_weights)

    # Compute the temperature loss (dual of the E-step optimization problem).
    q_logsumexp = jax.scipy.special.logsumexp(tempered_q_values, axis=0)
    log_num_actions = jnp.log(q_values.shape[0] / 1.0)
    if batch_weights is None:
        loss_temperature = epsilon + jnp.mean(q_logsumexp) - log_num_actions
    else:
        loss_temperature = (
            _scale_constant(epsilon, batch_weights)
            + _batch_mean(q_logsumexp, batch_weights)
            - _scale_constant(log_num_actions, batch_weights)
        )
    loss_temperature = temperature * loss_temperature

    return normalized_weights, loss_temperature


def compute_nonparametric_kl_from_normalized_weights(
    normalized_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate the actualized KL between the non-parametric and target policies."""

    # Compute integrand.
    num_action_samples = normalized_weights.shape[0] / 1.0
    integrand = jnp.log(num_action_samples * normalized_weights + 1e-8)

    # Return the expectation with respect to the non-parametric policy.
    return jnp.sum(normalized_weights * integrand, axis=0)


def compute_cross_entropy_loss(
    sampled_actions: jnp.ndarray,
    normalized_weights: jnp.ndarray,
    online_action_distribution: tfd.Distribution,
    batch_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute cross-entropy online and the reweighted target policy.

    Args:
      sampled_actions: samples used in the Monte Carlo integration in the policy
        loss. Expected shape is [N, B, ...], where N is the number of sampled
        actions and B is the number of sampled states.
      normalized_weights: target policy multiplied by the exponentiated Q values
        and normalized; expected shape is [N, B].
      online_action_distribution: policy to be optimized.
      batch_weights: optional nonnegative per-state weights, shape [B]; the
        mean over states is weighted (see ``_batch_mean``).

    Returns:
      loss_policy_gradient: the cross-entropy loss that, when differentiated,
        produces the policy gradient.
    """

    # Compute the M-step loss.
    log_prob = online_action_distribution.log_prob(sampled_actions)

    # Compute the weighted average log-prob using the normalized weights.
    loss_policy_gradient = -jnp.sum(log_prob * normalized_weights, axis=0)

    # Return the mean loss over the batch of states.
    return _batch_mean(loss_policy_gradient, batch_weights)


def compute_parametric_kl_penalty_and_dual_loss(
    kl: jnp.ndarray,
    alpha: jnp.ndarray,
    epsilon: float,
    batch_weights: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the KL cost to be added to the Lagragian and its dual loss.

    The KL cost is simply the alpha-weighted KL divergence and it is added as a
    regularizer to the policy loss. The dual variable alpha itself has a loss that
    can be minimized to adapt the strength of the regularizer to keep the KL
    between consecutive updates at the desired target value of epsilon.

    Args:
      kl: KL divergence between the target and online policies.
      alpha: Lagrange multipliers (dual variables) for the KL constraints.
      epsilon: Desired value for the KL.
      batch_weights: optional nonnegative per-state weights, shape [B]. The mean
        KL over states is weighted, and the constant epsilon of the dual loss is
        scaled by the same total weight, so an all-zero batch gives 0.

    Returns:
      loss_kl: alpha-weighted KL regularization to be added to the policy loss.
      loss_alpha: The Lagrange dual loss minimized to adapt alpha.
    """

    # Compute the mean KL over the batch.
    mean_kl = _batch_mean(kl, batch_weights)

    # Compute the regularization.
    loss_kl = jnp.sum(jax.lax.stop_gradient(alpha) * mean_kl)

    # Compute the dual loss.
    if batch_weights is None:
        loss_alpha = jnp.sum(alpha * (epsilon - jax.lax.stop_gradient(mean_kl)))
    else:
        loss_alpha = jnp.sum(
            alpha
            * (_scale_constant(epsilon, batch_weights) - jax.lax.stop_gradient(mean_kl))
        )

    return loss_kl, loss_alpha


def clip_mpo_params(
    params: MPOParams,
    min_log_temperature: float = _MIN_LOG_TEMPERATURE,
) -> MPOParams:
    """Floor each dual variable in log-space at _MIN_LOG_{TEMPERATURE,ALPHA}.

    Mirrors the projection that vnl-ray/Acme apply at the start of every MPO
    __call__. log_penalty_temperature is preserved iff it was created
    (i.e. iff the loss was constructed with action_penalization=True).

    ``min_log_temperature`` is exposed because the default of -18 is not a safe
    floor under a SPARSE reward, where Q can go exactly flat across the N sampled
    actions. When that happens ``compute_weights_and_temperature_loss`` reduces to

        loss = t * (epsilon + Q/t + log N - log N) = t * epsilon + Q

    whose derivative is ``epsilon > 0`` for every t, so the temperature decays
    without bound and pins here. Measured on arm_m1_sparse_gaponly:
    log_temperature -4.06 @10.1M -> -9.31 @20.8M -> -18.00 @31.4M, with kl_q_rel
    (realized KL(q||pi)/epsilon) falling 0.966 -> 0.441 -> 0.00002 and gap
    crossings/episode collapsing 0.221 -> 0.260 -> 0.030. With genuinely spread Q
    the small-t limit is ``epsilon - log N`` = 0.1 - 3.0 < 0, i.e. a restoring
    force; exact flatness is what removes it.

    A HIGHER floor is the conservative choice: it keeps the E-step weights
    near-uniform (a no-op update) instead of letting 1/t amplify critic noise into
    a hard argmax over the sampled actions. For reference the healthy dense arm
    arm_i1_nstep100_proprio sits at log_temperature -0.54 at 297.7M.

    NOTE this bounds the FAILURE, it does not fix the CAUSE. If Q is flat because
    the critic has stopped depending on the action, a higher floor stops the
    policy chasing noise but does not restore an improvement signal.

    The default reproduces the previous behaviour exactly, so no completed arm
    changes.
    """
    clipped_params = MPOParams(
        log_temperature=jnp.maximum(min_log_temperature, params.log_temperature),
        log_alpha_mean=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_mean),
        log_alpha_stddev=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_stddev),
    )
    if params.log_penalty_temperature is None:
        return clipped_params
    return clipped_params._replace(
        log_penalty_temperature=jnp.maximum(
            min_log_temperature, params.log_penalty_temperature
        )
    )
