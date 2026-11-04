# acme TF DMPO reference fixture

`fixture.json` holds four learner steps of DeepMind acme's TF
`DistributionalMPOLearner` (`acme/agents/tf/dmpo/learning.py`, unmodified),
run on inputs that track-mjx builds. `tests/agent/dmpo/test_acme_parity.py`
replays the same inputs through `learner.sgd_step` on the aligned defaults and
compares every value acme reports or holds after each step, at `rtol=1e-4`,
`atol=1e-5`:

- `critic_loss`, `policy_loss` and every MPO statistic acme's loss returns;
- every online and target policy and critic weight;
- the four dual variables.

The test needs only the JSON. Regenerating it needs a separate TF environment
and runs by hand, never on CI.

## What the reference contains

`make_inputs.py` (track-mjx environment) builds the inputs:

- the config: small networks (policy and critic torsos 16-16), 11 atoms on
  [-5, 5], N = 4 samples, `n_step` 3 on the compressed replay layout
  (`sequence_length` 4), discount 0.9, target periods 2 (policy) and 3
  (critic). The six alignment fields stay at their defaults.
- the initial weights: the JAX init (`jax.random.key(0)`) plus N(0, 0.1^2)
  noise on every leaf (numpy seed 1), so the policy head (initialized near
  zero) and the LayerNorm scales and offsets are not trivially constant.
- per step, a batch of 6 sequences (numpy seed 2) and the target-policy noise
  eps, shape [N, B, A] (numpy seed 3). Rows 0, 1 and 5 are mid-episode. Rows
  2, 3 and 4 terminate at window step 0, 1 and 2; the values after the
  termination belong to the next episode and must not reach the loss. No row
  has a time-out (time-outs are covered by `test_truncation_bootstrap.py`).

`generate.py` (TF environment) runs acme on them:

- sonnet networks that compute what `networks._PolicyNet` and `_CriticNet`
  compute: Linear, LayerNorm (eps 1e-6), SiLU blocks; the
  `MultivariateNormalDiagHead` scale formula; a categorical head. They carry
  the inputs' weights.
- the critic's action branch applies `clip(tanh(a), ±(1 - 1e-6))`, the
  analogue of acme's `ClipToSpec` for a policy whose actions are tanh-bound at
  the env.
- the target policy's `sample(n)` returns `mu + sigma * eps` with the stored
  eps.
- each transition comes from its window through acme's own
  `NStepTransitionAdder._compute_cumulative_quantities`. A window that
  terminates at step k is the shorter transition acme writes at an episode
  end, with `next_observation` = s_k, the last state the JAX replay stores
  for that episode.
- `losses.MPO` with the config's values and `action_penalization=True`;
  `snt.optimizers.Adam` at the config's policy, critic and dual learning
  rates (sonnet's Adam has optax's defaults: b1 0.9, b2 0.999, eps 1e-8);
  `clipping=True` (global norm 40, the config's `grad_clip`).

The test aligns two timing conventions before comparing; neither changes a
value. acme copies online to target weights at the start of a step, track-mjx
at the end of the previous one, so after step j the JAX target already holds
what acme's step j + 1 copies in. acme floors the log duals at -18 at the
start of a step, track-mjx after its update.

The learning rates (1e-3 policy and critic, 1e-2 dual) and
`epsilon_stddev` 1e-3 are larger than the learner's defaults so that four
steps move the weights well beyond float32 noise; Adam's early updates are
close to `lr * sign(g)`, so the weight comparison mostly checks gradient
directions, and the losses and statistics of steps 2 to 4 check the values
those updates lead to.

## Regenerating

1. TF environment (Python 3.10; the versions below):

   ```bash
   uv venv --python 3.10 acme_tf_env
   uv pip install --python acme_tf_env/bin/python \
       tensorflow==2.8.0 tensorflow-probability==0.15.0 dm-sonnet==2.0.2 \
       trfl==1.2.0 dm-launchpad==0.5.2 dm-reverb==0.7.2 dm-tree==0.1.10 \
       dm-env==1.6 absl-py "numpy<1.24" "protobuf<3.20" pillow gym
   ```

   jax is not installed there: no jaxlib wheel for acme's pin resolves.
   `_stubs/jax` stands in for it; acme's TF code uses jax only in type hints
   and one adders helper import. `generate.py` appends `_stubs` to
   `sys.path` itself.

2. Inputs, in the track-mjx environment, from the repo root:

   ```bash
   JAX_PLATFORMS=cpu python -m tests.agent.dmpo.acme_tf_reference.make_inputs \
       /tmp/acme_parity_inputs.json
   ```

3. Reference, in the TF environment. launchpad's courier needs
   `libpython3.10.so` on the library path:

   ```bash
   LD_LIBRARY_PATH=<python 3.10 install>/lib CUDA_VISIBLE_DEVICES="" \
       acme_tf_env/bin/python tests/agent/dmpo/acme_tf_reference/generate.py \
       --acme <acme checkout> --inputs /tmp/acme_parity_inputs.json \
       --out tests/agent/dmpo/acme_tf_reference/fixture.json
   ```

4. `pytest tests/agent/dmpo/test_acme_parity.py`.

If a step stops matching, fix the port, not the reference.

## Versions

Recorded with the current fixture (also in its `versions` field):

| Package | Version |
|---|---|
| acme (source, read only) | 0.4.1, commit 2e09e3ddecf4a1ba783688c92aa493c596a2beec |
| python | 3.10.20 |
| tensorflow | 2.8.0 |
| tensorflow-probability | 0.15.0 |
| dm-sonnet | 2.0.2 |
| trfl | 1.2.0 |
| dm-launchpad | 0.5.2 |
| dm-reverb | 0.7.2 |
| dm-tree | 0.1.10 |
| dm-env | 1.6 |
| absl-py | 2.5.1 |
| numpy | 1.23.5 |
| protobuf | 3.19.6 |
| keras | 2.8.0 |
| gym | 0.26.2 |
| pillow | 12.3.0 |

JAX side when the fixture was recorded: python 3.12.13, jax and jaxlib 0.10.2,
flax 0.12.7, optax 0.2.8, tfp-nightly 0.26.0.dev20260916, numpy 2.1.3.
