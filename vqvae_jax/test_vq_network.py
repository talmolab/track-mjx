"""Unit tests for VQ-VAE intention network components.

Tests verify:
1. Shape correctness at each stage
2. Gradient flow through quantization (straight-through estimator)
3. Codebook updates via gradients
4. Loss computation and metric calculations
"""

import jax
import jax.numpy as jnp
import pytest

# Add scratch directory to path for imports
import sys

sys.path.insert(0, "/home/jovyan/vast/kaiwen/track-mjx/scratch/vqvae_jax")

from vq_intention_network import (
    VQEncoder,
    VectorQuantizer,
    Decoder,
    VQIntentionNetwork,
    make_vq_intention_policy,
    TemporalConvBlock,
    VQTemporalEncoder,
    VQTemporalIntentionNetwork,
    upsample_temporal,
)
from vq_losses import (
    compute_vq_loss,
    compute_codebook_metrics,
    compute_ce_stickiness_cost,
    compute_vq_ppo_loss,
    PPONetworkParams,
)
from vq_ppo_networks import (
    make_vq_intention_ppo_networks,
    make_vq_inference_fn,
)


class TestVQEncoder:
    """Test VQEncoder shape and gradient behavior."""

    def test_output_shape(self):
        """Verify encoder output has correct shape."""
        encoder = VQEncoder(layer_sizes=[256, 128], latent_dim=64)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((4, 100))  # [batch, input_dim]

        params = encoder.init(key, x)
        z_e = encoder.apply(params, x)

        assert z_e.shape == (4, 64), f"Expected (4, 64), got {z_e.shape}"

    def test_output_shape_with_time(self):
        """Verify encoder handles time dimension."""
        encoder = VQEncoder(layer_sizes=[256, 128], latent_dim=64)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((10, 4, 100))  # [time, batch, input_dim]

        params = encoder.init(key, x)
        z_e = encoder.apply(params, x)

        assert z_e.shape == (10, 4, 64), f"Expected (10, 4, 64), got {z_e.shape}"

    def test_activation_output(self):
        """Verify activation dict is returned correctly."""
        encoder = VQEncoder(layer_sizes=[256, 128], latent_dim=64)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((4, 100))

        params = encoder.init(key, x, get_activation=True)
        z_e, activations = encoder.apply(params, x, get_activation=True)

        assert "layer_0" in activations
        assert "layer_1" in activations
        assert "z_e" in activations
        assert activations["layer_0"].shape == (4, 256)
        assert activations["layer_1"].shape == (4, 128)


class TestVectorQuantizer:
    """Test VectorQuantizer behavior."""

    def test_output_shapes(self):
        """Verify quantizer output shapes."""
        quantizer = VectorQuantizer(num_codes=512, latent_dim=64)
        key = jax.random.PRNGKey(0)
        z_e = jnp.ones((4, 64))  # [batch, latent_dim]

        params = quantizer.init(key, z_e)
        z_q_st, indices, z_q = quantizer.apply(params, z_e)

        assert z_q_st.shape == (4, 64), f"z_q_st shape: {z_q_st.shape}"
        assert indices.shape == (4,), f"indices shape: {indices.shape}"
        assert z_q.shape == (4, 64), f"z_q shape: {z_q.shape}"

    def test_output_shapes_with_time(self):
        """Verify quantizer handles time dimension."""
        quantizer = VectorQuantizer(num_codes=512, latent_dim=64)
        key = jax.random.PRNGKey(0)
        z_e = jnp.ones((10, 4, 64))  # [time, batch, latent_dim]

        params = quantizer.init(key, z_e)
        z_q_st, indices, z_q = quantizer.apply(params, z_e)

        assert z_q_st.shape == (10, 4, 64)
        assert indices.shape == (10, 4)
        assert z_q.shape == (10, 4, 64)

    def test_straight_through_gradient(self):
        """Verify gradients flow through quantization to encoder."""
        quantizer = VectorQuantizer(num_codes=512, latent_dim=64)
        key = jax.random.PRNGKey(0)
        z_e = jnp.ones((4, 64))

        params = quantizer.init(key, z_e)

        def loss_fn(z_e_input):
            z_q_st, _, _ = quantizer.apply(params, z_e_input)
            return jnp.mean(z_q_st**2)

        # Gradient should flow through straight-through estimator
        grad = jax.grad(loss_fn)(z_e)
        assert grad is not None
        assert grad.shape == z_e.shape
        assert not jnp.allclose(grad, 0.0), "Gradients should not be zero"

    def test_codebook_gradient(self):
        """Verify gradients flow to codebook via z_q."""
        quantizer = VectorQuantizer(num_codes=512, latent_dim=64)
        key = jax.random.PRNGKey(0)
        z_e = jnp.ones((4, 64))

        params = quantizer.init(key, z_e)

        def loss_fn(params):
            z_q_st, indices, z_q = quantizer.apply(params, z_e)
            # Codebook loss: gradient to codebook via z_q
            codebook = params["params"]["embeddings"]
            z_q_from_codebook = codebook[indices]
            return jnp.mean((jax.lax.stop_gradient(z_e) - z_q_from_codebook) ** 2)

        grad = jax.grad(loss_fn)(params)
        codebook_grad = grad["params"]["embeddings"]
        assert codebook_grad is not None
        assert codebook_grad.shape == (512, 64)
        # Only used codes should have non-zero gradients
        assert jnp.any(
            codebook_grad != 0
        ), "Some codebook entries should have gradients"


class TestDecoder:
    """Test Decoder shape behavior."""

    def test_output_shape(self):
        """Verify decoder output shape."""
        decoder = Decoder(layer_sizes=[256, 128, 32])
        key = jax.random.PRNGKey(0)
        x = jnp.ones((4, 100))  # [batch, input_dim]

        params = decoder.init(key, x)
        output, _ = decoder.apply(params, x)

        assert output.shape == (4, 32), f"Expected (4, 32), got {output.shape}"


class TestVQIntentionNetwork:
    """Test full VQ-VAE intention network."""

    def test_output_shapes(self):
        """Verify full network output shapes."""
        network = VQIntentionNetwork(
            encoder_layers=[256, 128],
            decoder_layers=[256, 128, 32],
            reference_obs_size=50,
            latent_dim=64,
            num_codes=512,
        )
        key = jax.random.PRNGKey(0)
        obs = jnp.ones((4, 100))  # [batch, obs_dim]

        params = network.init(key, obs, key)
        action, z_e, indices = network.apply(params, obs, key)

        assert action.shape == (4, 32), f"action shape: {action.shape}"
        assert z_e.shape == (4, 64), f"z_e shape: {z_e.shape}"
        assert indices.shape == (4,), f"indices shape: {indices.shape}"

    def test_output_shapes_with_time(self):
        """Verify network handles time dimension."""
        network = VQIntentionNetwork(
            encoder_layers=[256, 128],
            decoder_layers=[256, 128, 32],
            reference_obs_size=50,
            latent_dim=64,
            num_codes=512,
        )
        key = jax.random.PRNGKey(0)
        obs = jnp.ones((10, 4, 100))  # [time, batch, obs_dim]

        params = network.init(key, obs, key)
        action, z_e, indices = network.apply(params, obs, key)

        assert action.shape == (10, 4, 32)
        assert z_e.shape == (10, 4, 64)
        assert indices.shape == (10, 4)

    def test_end_to_end_gradient(self):
        """Verify gradients flow through entire network."""
        network = VQIntentionNetwork(
            encoder_layers=[256, 128],
            decoder_layers=[256, 128, 32],
            reference_obs_size=50,
            latent_dim=64,
            num_codes=512,
        )
        key = jax.random.PRNGKey(0)
        obs = jnp.ones((4, 100))

        params = network.init(key, obs, key)

        def loss_fn(params):
            action, z_e, indices = network.apply(params, obs, key)
            return jnp.mean(action**2)

        grad = jax.grad(loss_fn)(params)
        # Check encoder gradients exist
        encoder_grad = grad["params"]["encoder"]
        assert encoder_grad is not None
        # Check decoder gradients exist
        decoder_grad = grad["params"]["decoder"]
        assert decoder_grad is not None
        # Check codebook gradients exist (via straight-through)
        codebook_grad = grad["params"]["quantizer"]["embeddings"]
        assert codebook_grad is not None


class TestTemporalConvBlock:
    """Test TemporalConvBlock shape and behavior."""

    def test_output_shape(self):
        """Verify temporal conv block downsamples correctly."""
        block = TemporalConvBlock(out_channels=64, kernel_size=3, stride=2)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((4, 20, 128))  # [batch, time, channels]

        params = block.init(key, x)
        y = block.apply(params, x)

        # With stride=2 and padding='SAME', output time is ceil(20/2) = 10
        assert y.shape == (4, 10, 64), f"Expected (4, 10, 64), got {y.shape}"

    def test_stride_1(self):
        """Verify stride=1 preserves temporal dimension."""
        block = TemporalConvBlock(out_channels=64, kernel_size=3, stride=1)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((4, 20, 128))

        params = block.init(key, x)
        y = block.apply(params, x)

        assert y.shape == (4, 20, 64), f"Expected (4, 20, 64), got {y.shape}"

    def test_gradient_flow(self):
        """Verify gradients flow through the block."""
        block = TemporalConvBlock(out_channels=64, kernel_size=3, stride=2)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((4, 20, 128))

        params = block.init(key, x)

        def loss_fn(params):
            y = block.apply(params, x)
            return jnp.mean(y**2)

        grad = jax.grad(loss_fn)(params)
        assert grad is not None
        assert "params" in grad


class TestVQTemporalEncoder:
    """Test VQTemporalEncoder shape and behavior."""

    def test_output_shape(self):
        """Verify temporal encoder downsamples correctly."""
        encoder = VQTemporalEncoder(
            hidden_channels=[128, 128],
            latent_dim=16,
            temporal_stride=4,
            kernel_size=3,
        )
        key = jax.random.PRNGKey(0)
        x = jnp.ones((20, 4, 640))  # [time, batch, input_dim]

        params = encoder.init(key, x)
        z_e = encoder.apply(params, x)

        # With temporal_stride=4, output time is 20//4 = 5
        assert z_e.shape == (5, 4, 16), f"Expected (5, 4, 16), got {z_e.shape}"

    def test_different_strides(self):
        """Verify different temporal strides work correctly."""
        for stride in [2, 4, 8]:
            encoder = VQTemporalEncoder(
                hidden_channels=[64, 64],
                latent_dim=8,
                temporal_stride=stride,
            )
            key = jax.random.PRNGKey(0)
            x = jnp.ones((16, 2, 100))

            params = encoder.init(key, x)
            z_e = encoder.apply(params, x)

            expected_t = 16 // stride
            assert z_e.shape[0] == expected_t, (
                f"stride={stride}: expected T={expected_t}, got {z_e.shape[0]}"
            )

    def test_activation_output(self):
        """Verify activation dict is returned correctly."""
        encoder = VQTemporalEncoder(
            hidden_channels=[64, 64],
            latent_dim=8,
            temporal_stride=4,
        )
        key = jax.random.PRNGKey(0)
        x = jnp.ones((20, 4, 100))

        params = encoder.init(key, x, get_activation=True)
        z_e, activations = encoder.apply(params, x, get_activation=True)

        assert "input" in activations
        assert "conv_0" in activations
        assert "conv_1" in activations
        assert "z_e" in activations

    def test_gradient_flow(self):
        """Verify gradients flow through the encoder."""
        encoder = VQTemporalEncoder(
            hidden_channels=[64, 64],
            latent_dim=8,
            temporal_stride=4,
        )
        key = jax.random.PRNGKey(0)
        x = jnp.ones((20, 4, 100))

        params = encoder.init(key, x)

        def loss_fn(params):
            z_e = encoder.apply(params, x)
            return jnp.mean(z_e**2)

        grad = jax.grad(loss_fn)(params)
        assert grad is not None


class TestUpsampleTemporal:
    """Test upsample_temporal utility function."""

    def test_basic_upsample(self):
        """Verify basic upsampling works correctly."""
        z_q = jnp.ones((5, 4, 16))  # [T_down, B, D]
        target_length = 20

        z_q_up = upsample_temporal(z_q, target_length)

        assert z_q_up.shape == (20, 4, 16), f"Expected (20, 4, 16), got {z_q_up.shape}"

    def test_values_repeated(self):
        """Verify values are correctly repeated."""
        # Create distinct values for each timestep
        z_q = jnp.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]])  # [5, 1, 1]
        target_length = 20

        z_q_up = upsample_temporal(z_q, target_length)

        # Each original value should be repeated 4 times
        expected = jnp.array([
            [[1.0]], [[1.0]], [[1.0]], [[1.0]],
            [[2.0]], [[2.0]], [[2.0]], [[2.0]],
            [[3.0]], [[3.0]], [[3.0]], [[3.0]],
            [[4.0]], [[4.0]], [[4.0]], [[4.0]],
            [[5.0]], [[5.0]], [[5.0]], [[5.0]],
        ])
        assert jnp.allclose(z_q_up, expected), f"Values not repeated correctly"

    def test_no_upsample_needed(self):
        """Verify no change when lengths match."""
        z_q = jnp.ones((20, 4, 16))
        z_q_up = upsample_temporal(z_q, 20)
        assert jnp.allclose(z_q, z_q_up)

    def test_non_divisible_length(self):
        """Verify handling of non-divisible target lengths."""
        z_q = jnp.ones((3, 2, 8))  # [3, 2, 8]
        target_length = 10  # Not divisible by 3

        z_q_up = upsample_temporal(z_q, target_length)

        assert z_q_up.shape == (10, 2, 8), f"Expected (10, 2, 8), got {z_q_up.shape}"


class TestVQTemporalIntentionNetwork:
    """Test full VQ-VAE temporal intention network."""

    def test_output_shapes(self):
        """Verify full network output shapes with temporal downsampling."""
        network = VQTemporalIntentionNetwork(
            encoder_hidden_channels=[64, 64],
            decoder_layers=[128, 64, 32],
            latent_dim=16,
            temporal_stride=4,
            num_codes=8,
        )
        key = jax.random.PRNGKey(0)
        obs = {
            "imitation_target": jnp.ones((20, 4, 100)),
            "proprioception": jnp.ones((20, 4, 50)),
        }

        params = network.init(key, obs, key)
        action, z_e, indices = network.apply(params, obs, key)

        # Action should match original temporal resolution
        assert action.shape == (20, 4, 32), f"action shape: {action.shape}"
        # z_e should be downsampled
        assert z_e.shape == (5, 4, 16), f"z_e shape: {z_e.shape}"
        # indices should be downsampled
        assert indices.shape == (5, 4), f"indices shape: {indices.shape}"

    def test_get_activation(self):
        """Verify activations include upsampled z_q."""
        network = VQTemporalIntentionNetwork(
            encoder_hidden_channels=[64, 64],
            decoder_layers=[128, 64, 32],
            latent_dim=16,
            temporal_stride=4,
            num_codes=8,
        )
        key = jax.random.PRNGKey(0)
        obs = {
            "imitation_target": jnp.ones((20, 4, 100)),
            "proprioception": jnp.ones((20, 4, 50)),
        }

        params = network.init(key, obs, key, get_activation=True)
        action, z_e, indices, extras = network.apply(
            params, obs, key, get_activation=True
        )

        assert "z_q_upsampled" in extras
        assert extras["z_q_upsampled"].shape == (20, 4, 16)
        assert extras["temporal_stride"] == 4

    def test_end_to_end_gradient(self):
        """Verify gradients flow through entire temporal network."""
        network = VQTemporalIntentionNetwork(
            encoder_hidden_channels=[64, 64],
            decoder_layers=[128, 64, 32],
            latent_dim=16,
            temporal_stride=4,
            num_codes=8,
        )
        key = jax.random.PRNGKey(0)
        obs = {
            "imitation_target": jnp.ones((20, 4, 100)),
            "proprioception": jnp.ones((20, 4, 50)),
        }

        params = network.init(key, obs, key)

        def loss_fn(params):
            action, z_e, indices = network.apply(params, obs, key)
            return jnp.mean(action**2)

        grad = jax.grad(loss_fn)(params)

        # Check encoder gradients exist
        encoder_grad = grad["params"]["encoder"]
        assert encoder_grad is not None
        # Check decoder gradients exist
        decoder_grad = grad["params"]["decoder"]
        assert decoder_grad is not None
        # Check codebook gradients exist (via straight-through)
        codebook_grad = grad["params"]["quantizer"]["embeddings"]
        assert codebook_grad is not None

    def test_single_step_inference(self):
        """Verify network handles single-step inference (no time dimension)."""
        network = VQTemporalIntentionNetwork(
            encoder_hidden_channels=[64, 64],
            decoder_layers=[128, 64, 32],
            latent_dim=16,
            temporal_stride=4,
            num_codes=8,
        )
        key = jax.random.PRNGKey(0)

        # Initialize with batched input (has time dimension)
        batch_obs = {
            "imitation_target": jnp.ones((20, 4, 100)),
            "proprioception": jnp.ones((20, 4, 50)),
        }
        params = network.init(key, batch_obs, key)

        # Test single-step inference with batch dimension [B, D]
        single_obs = {
            "imitation_target": jnp.ones((4, 100)),  # [B, D] instead of [T, B, D]
            "proprioception": jnp.ones((4, 50)),
        }
        action, z_e, indices = network.apply(params, single_obs, key)

        # For single-step: action should be [B, action_size]
        assert action.shape == (4, 32), f"action shape: {action.shape}"
        # z_e should be [B, latent_dim]
        assert z_e.shape == (4, 16), f"z_e shape: {z_e.shape}"
        # indices should be [B]
        assert indices.shape == (4,), f"indices shape: {indices.shape}"

    def test_single_observation_inference(self):
        """Verify network handles single observation (no batch, no time)."""
        network = VQTemporalIntentionNetwork(
            encoder_hidden_channels=[64, 64],
            decoder_layers=[128, 64, 32],
            latent_dim=16,
            temporal_stride=4,
            num_codes=8,
        )
        key = jax.random.PRNGKey(0)

        # Initialize with batched input (has time dimension)
        batch_obs = {
            "imitation_target": jnp.ones((20, 4, 100)),
            "proprioception": jnp.ones((20, 4, 50)),
        }
        params = network.init(key, batch_obs, key)

        # Test single observation [D] (used in logging rollouts)
        single_obs = {
            "imitation_target": jnp.ones((100,)),  # [D] only
            "proprioception": jnp.ones((50,)),
        }
        action, z_e, indices = network.apply(params, single_obs, key)

        # For single observation: all outputs should be scalars or 1D
        assert action.shape == (32,), f"action shape: {action.shape}"
        assert z_e.shape == (16,), f"z_e shape: {z_e.shape}"
        assert indices.shape == (), f"indices shape: {indices.shape}"


class TestVQLoss:
    """Test VQ-VAE loss computation."""

    def test_vq_loss_shapes(self):
        """Verify VQ loss returns correct shapes."""
        z_e = jnp.ones((10, 4, 64))
        z_q = jnp.ones((10, 4, 64)) * 0.5

        vq_loss, commitment_loss, codebook_loss = compute_vq_loss(z_e, z_q)

        assert vq_loss.shape == ()
        assert commitment_loss.shape == ()
        assert codebook_loss.shape == ()

    def test_commitment_loss_gradient_routing(self):
        """Verify commitment loss gradients go to encoder only."""
        z_e = jnp.ones((4, 64))
        z_q = jnp.ones((4, 64)) * 0.5

        # Commitment loss: beta * ||z_e - sg(z_q)||^2
        def commitment_fn(z_e_in, z_q_in):
            return jnp.mean((z_e_in - jax.lax.stop_gradient(z_q_in)) ** 2)

        grad_z_e, grad_z_q = jax.grad(commitment_fn, argnums=(0, 1))(z_e, z_q)
        assert not jnp.allclose(grad_z_e, 0.0), "z_e should have gradients"
        assert jnp.allclose(grad_z_q, 0.0), "z_q should NOT have gradients"

    def test_codebook_loss_gradient_routing(self):
        """Verify codebook loss gradients go to codebook only."""
        z_e = jnp.ones((4, 64))
        z_q = jnp.ones((4, 64)) * 0.5

        # Codebook loss: ||sg(z_e) - z_q||^2
        def codebook_fn(z_e_in, z_q_in):
            return jnp.mean((jax.lax.stop_gradient(z_e_in) - z_q_in) ** 2)

        grad_z_e, grad_z_q = jax.grad(codebook_fn, argnums=(0, 1))(z_e, z_q)
        assert jnp.allclose(grad_z_e, 0.0), "z_e should NOT have gradients"
        assert not jnp.allclose(grad_z_q, 0.0), "z_q should have gradients"


class TestCEStickinessCost:
    """Test cross-entropy stickiness cost computation."""

    def test_output_shape(self):
        """Verify CE stickiness loss returns scalar."""
        T, B, D, K = 10, 4, 64, 8
        z_e = jnp.ones((T, B, D))
        indices = jax.random.randint(jax.random.PRNGKey(0), (T, B), 0, K)
        codebook = jnp.ones((K, D))
        valid_mask = jnp.ones((T - 1, B))

        loss, metrics = compute_ce_stickiness_cost(
            z_e=z_e,
            indices=indices,
            codebook=codebook,
            valid_mask=valid_mask,
            temperature=1.0,
        )

        assert loss.shape == (), f"Expected scalar, got {loss.shape}"
        assert "ce_stickiness_loss" in metrics
        assert "prob_of_prev_code" in metrics

    def test_same_code_low_loss(self):
        """When z_e is always close to same code, loss should be low."""
        T, B, D, K = 10, 4, 64, 8
        key = jax.random.PRNGKey(42)

        # Create codebook with well-separated codes
        codebook = jnp.eye(K, D) * 10.0  # Each code is far from others

        # All timesteps have z_e very close to code 0
        z_e = jnp.broadcast_to(codebook[0:1] + 0.01, (T, B, D))
        indices = jnp.zeros((T, B), dtype=jnp.int32)  # All code 0
        valid_mask = jnp.ones((T - 1, B))

        loss, metrics = compute_ce_stickiness_cost(
            z_e=z_e,
            indices=indices,
            codebook=codebook,
            valid_mask=valid_mask,
            temperature=1.0,
        )

        # Loss should be low (high prob of staying with same code)
        assert loss < 1.0, f"Expected low loss when staying at same code, got {loss}"
        # prob_of_prev_code should be high
        assert (
            metrics["prob_of_prev_code"] > 0.5
        ), f"Expected high prob, got {metrics['prob_of_prev_code']}"

    def test_code_switch_high_loss(self):
        """When z_e switches to different code, loss should be high."""
        T, B, D, K = 10, 4, 64, 8

        # Create codebook with well-separated codes
        codebook = jnp.eye(K, D) * 10.0

        # Create z_e that switches from code 0 to code 1
        z_e_list = []
        for t in range(T):
            if t < T // 2:
                z_e_list.append(jnp.broadcast_to(codebook[0:1] + 0.01, (1, B, D)))
            else:
                z_e_list.append(jnp.broadcast_to(codebook[1:2] + 0.01, (1, B, D)))
        z_e = jnp.concatenate(z_e_list, axis=0)

        # Indices reflect that prev timestep was code 0 until switch
        indices = jnp.zeros((T, B), dtype=jnp.int32)
        indices = indices.at[T // 2 :].set(1)

        valid_mask = jnp.ones((T - 1, B))

        loss, metrics = compute_ce_stickiness_cost(
            z_e=z_e,
            indices=indices,
            codebook=codebook,
            valid_mask=valid_mask,
            temperature=1.0,
        )

        # Loss at the switch point should be high
        # Overall loss is averaged, so it will be moderate
        assert loss > 0.1, f"Expected non-trivial loss when code switches, got {loss}"

    def test_temperature_effect(self):
        """Lower temperature should give sharper probabilities."""
        T, B, D, K = 10, 4, 64, 8

        codebook = jnp.eye(K, D) * 5.0
        z_e = jnp.broadcast_to(
            codebook[0:1] + 0.5, (T, B, D)
        )  # Slightly off from code 0
        indices = jnp.zeros((T, B), dtype=jnp.int32)
        valid_mask = jnp.ones((T - 1, B))

        # Low temperature
        loss_low_temp, metrics_low = compute_ce_stickiness_cost(
            z_e=z_e,
            indices=indices,
            codebook=codebook,
            valid_mask=valid_mask,
            temperature=0.1,
        )

        # High temperature
        loss_high_temp, metrics_high = compute_ce_stickiness_cost(
            z_e=z_e,
            indices=indices,
            codebook=codebook,
            valid_mask=valid_mask,
            temperature=10.0,
        )

        # Low temperature should have higher prob of correct code
        assert (
            metrics_low["prob_of_prev_code"] > metrics_high["prob_of_prev_code"]
        ), "Lower temp should give sharper probs"

    def test_gradient_flows_to_encoder(self):
        """Verify gradients flow to encoder z_e."""
        T, B, D, K = 5, 2, 16, 4
        codebook = jnp.eye(K, D) * 2.0
        z_e = jnp.ones((T, B, D))
        indices = jnp.zeros((T, B), dtype=jnp.int32)
        valid_mask = jnp.ones((T - 1, B))

        def loss_fn(z_e_input):
            loss, _ = compute_ce_stickiness_cost(
                z_e=z_e_input,
                indices=indices,
                codebook=codebook,
                valid_mask=valid_mask,
                temperature=1.0,
            )
            return loss

        grad = jax.grad(loss_fn)(z_e)
        assert grad is not None
        assert grad.shape == z_e.shape
        assert not jnp.allclose(grad, 0.0), "Gradients should not be zero"

    def test_codebook_gradient_stopped(self):
        """Verify gradients do NOT flow to codebook (stop_gradient)."""
        T, B, D, K = 5, 2, 16, 4
        z_e = jnp.ones((T, B, D))
        indices = jnp.zeros((T, B), dtype=jnp.int32)
        valid_mask = jnp.ones((T - 1, B))

        def loss_fn(codebook_input):
            loss, _ = compute_ce_stickiness_cost(
                z_e=z_e,
                indices=indices,
                codebook=codebook_input,
                valid_mask=valid_mask,
                temperature=1.0,
            )
            return loss

        codebook = jnp.eye(K, D) * 2.0
        grad = jax.grad(loss_fn)(codebook)
        assert jnp.allclose(
            grad, 0.0
        ), "Codebook should NOT have gradients (stop_gradient)"

    def test_masking_episode_boundaries(self):
        """Verify masking correctly excludes episode boundaries."""
        T, B, D, K = 10, 4, 64, 8
        codebook = jnp.eye(K, D) * 10.0

        # z_e switches codes to create high loss transitions
        z_e_list = []
        for t in range(T):
            code_idx = t % K  # Cycle through codes
            z_e_list.append(
                jnp.broadcast_to(codebook[code_idx : code_idx + 1], (1, B, D))
            )
        z_e = jnp.concatenate(z_e_list, axis=0)
        indices = jnp.array(
            [[t % K for _ in range(B)] for t in range(T)], dtype=jnp.int32
        )

        # Mask out all transitions (pretend all are episode boundaries)
        valid_mask_none = jnp.zeros((T - 1, B))
        loss_none, _ = compute_ce_stickiness_cost(
            z_e=z_e,
            indices=indices,
            codebook=codebook,
            valid_mask=valid_mask_none,
            temperature=1.0,
        )

        # All transitions valid
        valid_mask_all = jnp.ones((T - 1, B))
        loss_all, _ = compute_ce_stickiness_cost(
            z_e=z_e,
            indices=indices,
            codebook=codebook,
            valid_mask=valid_mask_all,
            temperature=1.0,
        )

        # With no valid transitions, loss should be essentially zero
        # (numerator is 0, denominator is 1e-8, so result is 0)
        assert jnp.isclose(
            loss_none, 0.0, atol=1e-6
        ), f"Masked loss should be zero, got {loss_none}"
        # With all valid, loss should be positive (code switching creates loss)
        assert loss_all > 0.1, f"Unmasked loss should be positive, got {loss_all}"


class TestCodebookMetrics:
    """Test codebook health metrics."""

    def test_perplexity_bounds(self):
        """Verify perplexity is in valid range."""
        num_codes = 512
        # All same code -> low perplexity
        indices_collapsed = jnp.zeros((100,), dtype=jnp.int32)
        perp_low, util_low, _ = compute_codebook_metrics(indices_collapsed, num_codes)
        assert perp_low >= 1.0
        assert perp_low < 10.0  # Should be close to 1

        # Uniform random codes -> high perplexity
        key = jax.random.PRNGKey(0)
        indices_uniform = jax.random.randint(key, (10000,), 0, num_codes)
        perp_high, util_high, _ = compute_codebook_metrics(indices_uniform, num_codes)
        assert perp_high > 100.0  # Should be close to num_codes
        assert perp_high <= num_codes

    def test_utilization_bounds(self):
        """Verify utilization is in [0, 1]."""
        num_codes = 512
        indices = jnp.array([0, 1, 2, 3, 4])
        _, utilization, codes_used = compute_codebook_metrics(indices, num_codes)

        assert 0.0 <= utilization <= 1.0
        assert codes_used == 5


class TestMakeVQIntentionPolicy:
    """Test the factory function."""

    def test_factory_creates_valid_network(self):
        """Verify factory creates working network."""
        policy = make_vq_intention_policy(
            action_param_size=32,
            latent_dim=64,
            total_obs_size=100,
            reference_obs_size=50,
            encoder_hidden_layer_sizes=(256, 128),
            decoder_hidden_layer_sizes=(256, 128),
            num_codes=512,
        )

        key = jax.random.PRNGKey(0)
        params = policy.init(key)

        # Verify init creates params
        assert "params" in params

        # Verify apply works
        obs = jnp.ones((4, 100))
        action, z_e, indices = policy.apply(None, params, obs, key)

        assert action.shape == (4, 32)
        assert z_e.shape == (4, 64)
        assert indices.shape == (4,)


class TestMakeVQIntentionPPONetworks:
    """Test the PPO networks factory."""

    def test_factory_creates_all_components(self):
        """Verify factory creates policy, value, and distribution."""
        networks = make_vq_intention_ppo_networks(
            observation_size=100,
            reference_obs_size=50,
            action_size=8,
            latent_dim=64,
            num_codes=512,
            encoder_hidden_layer_sizes=(256, 128),
            decoder_hidden_layer_sizes=(256, 128),
            value_hidden_layer_sizes=(256, 128),
        )

        assert networks.policy_network is not None
        assert networks.value_network is not None
        assert networks.parametric_action_distribution is not None
        assert networks.num_codes == 512
        assert networks.latent_dim == 64

    def test_inference_fn_works(self):
        """Verify inference function produces valid outputs."""
        networks = make_vq_intention_ppo_networks(
            observation_size=100,
            reference_obs_size=50,
            action_size=8,
            latent_dim=64,
            num_codes=512,
            encoder_hidden_layer_sizes=(256, 128),
            decoder_hidden_layer_sizes=(256, 128),
            value_hidden_layer_sizes=(256, 128),
        )

        key = jax.random.PRNGKey(0)
        policy_params = networks.policy_network.init(key)

        make_policy = make_vq_inference_fn(networks)
        policy_fn = make_policy((None, policy_params), deterministic=False)

        obs = jnp.ones((4, 100))
        action, extras = policy_fn(obs, key)

        assert action.shape == (4, 8)
        assert "z_e" in extras
        assert "indices" in extras
        assert "log_prob" in extras


def run_tests():
    """Run all tests and report results."""
    print("Running VQ-VAE Network Tests...")
    print("=" * 60)

    test_classes = [
        TestVQEncoder,
        TestVectorQuantizer,
        TestDecoder,
        TestVQIntentionNetwork,
        TestTemporalConvBlock,
        TestVQTemporalEncoder,
        TestUpsampleTemporal,
        TestVQTemporalIntentionNetwork,
        TestVQLoss,
        TestCEStickinessCost,
        TestCodebookMetrics,
        TestMakeVQIntentionPolicy,
        TestMakeVQIntentionPPONetworks,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  [PASS] {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  [FAIL] {method_name}: {e}")
                    failed_tests.append((test_class.__name__, method_name, str(e)))

    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")

    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        return False
    else:
        print("\nAll tests passed!")
        return True


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
