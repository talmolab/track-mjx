"""Tests for learning rate schedule creation."""
import pytest
import jax.numpy as jnp
from track_mjx.agent.ff_ppo.losses import create_lr_schedule


class TestCreateLrScheduleConstant:
    """When schedule='constant', LR never changes."""

    def test_constant_returns_init_value_at_step_0(self):
        schedule = create_lr_schedule(init_value=1e-4, schedule="constant")
        assert float(schedule(0)) == pytest.approx(1e-4)

    def test_constant_returns_init_value_at_any_step(self):
        schedule = create_lr_schedule(init_value=3e-4, schedule="constant")
        assert float(schedule(100_000)) == pytest.approx(3e-4)


class TestCreateLrScheduleLinear:
    """Linear decay from init_value to end_value over total_steps."""

    def test_linear_starts_at_init_value(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=0.0, total_steps=1000, schedule="linear"
        )
        assert float(schedule(0)) == pytest.approx(1e-4)

    def test_linear_ends_at_end_value(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=0.0, total_steps=1000, schedule="linear"
        )
        assert float(schedule(1000)) == pytest.approx(0.0, abs=1e-8)

    def test_linear_midpoint(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=0.0, total_steps=1000, schedule="linear"
        )
        assert float(schedule(500)) == pytest.approx(5e-5, rel=1e-4)

    def test_linear_clamps_after_total_steps(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=1e-5, total_steps=1000, schedule="linear"
        )
        assert float(schedule(2000)) == pytest.approx(1e-5, abs=1e-8)

    def test_linear_monotonically_decreasing(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=0.0, total_steps=100, schedule="linear"
        )
        values = [float(schedule(i)) for i in range(101)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-8


class TestCreateLrScheduleCosine:
    """Cosine decay from init_value to end_value over total_steps."""

    def test_cosine_starts_at_init_value(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=0.0, total_steps=1000, schedule="cosine"
        )
        assert float(schedule(0)) == pytest.approx(1e-4)

    def test_cosine_ends_at_end_value(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=0.0, total_steps=1000, schedule="cosine"
        )
        assert float(schedule(1000)) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_monotonically_decreasing(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=0.0, total_steps=100, schedule="cosine"
        )
        values = [float(schedule(i)) for i in range(101)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-8


class TestCreateLrScheduleWarmup:
    """Warmup: linear ramp from end_value to init_value, then decay."""

    def test_warmup_starts_at_end_value(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=0.0, total_steps=1000,
            warmup_steps=100, schedule="linear",
        )
        assert float(schedule(0)) == pytest.approx(0.0, abs=1e-8)

    def test_warmup_reaches_init_value(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=0.0, total_steps=1000,
            warmup_steps=100, schedule="linear",
        )
        assert float(schedule(100)) == pytest.approx(1e-4, rel=1e-4)

    def test_warmup_then_decays(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=0.0, total_steps=1000,
            warmup_steps=100, schedule="linear",
        )
        val_at_warmup = float(schedule(100))
        val_after = float(schedule(500))
        assert val_after < val_at_warmup

    def test_warmup_cosine(self):
        schedule = create_lr_schedule(
            init_value=1e-4, end_value=0.0, total_steps=1000,
            warmup_steps=200, schedule="cosine",
        )
        assert float(schedule(0)) == pytest.approx(0.0, abs=1e-8)
        assert float(schedule(200)) == pytest.approx(1e-4, rel=1e-4)
        assert float(schedule(1000)) == pytest.approx(0.0, abs=1e-6)


class TestCreateLrScheduleInvalid:
    """Invalid schedule type should raise ValueError."""

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule must be"):
            create_lr_schedule(init_value=1e-4, schedule="exponential")
