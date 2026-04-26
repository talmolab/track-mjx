"""Verify create_ramp_schedule supports decay (min_value > max_value)."""
import pytest
from track_mjx.agent.ff_ppo.losses import create_ramp_schedule


def test_decay_schedule_linear():
    """Linear schedule with min > max should decay."""
    schedule = create_ramp_schedule(
        min_value=1.0, max_value=0.0, ramp_steps=100, schedule="linear"
    )
    assert float(schedule(0)) == pytest.approx(1.0), "Should start at min_value (high)"
    assert float(schedule(50)) == pytest.approx(0.5), "Should be at midpoint"
    assert float(schedule(100)) == pytest.approx(0.0), "Should reach max_value (low)"
    # Monotonically decreasing
    values = [float(schedule(i)) for i in range(101)]
    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1] - 1e-6, f"Not monotonically decreasing at step {i}"


def test_decay_schedule_with_warmup():
    """Decay should hold at start_weight during warmup period."""
    schedule = create_ramp_schedule(
        min_value=0.5, max_value=0.0, ramp_steps=100, warmup_steps=20, schedule="linear"
    )
    assert float(schedule(0)) == pytest.approx(0.5), "Should be at start weight during warmup"
    assert float(schedule(10)) == pytest.approx(0.5), "Should hold during warmup"
    assert float(schedule(20)) == pytest.approx(0.5), "Should start decay after warmup"
    assert float(schedule(120)) == pytest.approx(0.0), "Should reach end weight"


def test_no_penalty_when_zero():
    """When start_weight=0, schedule should always return 0."""
    schedule = create_ramp_schedule(
        min_value=0.0, max_value=0.0, ramp_steps=100, schedule="linear"
    )
    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(50)) == pytest.approx(0.0)
    assert float(schedule(100)) == pytest.approx(0.0)
