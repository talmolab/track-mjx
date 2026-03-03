"""Curriculum training manager for gap-jump task.

Manages progressive training stages to shape learning:
1. Fixed short gap (easy jump)
2. Variable gap distances (calibrated jumping)
3. Add hold phase (vision masking)
4. Full task

Transitions between stages are based on mean reward thresholds
from recent evaluations.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import numpy as np


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    name: str
    gap_distances: tuple[float, ...]
    hold_duration: int
    reward_threshold: float  # Mean reward to advance to next stage
    description: str = ""


# Default curriculum stages
DEFAULT_STAGES = [
    CurriculumStage(
        name="stage1_fixed_easy",
        gap_distances=(0.06,),
        hold_duration=0,
        reward_threshold=5.0,
        description="Fixed short gap, no hold phase",
    ),
    CurriculumStage(
        name="stage2_variable_gap",
        gap_distances=(0.06, 0.08, 0.10, 0.12, 0.14),
        hold_duration=0,
        reward_threshold=4.0,
        description="Variable gap distances, no hold phase",
    ),
    CurriculumStage(
        name="stage3_with_hold",
        gap_distances=(0.06, 0.08, 0.10, 0.12, 0.14),
        hold_duration=50,
        reward_threshold=3.5,
        description="Variable gaps with hold phase",
    ),
    CurriculumStage(
        name="stage4_full_task",
        gap_distances=(0.06, 0.08, 0.10, 0.12, 0.14),
        hold_duration=50,
        reward_threshold=float('inf'),  # Final stage, no advancement
        description="Full task with all features",
    ),
]


class CurriculumManager:
    """Manages curriculum progression for gap-jump training.

    Tracks training progress and determines when to advance to
    the next curriculum stage based on reward thresholds.

    Usage:
        curriculum = CurriculumManager()

        # In training loop / progress callback:
        def progress_fn(num_steps, metrics):
            mean_reward = metrics.get("eval/episode_reward", 0.0)
            advanced = curriculum.update(mean_reward)
            if advanced:
                # Rebuild environment with new config
                new_cfg = curriculum.get_env_config_overrides()
                # ... update env
    """

    def __init__(
        self,
        stages: list[CurriculumStage] = None,
        window_size: int = 5,
    ):
        self.stages = stages or list(DEFAULT_STAGES)
        self.current_stage_idx = 0
        self.window_size = window_size
        self.reward_history = []
        self.stage_history = []  # (step, stage_idx) transitions

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1

    def get_env_config_overrides(self) -> dict:
        """Get environment config overrides for current stage."""
        stage = self.current_stage
        return {
            "gap_distances": stage.gap_distances,
            "hold_duration": stage.hold_duration,
        }

    def update(self, mean_reward: float, step: int = 0) -> bool:
        """Update curriculum based on latest reward.

        Args:
            mean_reward: Mean evaluation reward.
            step: Current training step.

        Returns:
            True if stage advanced, False otherwise.
        """
        self.reward_history.append(mean_reward)

        if self.is_final_stage:
            return False

        # Check if rolling average exceeds threshold
        window = self.reward_history[-self.window_size:]
        if len(window) >= self.window_size:
            rolling_avg = np.mean(window)
            if rolling_avg >= self.current_stage.reward_threshold:
                self.current_stage_idx += 1
                self.stage_history.append((step, self.current_stage_idx))
                self.reward_history = []  # Reset for new stage
                return True
        return False

    def save(self, path: str):
        """Save curriculum state to JSON."""
        state = {
            "current_stage_idx": self.current_stage_idx,
            "reward_history": self.reward_history,
            "stage_history": self.stage_history,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str):
        """Load curriculum state from JSON."""
        with open(path) as f:
            state = json.load(f)
        self.current_stage_idx = state["current_stage_idx"]
        self.reward_history = state["reward_history"]
        self.stage_history = state.get("stage_history", [])

    def __repr__(self):
        stage = self.current_stage
        return (
            f"CurriculumManager(stage={self.current_stage_idx}/{len(self.stages)-1}, "
            f"name='{stage.name}', gaps={stage.gap_distances}, "
            f"hold={stage.hold_duration})"
        )
