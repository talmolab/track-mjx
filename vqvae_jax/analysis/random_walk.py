"""Random walk generation for VQ-VAE motion synthesis.

Generates motion by:
1. Random walk on learned transition probabilities to sample code sequence
2. Free-running decoder execution with real proprioceptive feedback

Prerequisites:
    Run the analysis pipeline first to build transition matrices:
    python -m analysis.analyze --mode transitions

Usage:
    # Generate motion
    python -m analysis.random_walk generate

    # Generate with different temperature
    python -m analysis.random_walk generate --temperature 0.5

    # Compare strategies
    python -m analysis.random_walk compare
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from brax.training import distribution
from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf


# =============================================================================
# CONFIGURATION
# =============================================================================


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_default_config_path() -> Path:
    """Get default config path."""
    return Path(__file__).parent.parent / "configs" / "random_walk_config.yaml"


@dataclass
class GenerationResult:
    """Result of two-stage generation."""

    states: list[Any]
    actions: np.ndarray
    rewards: np.ndarray
    abstract_codes: list[int]
    frame_codes: np.ndarray
    codes_used: np.ndarray
    survival_steps: int
    terminated: bool
    fallen: bool
    total_reward: float
    code_switches: int
    mean_hold_duration: float


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================


def create_environment(cfg: DictConfig) -> Any:
    """Create VNL imitation environment from config."""
    from vnl_playground.tasks.rodent import imitation
    from vnl_playground.tasks.rodent import wrappers as vnl_wrappers

    env_cfg = cfg.env_config
    env_cfg_ml = config_dict.ConfigDict(OmegaConf.to_container(env_cfg, resolve=True))
    return vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=env_cfg_ml))


# =============================================================================
# CODE SEQUENCE SAMPLING (STAGE 1)
# =============================================================================


def get_initial_code_from_clip(
    env: Any,
    inference_fn: Callable,
    clip_idx: int = 0,
    seed: int = 42,
) -> tuple[int, Any]:
    """Get initial code by encoding first frame of a reference clip."""
    jit_reset = jax.jit(env.reset)
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    action_rng = jax.random.PRNGKey(seed + 1)
    _, extras = inference_fn(state.obs, action_rng)
    initial_code = int(extras["indices"])

    return initial_code, state


def sample_code_sequence(
    trans_probs: np.ndarray,
    start_code: int,
    num_codes: int,
    strategy: Literal["temperature", "nucleus", "greedy"] = "temperature",
    temperature: float = 1.0,
    top_p: float = 0.9,
    seed: int = 42,
) -> list[int]:
    """Sample abstract code sequence using learned transition probabilities.

    Performs a random walk on the transition graph.

    Args:
        trans_probs: Transition probability matrix [num_codes, num_codes].
        start_code: Starting code index.
        num_codes: Number of codes to sample.
        strategy: Sampling strategy (temperature, nucleus, greedy).
        temperature: Temperature for temperature sampling.
        top_p: Cumulative probability threshold for nucleus sampling.
        seed: Random seed.

    Returns:
        List of code indices.
    """
    rng = np.random.default_rng(seed)
    num_codes_total = trans_probs.shape[0]

    marginal = trans_probs.sum(axis=1)
    marginal = marginal / (marginal.sum() + 1e-10)

    sequence = [start_code]
    current = start_code

    for _ in range(num_codes - 1):
        probs = trans_probs[current].copy()

        if probs.sum() < 1e-6:
            probs = marginal

        if strategy == "temperature":
            if temperature < 0.01:
                next_code = int(np.argmax(probs))
            else:
                log_probs = np.log(probs + 1e-10)
                scaled = log_probs / temperature
                probs = np.exp(scaled - np.max(scaled))
                probs = probs / probs.sum()
                next_code = int(rng.choice(num_codes_total, p=probs))

        elif strategy == "nucleus":
            sorted_idx = np.argsort(-probs)
            sorted_probs = probs[sorted_idx]
            cumsum = np.cumsum(sorted_probs)
            cutoff = min(np.searchsorted(cumsum, top_p) + 1, len(sorted_idx))
            selected_idx = sorted_idx[:cutoff]
            selected_probs = probs[selected_idx]
            selected_probs = selected_probs / selected_probs.sum()
            next_code = int(rng.choice(selected_idx, p=selected_probs))

        elif strategy == "greedy":
            next_code = int(np.argmax(probs))

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        sequence.append(next_code)
        current = next_code

    return sequence


def expand_code_sequence_with_holds(
    code_sequence: list[int],
    trans_probs: np.ndarray,
    total_frames: int,
    min_hold: int = 1,
    max_hold: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Expand abstract code sequence to frame-level with hold durations.

    Uses self-transition probabilities to determine hold duration.

    Args:
        code_sequence: Abstract code sequence.
        trans_probs: Transition probability matrix.
        total_frames: Target number of frames.
        min_hold: Minimum hold duration per code.
        max_hold: Maximum hold duration per code.
        seed: Random seed.

    Returns:
        Frame-level code array [total_frames].
    """
    rng = np.random.default_rng(seed)
    expanded = []
    seq_idx = 0

    while len(expanded) < total_frames and seq_idx < len(code_sequence):
        code = code_sequence[seq_idx]
        self_prob = trans_probs[code, code]

        if self_prob > 0.1:
            hold_time = int(rng.geometric(1 - self_prob + 1e-6))
            hold_time = max(min_hold, min(hold_time, max_hold))
            hold_time = min(hold_time, total_frames - len(expanded))
        else:
            hold_time = min_hold

        expanded.extend([code] * hold_time)
        seq_idx += 1

    while len(expanded) < total_frames:
        expanded.append(code_sequence[-1])

    return np.array(expanded[:total_frames], dtype=np.int32)


# =============================================================================
# AUTOREGRESSIVE DECODING (STAGE 2)
# =============================================================================


def run_autoregressive_decoding(
    env: Any,
    decoder_apply_fn: Callable,
    codebook: np.ndarray,
    frame_codes: np.ndarray,
    action_distribution: Any,
    initial_state: Any | None = None,
    proprio_size: int = 264,
    seed: int = 42,
    torso_z_threshold: float = 0.03,
) -> GenerationResult:
    """Execute free-running generation with pre-planned code sequence.

    Uses the decoder with real proprioceptive feedback from simulation.

    Args:
        env: MuJoCo environment.
        decoder_apply_fn: Function (z_q, proprio) -> action_params.
        codebook: Codebook embeddings [num_codes, latent_dim].
        frame_codes: Per-frame code indices [horizon].
        action_distribution: Action distribution for sampling.
        initial_state: Optional initial state.
        proprio_size: Size of proprioceptive observation.
        seed: Random seed.
        torso_z_threshold: Z threshold for fall detection.

    Returns:
        GenerationResult with trajectory data.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_decoder = jax.jit(decoder_apply_fn)

    rng = jax.random.PRNGKey(seed)

    if initial_state is None:
        reset_rng, rng = jax.random.split(rng)
        state = jit_reset(reset_rng)
    else:
        state = initial_state

    states = [state]
    actions = []
    rewards = []
    codes_used = []

    horizon = len(frame_codes)
    terminated = False
    fallen = False
    code_switches = 0
    prev_code = None

    for step in range(horizon):
        if step % 100 == 0:
            logging.debug(f"  Decoding step {step}/{horizon}")

        code_idx = int(frame_codes[step])
        codes_used.append(code_idx)

        if prev_code is not None and code_idx != prev_code:
            code_switches += 1
        prev_code = code_idx

        z_q = codebook[code_idx]
        proprio_obs = state.obs[..., -proprio_size:]
        action_params = jit_decoder(z_q, proprio_obs)
        action = action_distribution.mode(action_params)

        state = jit_step(state, action)
        states.append(state)
        actions.append(np.array(action))
        rewards.append(float(state.reward))

        if state.done:
            terminated = True
            break

        qpos = np.array(state.data.qpos)
        if qpos[2] < torso_z_threshold:
            fallen = True

    survival_steps = len(states) - 1
    total_reward = float(np.sum(rewards))

    codes_array = np.array(codes_used)
    changes = np.sum(np.diff(codes_array) != 0)
    mean_hold = len(codes_used) / (changes + 1) if len(codes_used) > 0 else 0

    return GenerationResult(
        states=states,
        actions=np.array(actions) if actions else np.array([]),
        rewards=np.array(rewards),
        abstract_codes=[],
        frame_codes=frame_codes,
        codes_used=np.array(codes_used),
        survival_steps=survival_steps,
        terminated=terminated,
        fallen=fallen,
        total_reward=total_reward,
        code_switches=code_switches,
        mean_hold_duration=mean_hold,
    )


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================


def run_random_walk_generation(config: dict) -> dict[str, Any]:
    """Run two-stage conditional generation.

    Stage 1: Sample code sequence using transition probabilities
    Stage 2: Free-running decoder execution

    Args:
        config: Full configuration dict.

    Returns:
        Dictionary with generation results.
    """
    from analysis.checkpoint_utils import (
        create_decoder_apply_fn,
        get_codebook,
        load_vq_checkpoint,
        load_vq_inference_fn,
    )
    from analysis.rendering import render_rollout_to_video

    output_dir = Path(config["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logging.info(f"Loading checkpoint from {config['checkpoint']['path']}")
    ckpt = load_vq_checkpoint(
        config["checkpoint"]["path"],
        step=config["checkpoint"]["step"],
    )
    cfg = ckpt["cfg"]
    policy_params = ckpt["policy"]

    # Load transition probabilities
    trans_dir = Path(config["transitions"]["dir"])
    logging.info(f"Loading transition probabilities from {trans_dir}")
    trans_probs = np.load(trans_dir / "transition_probs.npy")

    # Setup
    codebook = np.array(get_codebook(policy_params))
    decoder_apply_fn = create_decoder_apply_fn(cfg, policy_params)

    num_codes = cfg.network_config.num_codes
    action_size = cfg.network_config.action_size
    proprio_size = cfg.network_config.proprioceptive_obs_size

    action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    env = create_environment(cfg)
    inference_fn = load_vq_inference_fn(cfg, policy_params, deterministic=True)

    sampling_cfg = config["sampling"]
    holds_cfg = config["holds"]
    gen_cfg = config["generation"]

    # Stage 1: Get initial code and sample sequence
    logging.info("=" * 60)
    logging.info("STAGE 1: Sampling code sequence")

    initial_code, initial_state = get_initial_code_from_clip(
        env=env,
        inference_fn=inference_fn,
        clip_idx=sampling_cfg["initial_clip_idx"],
        seed=sampling_cfg["seed"],
    )
    logging.info(f"  Initial code: {initial_code}")

    horizon = gen_cfg["horizon"]
    abstract_length = max(50, horizon // 5)

    abstract_codes = sample_code_sequence(
        trans_probs=trans_probs,
        start_code=initial_code,
        num_codes=abstract_length,
        strategy=sampling_cfg["strategy"],
        temperature=sampling_cfg["temperature"],
        top_p=sampling_cfg["top_p"],
        seed=sampling_cfg["seed"],
    )
    logging.info(f"  Sampled {len(abstract_codes)} abstract codes")
    logging.info(f"  First 10: {abstract_codes[:10]}")

    frame_codes = expand_code_sequence_with_holds(
        code_sequence=abstract_codes,
        trans_probs=trans_probs,
        total_frames=horizon,
        min_hold=holds_cfg["min_duration"],
        max_hold=holds_cfg["max_duration"],
        seed=sampling_cfg["seed"] + 1,
    )
    logging.info(f"  Expanded to {len(frame_codes)} frame codes")

    # Stage 2: Autoregressive decoding
    logging.info("=" * 60)
    logging.info("STAGE 2: Autoregressive decoding")

    result = run_autoregressive_decoding(
        env=env,
        decoder_apply_fn=decoder_apply_fn,
        codebook=codebook,
        frame_codes=frame_codes,
        action_distribution=action_distribution,
        initial_state=initial_state,
        proprio_size=proprio_size,
        seed=sampling_cfg["seed"] + 2,
        torso_z_threshold=gen_cfg["torso_z_threshold"],
    )
    result.abstract_codes = abstract_codes

    logging.info("=" * 60)
    logging.info("GENERATION COMPLETE")
    logging.info(f"  Survival: {result.survival_steps}/{horizon}")
    logging.info(f"  Terminated: {result.terminated}")
    logging.info(f"  Fallen: {result.fallen}")
    logging.info(f"  Total reward: {result.total_reward:.2f}")
    logging.info(f"  Code switches: {result.code_switches}")
    logging.info(f"  Mean hold: {result.mean_hold_duration:.1f}")

    # Save metrics
    metrics = {
        "config": {
            "strategy": sampling_cfg["strategy"],
            "temperature": sampling_cfg["temperature"],
            "horizon": horizon,
            "seed": sampling_cfg["seed"],
        },
        "abstract_codes": abstract_codes,
        "frame_codes": frame_codes.tolist(),
        "codes_used": result.codes_used.tolist(),
        "survival_steps": result.survival_steps,
        "terminated": result.terminated,
        "fallen": result.fallen,
        "total_reward": result.total_reward,
        "code_switches": result.code_switches,
        "mean_hold_duration": result.mean_hold_duration,
        "rewards": result.rewards.tolist(),
    }

    with open(output_dir / "generation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Render video
    if config["render"]["enabled"]:
        logging.info("Rendering video...")
        render_cfg = config["render"]
        video_path = output_dir / "random_walk_generation.mp4"

        render_rollout_to_video(
            env=env,
            rollout_states=result.states,
            output_path=video_path,
            camera=render_cfg["camera"],
            width=render_cfg["width"],
            height=render_cfg["height"],
            fps=render_cfg["fps"],
            indices=result.codes_used,
            num_codes=num_codes,
            rewards=result.rewards,
            code_bar_height=render_cfg["code_bar_height"],
        )
        metrics["video_path"] = str(video_path)

    return metrics


def compare_strategies(config: dict) -> dict[str, Any]:
    """Compare different generation strategies.

    Args:
        config: Full configuration dict.

    Returns:
        Comparison results.
    """
    from analysis.checkpoint_utils import (
        create_decoder_apply_fn,
        get_codebook,
        load_vq_checkpoint,
        load_vq_inference_fn,
    )

    output_dir = Path(config["output"]["base_dir"]) / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = load_vq_checkpoint(
        config["checkpoint"]["path"],
        step=config["checkpoint"]["step"],
    )
    cfg = ckpt["cfg"]
    policy_params = ckpt["policy"]

    # Load transitions
    trans_probs = np.load(Path(config["transitions"]["dir"]) / "transition_probs.npy")

    # Setup
    codebook = np.array(get_codebook(policy_params))
    decoder_apply_fn = create_decoder_apply_fn(cfg, policy_params)

    action_size = cfg.network_config.action_size
    proprio_size = cfg.network_config.proprioceptive_obs_size
    action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    env = create_environment(cfg)
    inference_fn = load_vq_inference_fn(cfg, policy_params, deterministic=True)

    comp_cfg = config["comparison"]
    num_trials = comp_cfg["num_trials"]
    horizon = comp_cfg["horizon"]
    strategies = comp_cfg["strategies"]

    results = {}

    for strategy in strategies:
        name = strategy["name"]
        logging.info(f"Running strategy: {name}")

        strategy_results = {
            "survival_rates": [],
            "survival_steps": [],
            "total_rewards": [],
            "falls": [],
        }

        for trial in range(num_trials):
            seed = 42 + trial * 100

            initial_code, initial_state = get_initial_code_from_clip(
                env=env,
                inference_fn=inference_fn,
                clip_idx=trial % 10,
                seed=seed,
            )

            abstract_codes = sample_code_sequence(
                trans_probs=trans_probs,
                start_code=initial_code,
                num_codes=max(50, horizon // 5),
                strategy=strategy.get("strategy", "temperature"),
                temperature=strategy.get("temperature", 1.0),
                seed=seed,
            )

            frame_codes = expand_code_sequence_with_holds(
                code_sequence=abstract_codes,
                trans_probs=trans_probs,
                total_frames=horizon,
                seed=seed + 1,
            )

            result = run_autoregressive_decoding(
                env=env,
                decoder_apply_fn=decoder_apply_fn,
                codebook=codebook,
                frame_codes=frame_codes,
                action_distribution=action_distribution,
                initial_state=initial_state,
                proprio_size=proprio_size,
                seed=seed + 2,
            )

            strategy_results["survival_rates"].append(result.survival_steps / horizon)
            strategy_results["survival_steps"].append(result.survival_steps)
            strategy_results["total_rewards"].append(result.total_reward)
            strategy_results["falls"].append(result.fallen)

        results[name] = {
            "mean_survival_rate": float(np.mean(strategy_results["survival_rates"])),
            "std_survival_rate": float(np.std(strategy_results["survival_rates"])),
            "mean_survival_steps": float(np.mean(strategy_results["survival_steps"])),
            "mean_total_reward": float(np.mean(strategy_results["total_rewards"])),
            "fall_rate": float(np.mean(strategy_results["falls"])),
        }

        logging.info(f"  {name}: survival={results[name]['mean_survival_rate']:.1%}")

    # Save results
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump({"num_trials": num_trials, "horizon": horizon, "results": results}, f, indent=2)

    # Print summary
    logging.info("\n" + "=" * 60)
    logging.info("COMPARISON SUMMARY")
    logging.info("=" * 60)
    logging.info(f"{'Strategy':<25} {'Survival':>10} {'Reward':>10} {'Falls':>8}")
    logging.info("-" * 60)
    for name, data in results.items():
        logging.info(f"{name:<25} {data['mean_survival_rate']:>9.1%} "
                     f"{data['mean_total_reward']:>10.1f} "
                     f"{data['fall_rate']:>7.1%}")

    return results


# =============================================================================
# CLI
# =============================================================================


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Two-stage conditional generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m analysis.two_stage_generation generate
    python -m analysis.two_stage_generation generate --temperature 0.5
    python -m analysis.two_stage_generation compare
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(get_default_config_path()),
        help="Path to config YAML file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Run two-stage generation")
    gen_parser.add_argument("--checkpoint", type=str, help="Override checkpoint path")
    gen_parser.add_argument("--strategy", type=str, choices=["temperature", "nucleus", "greedy"])
    gen_parser.add_argument("--temperature", type=float)
    gen_parser.add_argument("--horizon", type=int)
    gen_parser.add_argument("--seed", type=int)
    gen_parser.add_argument("--no-render", action="store_true")

    # Compare command
    comp_parser = subparsers.add_parser("compare", help="Compare strategies")
    comp_parser.add_argument("--checkpoint", type=str)
    comp_parser.add_argument("--num-trials", type=int)
    comp_parser.add_argument("--horizon", type=int)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)

    if args.command == "generate":
        if args.checkpoint:
            config["checkpoint"]["path"] = args.checkpoint
        if args.strategy:
            config["sampling"]["strategy"] = args.strategy
        if args.temperature is not None:
            config["sampling"]["temperature"] = args.temperature
        if args.horizon:
            config["generation"]["horizon"] = args.horizon
        if args.seed:
            config["sampling"]["seed"] = args.seed
        if args.no_render:
            config["render"]["enabled"] = False

        run_random_walk_generation(config)

    elif args.command == "compare":
        if args.checkpoint:
            config["checkpoint"]["path"] = args.checkpoint
        if args.num_trials:
            config["comparison"]["num_trials"] = args.num_trials
        if args.horizon:
            config["comparison"]["horizon"] = args.horizon

        compare_strategies(config)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
