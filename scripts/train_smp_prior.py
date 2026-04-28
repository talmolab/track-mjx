"""Train a JAX TinyMDM SMP prior on rodent reference clips."""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import wandb
from vnl_playground.tasks.reference_clips import ReferenceClips

from track_mjx.agent.smp.checkpointing import save_prior
from track_mjx.agent.smp.features import (
    SMPFeatureSpec,
    metadata_from_reference,
    sample_reference_smp_obs,
    validate_reference_metadata,
)
from track_mjx.agent.smp.reward import (
    SMPRewardConfig,
    diff_normalizer_from_losses,
    esm_sds_losses,
)
from track_mjx.agent.smp.tinymdm import (
    EMAState,
    TinyDiTDenoiser,
    TinyMDMConfig,
    denoising_loss,
    init_denoiser_params,
    make_optimizer,
    normalizer_from_samples,
    update_ema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        default="data/rodent/rodent_reference_clips.h5",
        help="Rodent HDF5 reference clip file.",
    )
    parser.add_argument("--output_dir", required=True, help="Prior checkpoint dir.")
    parser.add_argument("--clip_length", type=int, default=250)
    parser.add_argument("--num_history_steps", type=int, default=10)
    parser.add_argument("--mocap_hz", type=float, default=50.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_iterations", type=int, default=200_000)
    parser.add_argument("--num_samples_stat", type=int, default=20_000)
    parser.add_argument("--num_samples_diff", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--output_interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--attention_head_dim", type=int, default=64)
    parser.add_argument("--wandb_project", type=str, default="track-mjx-smp")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = str(Path(args.data_path).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=output_dir.name,
            notes=f"SMP prior on {data_path}",
        )

    key = jax.random.PRNGKey(args.seed)
    spec = SMPFeatureSpec(
        num_history_steps=args.num_history_steps,
        mocap_hz=args.mocap_hz,
    )
    clips = ReferenceClips(data_path, n_frames_per_clip=args.clip_length)
    validate_reference_metadata(clips, key_body_names=spec.key_body_names)

    key, stat_key = jax.random.split(key)
    stat_samples = sample_reference_smp_obs(
        clips, stat_key, num_samples=args.num_samples_stat, spec=spec
    )
    normalizer = normalizer_from_samples(
        stat_samples,
        num_history_steps=spec.num_history_steps,
        std_clip=0.2,
    )

    model_config = TinyMDMConfig(
        input_dim=stat_samples.shape[-1],
        num_history_steps=spec.num_history_steps,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        attention_head_dim=args.attention_head_dim,
    )
    model = TinyDiTDenoiser(model_config)
    key, init_key = jax.random.split(key)
    params = init_denoiser_params(init_key, model_config)
    ema = EMAState(params=params, step=jnp.array(0, dtype=jnp.int32))
    optimizer = make_optimizer(args.learning_rate, args.grad_clip_norm)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, ema, batch, step_key):
        def loss_fn(p):
            return denoising_loss(p, batch, step_key, model_config, model)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema = update_ema(ema, params)
        metrics = {**metrics, "grad_norm": optax.global_norm(grads)}
        return params, opt_state, ema, loss, metrics

    def save_checkpoint(step: int, save_params) -> None:
        nonlocal key
        key, diff_key, reward_key = jax.random.split(key, 3)
        diff_obs = sample_reference_smp_obs(
            clips, diff_key, num_samples=args.num_samples_diff, spec=spec
        )
        losses = esm_sds_losses(
            params=save_params,
            normalizer=normalizer,
            x_obs=diff_obs,
            rng=reward_key,
            model_config=model_config,
            diffusion_steps=SMPRewardConfig().diffusion_steps,
            model=model,
        )
        diff_normalizer = diff_normalizer_from_losses(losses)
        save_prior(
            output_dir / f"step_{step}",
            params=params,
            ema_params=ema.params,
            normalizer=normalizer,
            diff_normalizer=diff_normalizer,
            model_config=model_config,
            feature_spec=spec,
            reward_config=SMPRewardConfig(),
            metadata=metadata_from_reference(clips, spec, data_path),
        )
        save_prior(
            output_dir / "latest",
            params=params,
            ema_params=ema.params,
            normalizer=normalizer,
            diff_normalizer=diff_normalizer,
            model_config=model_config,
            feature_spec=spec,
            reward_config=SMPRewardConfig(),
            metadata=metadata_from_reference(clips, spec, data_path),
        )

    start = time.time()
    for step in range(1, args.num_iterations + 1):
        key, sample_key, step_key = jax.random.split(key, 3)
        batch = sample_reference_smp_obs(
            clips, sample_key, num_samples=args.batch_size, spec=spec
        )
        batch = normalizer.normalize(batch, spec.num_history_steps)
        params, opt_state, ema, loss, metrics = train_step(
            params, opt_state, ema, batch, step_key
        )
        if step == 1 or step % args.log_interval == 0:
            elapsed = time.time() - start
            metrics_log = {k: float(v) for k, v in metrics.items()}
            metrics_log["loss"] = float(loss)
            metrics_log["elapsed_s"] = elapsed
            wandb.log(metrics_log, step=step)
            print(
                f"step={step} loss={metrics_log['loss']:.5f} "
                f"grad_norm={metrics_log['grad_norm']:.3f} elapsed={elapsed:.1f}s",
                flush=True,
            )
        if step % args.output_interval == 0:
            save_checkpoint(step, ema.params)

    save_checkpoint(args.num_iterations, ema.params)
    wandb.finish()


if __name__ == "__main__":
    main()
