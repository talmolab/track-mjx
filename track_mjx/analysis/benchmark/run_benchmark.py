"""Sweep num_envs, time the four components, write flybody-style table + raw results.

Usage:
    python -m track_mjx.analysis.benchmark.run_benchmark \
        --config track_mjx/config/rodent-full-clips.yaml \
        --num-envs 1 256 1024 4096 \
        --reps 50 --warmup 5 \
        --out analysis/2026-06-01-mimicmjx-step-time-breakdown \
        --variant rodent_mlp
"""

import argparse
import json
import gc
from pathlib import Path
from typing import Any

import jax
import pandas as pd
from omegaconf import OmegaConf

from track_mjx.config.utils import prepare_config
from track_mjx.analysis.benchmark import timing, components, policy_factory


def assemble_row(num_envs: int, res: dict, thr: dict, ctrl_dt: float) -> dict[str, Any]:
    """Per-env amortized ms + real-time + throughput for one num_envs config."""
    n = num_envs
    policy_ms = res["policy"]["median_ms"] / n
    mujoco_ms = res["mujoco"]["median_ms"] / n
    rl_env_ms = res["rl_env"]["median_ms"] / n
    total_ms = res["full_step"]["median_ms"] / n
    component_sum_ms = policy_ms + mujoco_ms + rl_env_ms
    real_time_ms = ctrl_dt * 1e3

    batch_wall_ms = thr["per_step_ms"]            # throughput regime, batch ms/step
    env_steps_per_s = n / (batch_wall_ms / 1e3)
    sim_s_per_wall_s = n * ctrl_dt / (batch_wall_ms / 1e3)

    return {
        "num_envs": n,
        "status": "ok",
        "policy_ms": policy_ms,
        "rl_env_ms": rl_env_ms,
        "mujoco_ms": mujoco_ms,
        "total_ms": total_ms,
        "component_sum_ms": component_sum_ms,
        "fused_vs_sum_ratio": (total_ms / component_sum_ms) if component_sum_ms else float("nan"),
        "real_time_ms": real_time_ms,
        "pct_rt_total": real_time_ms / total_ms * 100 if total_ms else float("nan"),
        "pct_rt_mujoco": real_time_ms / mujoco_ms * 100 if mujoco_ms else float("nan"),
        "batch_wall_ms_per_step": batch_wall_ms,
        "env_steps_per_s": env_steps_per_s,
        "sim_s_per_wall_s": sim_s_per_wall_s,
    }


def _error_row(num_envs: int, exc: Exception) -> dict[str, Any]:
    return {"num_envs": num_envs, "status": f"{type(exc).__name__}: {exc}"[:200]}


def benchmark_one(cfg, num_envs: int, reps: int, warmup: int) -> dict[str, Any]:
    ctrl_dt = float(cfg.env_config.ctrl_dt)
    env, state = components.build_env_and_state(cfg, num_envs, seed=0)
    base = components.unwrap(env)
    inference_fn = policy_factory.build_inference_fn(cfg, base, state, seed=0)
    callables = components.build_timed_callables(cfg, env, state, inference_fn)

    res = {
        name: timing.time_op(fn, *args, n_warmup=warmup, n_reps=reps)
        for name, (fn, args) in callables.items()
    }
    control_step = components.build_control_step(env, inference_fn, jax.random.PRNGKey(0))
    thr = timing.scan_throughput(control_step, state, n_steps=100, n_warmup=2, n_reps=5)
    return assemble_row(num_envs, res, thr, ctrl_dt)


def run(config_path: str, num_envs_list: list[int], reps: int, warmup: int,
        out_dir: str, variant: str) -> pd.DataFrame:
    cfg, cfg_dict, _ = prepare_config(OmegaConf.load(config_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for n in num_envs_list:
        print(f"[benchmark] num_envs={n} ...", flush=True)
        try:
            rows.append(benchmark_one(cfg, n, reps, warmup))
        except Exception as exc:  # OOM or anything else: record and continue
            print(f"[benchmark] num_envs={n} FAILED: {exc}", flush=True)
            rows.append(_error_row(n, exc))
        finally:
            jax.clear_caches()
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    (out / "results.json").write_text(json.dumps(rows, indent=2))

    md = render_markdown_table(
        [r for r in rows if r.get("status") == "ok"],
        title=f"{variant} ({config_path})",
        skipped=[r for r in rows if r.get("status") != "ok"],
    )
    (out / f"table_{variant}.md").write_text(md)

    meta = timing.collect_metadata()
    meta.update({
        "config": config_path, "variant": variant,
        "num_envs": num_envs_list, "reps": reps, "warmup": warmup,
        "ctrl_dt": float(cfg.env_config.ctrl_dt), "sim_dt": float(cfg.env_config.sim_dt),
        "n_substeps": int(round(float(cfg.env_config.ctrl_dt) / float(cfg.env_config.sim_dt))),
        "solver": str(cfg.env_config.get("solver")),
        "iterations": int(cfg.env_config.get("iterations", -1)),
        "ls_iterations": int(cfg.env_config.get("ls_iterations", -1)),
        "domain_randomization": False,
    })
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[benchmark] wrote results to {out}", flush=True)
    return df


def render_markdown_table(rows: list[dict], title: str, skipped: list[dict] | None = None) -> str:
    """flybody Table-5-style markdown. All component times per-env amortized (ms)."""
    head = (
        f"### mimic-mjx step-time breakdown — {title}\n\n"
        "All component times are **per-env amortized** (batch wall-clock ÷ num_envs), in **ms**.\n"
        "`Total` is the true fused control step; `Σ comp` is Policy+RL env+MuJoCo (differs from "
        "Total due to XLA fusion). % RT = ctrl_dt / time.\n\n"
        "| num_envs | Policy | RL env | MuJoCo | Total | Σ comp | RT sim (ms) | % RT total | % RT MuJoCo | env-steps/s | sim-s/wall-s |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = ""
    for r in rows:
        body += (
            f"| {r['num_envs']} | {r['policy_ms']:.4f} | {r['rl_env_ms']:.4f} | "
            f"{r['mujoco_ms']:.4f} | {r['total_ms']:.4f} | {r['component_sum_ms']:.4f} | "
            f"{r['real_time_ms']:.1f} | {r['pct_rt_total']:.1f}% | {r['pct_rt_mujoco']:.1f}% | "
            f"{r['env_steps_per_s']:,.0f} | {r['sim_s_per_wall_s']:,.1f} |\n"
        )
    note = ""
    if skipped:
        note = "\n**Skipped/failed:** " + ", ".join(
            f"num_envs={r['num_envs']} ({r['status']})" for r in skipped) + "\n"
    return head + body + note


def main() -> None:
    p = argparse.ArgumentParser(description="mimic-mjx step-time breakdown benchmark")
    p.add_argument("--config", default="track_mjx/config/rodent-full-clips.yaml")
    p.add_argument("--num-envs", type=int, nargs="+", default=[1, 256, 1024, 4096])
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", default="analysis/2026-06-01-mimicmjx-step-time-breakdown")
    p.add_argument("--variant", default="rodent_mlp")
    args = p.parse_args()
    run(args.config, args.num_envs, args.reps, args.warmup, args.out, args.variant)


if __name__ == "__main__":
    main()
