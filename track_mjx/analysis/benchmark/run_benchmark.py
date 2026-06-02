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
    """Two clearly-separated views for one num_envs config.

    (A) Cost breakdown: per-env *amortized* component times (batch wall / num_envs),
        reported in microseconds. ``env_step`` is the fused RL-env + MuJoCo (no policy);
        ``control_step`` = Policy + env.step (the full per-env step cost).
    (B) Speed: ``batch_wall_ms`` is the ACTUAL wall time for one batched control step
        (all envs advance in lockstep). The single-env real-time factor is
        ``ctrl_dt / batch_wall`` (NOT divided by num_envs) — it falls below 100% as the
        batch grows. Aggregate throughput (env-steps/s, sim-s per wall-s) rises instead.
    """
    n = num_envs
    ctrl_dt_ms = ctrl_dt * 1e3

    # (A) per-env amortized component times (ms -> us), per-step latency regime
    policy_ms = res["policy"]["median_ms"] / n
    rl_env_ms = res["rl_env"]["median_ms"] / n
    mujoco_ms = res["mujoco"]["median_ms"] / n
    env_step_ms = res["full_step"]["median_ms"] / n         # fused RL-env + MuJoCo, NO policy
    control_step_ms = policy_ms + env_step_ms               # full per-env control step
    component_sum_ms = policy_ms + rl_env_ms + mujoco_ms    # cross-check (isolated parts)

    # (B) actual batch wall time per control step (throughput regime: policy + env.step)
    batch_wall_ms = thr["per_step_ms"]
    one_env_realtime_pct = ctrl_dt_ms / batch_wall_ms * 100.0 if batch_wall_ms else float("nan")
    env_steps_per_s = n / (batch_wall_ms / 1e3) if batch_wall_ms else float("nan")
    sim_s_per_wall_s = n * ctrl_dt / (batch_wall_ms / 1e3) if batch_wall_ms else float("nan")

    return {
        "num_envs": n,
        "status": "ok",
        # (A) breakdown, microseconds per env (amortized)
        "policy_us": policy_ms * 1e3,
        "rl_env_us": rl_env_ms * 1e3,
        "mujoco_us": mujoco_ms * 1e3,
        "env_step_us": env_step_ms * 1e3,
        "control_step_us": control_step_ms * 1e3,
        "component_sum_us": component_sum_ms * 1e3,
        # (B) speed
        "real_time_sim_ms": ctrl_dt_ms,
        "batch_wall_ms": batch_wall_ms,
        "one_env_realtime_pct": one_env_realtime_pct,
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
    """Two clear tables: (A) per-env cost breakdown, (B) real-time + throughput."""
    note = ""
    if skipped:
        note = "\n**Skipped/failed:** " + ", ".join(
            f"num_envs={r['num_envs']} ({r['status']})" for r in skipped) + "\n"
    if not rows:
        return f"### mimic-mjx step-time breakdown — {title}\n\n_No successful rows._\n" + note

    rt_sim = rows[0]["real_time_sim_ms"]
    table_a = (
        f"### mimic-mjx step-time breakdown — {title}\n\n"
        "**Table A — where the time goes** — per-env *amortized* cost (batch wall-clock ÷ num_envs), "
        "in **µs/env**. `env.step` = fused RL-env + MuJoCo (no policy); `control step` = Policy + env.step.\n"
        "_Caveat: at large batch, Policy and RL env are dispatch-floor-limited (≈ GPU kernel-launch "
        "latency, not compute) — read them as 'negligible', not exact. MuJoCo/control-step scale with "
        "real compute and are trustworthy._\n\n"
        "| num_envs | Policy (µs) | RL env (µs) | MuJoCo (µs) | env.step (µs) | control step (µs) |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in rows:
        table_a += (
            f"| {r['num_envs']} | {r['policy_us']:.3f} | {r['rl_env_us']:.3f} | "
            f"{r['mujoco_us']:.3f} | {r['env_step_us']:.3f} | {r['control_step_us']:.3f} |\n"
        )

    table_b = (
        f"\n**Table B — speed** — one control step advances **{rt_sim:.0f} ms** of simulated time. "
        "`batch wall/step` is the ACTUAL wall time for one step of all envs (they advance in lockstep). "
        "`1-env real-time` = ctrl_dt ÷ batch-wall (a *single* env's speed vs wall-clock; "
        ">100% = faster than real time; falls as batch grows). Throughput columns are *aggregate* over all envs.\n\n"
        "| num_envs | batch wall/step (ms) | 1-env real-time | env-steps/s (all) | sim-s per wall-s (all) |\n"
        "|---:|---:|---:|---:|---:|\n"
    )
    for r in rows:
        table_b += (
            f"| {r['num_envs']} | {r['batch_wall_ms']:.2f} | {r['one_env_realtime_pct']:.1f}% | "
            f"{r['env_steps_per_s']:,.0f} | {r['sim_s_per_wall_s']:,.1f} |\n"
        )
    return table_a + table_b + note


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
