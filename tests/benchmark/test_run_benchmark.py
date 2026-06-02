from track_mjx.analysis.benchmark import run_benchmark


def test_assemble_row_math():
    # batch medians (ms) for num_envs=4
    res = {
        "policy": {"median_ms": 4.0},
        "mujoco": {"median_ms": 8.0},
        "rl_env": {"median_ms": 2.0},
        "full_step": {"median_ms": 12.0},
    }
    thr = {"per_step_ms": 10.0}
    row = run_benchmark.assemble_row(num_envs=4, res=res, thr=thr, ctrl_dt=0.01)
    # per-env amortized = batch / num_envs
    assert abs(row["policy_ms"] - 1.0) < 1e-9
    assert abs(row["mujoco_ms"] - 2.0) < 1e-9
    assert abs(row["rl_env_ms"] - 0.5) < 1e-9
    assert abs(row["total_ms"] - 3.0) < 1e-9          # full_step 12/4
    assert abs(row["component_sum_ms"] - 3.5) < 1e-9  # 1.0+2.0+0.5
    assert abs(row["real_time_ms"] - 10.0) < 1e-9     # ctrl_dt in ms
    # % real time (total) = 10 / 3.0 * 100
    assert abs(row["pct_rt_total"] - (10.0 / 3.0 * 100)) < 1e-6
    assert abs(row["pct_rt_mujoco"] - (10.0 / 2.0 * 100)) < 1e-6
    # throughput: env-steps/s = num_envs / (per_step_ms/1000)
    assert abs(row["env_steps_per_s"] - (4 / (10.0 / 1000))) < 1e-3


def test_render_markdown_table_has_header():
    rows = [run_benchmark.assemble_row(
        4,
        {"policy": {"median_ms": 4.0}, "mujoco": {"median_ms": 8.0},
         "rl_env": {"median_ms": 2.0}, "full_step": {"median_ms": 12.0}},
        {"per_step_ms": 10.0}, 0.01)]
    md = run_benchmark.render_markdown_table(rows, title="rodent MLP")
    assert "Policy" in md and "MuJoCo" in md and "% RT" in md
    assert "rodent MLP" in md
