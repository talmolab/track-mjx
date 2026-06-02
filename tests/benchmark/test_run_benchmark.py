from track_mjx.analysis.benchmark import run_benchmark


# Batch medians (ms) for a num_envs=4 toy case:
#   policy=4, mujoco=8, rl_env=2, full_step(env.step)=12; throughput batch=10ms/step
_RES = {
    "policy": {"median_ms": 4.0},
    "mujoco": {"median_ms": 8.0},
    "rl_env": {"median_ms": 2.0},
    "full_step": {"median_ms": 12.0},
}
_THR = {"per_step_ms": 10.0}


def test_assemble_row_math():
    row = run_benchmark.assemble_row(num_envs=4, res=_RES, thr=_THR, ctrl_dt=0.01)
    # (A) per-env amortized component cost = batch / num_envs, reported in microseconds
    assert abs(row["policy_us"] - 1000.0) < 1e-6   # 4/4 ms = 1.0 ms
    assert abs(row["mujoco_us"] - 2000.0) < 1e-6   # 8/4 ms = 2.0 ms
    assert abs(row["rl_env_us"] - 500.0) < 1e-6    # 2/4 ms = 0.5 ms
    assert abs(row["env_step_us"] - 3000.0) < 1e-6     # full_step 12/4 = 3.0 ms (no policy)
    assert abs(row["control_step_us"] - 4000.0) < 1e-6  # policy 1.0 + env.step 3.0 = 4.0 ms
    assert abs(row["component_sum_us"] - 3500.0) < 1e-6  # 1.0 + 2.0 + 0.5 ms
    # (B) speed: real-time uses the ACTUAL batch wall time, NOT divided by num_envs
    assert abs(row["real_time_sim_ms"] - 10.0) < 1e-9
    assert abs(row["batch_wall_ms"] - 10.0) < 1e-9
    assert abs(row["one_env_realtime_pct"] - 100.0) < 1e-9   # 10ms sim / 10ms wall = 100%
    assert abs(row["env_steps_per_s"] - (4 / (10.0 / 1000))) < 1e-3   # 400
    assert abs(row["sim_s_per_wall_s"] - (4 * 0.01 / (10.0 / 1000))) < 1e-9  # 4.0


def test_one_env_realtime_falls_with_batch():
    # Same per-env compute, but a bigger batch takes longer wall-clock per step,
    # so a single env's real-time factor must DROP (lockstep advance).
    small = run_benchmark.assemble_row(1, _RES, {"per_step_ms": 8.0}, 0.01)
    big = run_benchmark.assemble_row(4096, _RES, {"per_step_ms": 110.0}, 0.01)
    assert small["one_env_realtime_pct"] > 100.0   # one env faster than real time
    assert big["one_env_realtime_pct"] < 100.0     # one env slower than real time
    assert big["sim_s_per_wall_s"] > small["sim_s_per_wall_s"]  # aggregate throughput rises


def test_render_markdown_table_two_tables():
    rows = [run_benchmark.assemble_row(4, _RES, _THR, 0.01)]
    md = run_benchmark.render_markdown_table(rows, title="rodent MLP")
    assert "Table A" in md and "Table B" in md
    assert "Policy" in md and "MuJoCo" in md
    assert "real-time" in md.lower()
    assert "rodent MLP" in md
