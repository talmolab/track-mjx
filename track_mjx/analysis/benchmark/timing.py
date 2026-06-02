"""Generic, GPU-safe timing primitives. No knowledge of envs or networks.

The single most important detail: JAX dispatch is asynchronous, so every timed
region MUST end with ``jax.block_until_ready`` on the outputs, otherwise we time
queue submission instead of device compute.
"""

import platform
import statistics
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

import jax


def _percentile(sorted_vals: list[float], frac: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(frac * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def time_op(fn: Callable, *args, n_warmup: int = 5, n_reps: int = 50) -> dict[str, float]:
    """Time a single jitted op, per-call latency regime (matches flybody 'one step').

    Runs ``n_warmup`` discarded warmup calls (compile + GPU warm), then ``n_reps``
    timed calls, each followed by ``jax.block_until_ready`` on the outputs. Returns
    summary stats in milliseconds (wall-clock for the *whole batch* — caller divides
    by num_envs to amortize).
    """
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    times_ms: list[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    times_ms.sort()
    return {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": times_ms[0],
        "p25_ms": _percentile(times_ms, 0.25),
        "p75_ms": _percentile(times_ms, 0.75),
        "n_reps": n_reps,
    }


def scan_throughput(
    step_fn: Callable[[Any], Any],
    init_carry: Any,
    n_steps: int = 100,
    n_warmup: int = 2,
    n_reps: int = 5,
) -> dict[str, float]:
    """Throughput regime: lax.scan ``n_steps`` of ``step_fn`` (carry->carry), block once.

    Removes per-call Python/dispatch overhead. Returns median per-step *batch* ms.
    """
    def scanned(carry):
        carry, _ = jax.lax.scan(
            lambda c, _: (step_fn(c), None), carry, None, length=n_steps
        )
        return carry

    jscanned = jax.jit(scanned)
    for _ in range(n_warmup):
        jax.block_until_ready(jscanned(init_carry))
    per_step_ms: list[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        jax.block_until_ready(jscanned(init_carry))
        per_step_ms.append((time.perf_counter() - t0) * 1e3 / n_steps)
    per_step_ms.sort()
    return {
        "per_step_ms": statistics.median(per_step_ms),
        "n_steps": n_steps,
        "n_reps": n_reps,
    }


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def collect_metadata() -> dict[str, Any]:
    """Hardware + library versions for reproducibility. Best-effort; never raises."""
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        gpu = "unknown"
    return {
        "gpu": gpu,
        "cpu": _cpu_model(),
        "python": platform.python_version(),
        "jax": _pkg_version("jax"),
        "jaxlib": _pkg_version("jaxlib"),
        "mujoco": _pkg_version("mujoco"),
        "mujoco_mjx": _pkg_version("mujoco-mjx"),
        "brax": _pkg_version("brax"),
        "mujoco_playground": _pkg_version("playground"),
        "vnl_playground": _pkg_version("vnl_playground"),
        "track_mjx": _pkg_version("track-mjx"),
        "jax_devices": [str(d) for d in jax.devices()],
    }
