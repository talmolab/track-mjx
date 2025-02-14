import orbax.checkpoint as ocp


def load_config(
    checkpoint_path: str, step_prefix: str = "PPONetwork", step: int = None
):
    """Load the config from a checkpoint."""
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix=step_prefix,
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)
    if step is None:
        step = ckpt_mgr.latest_step()
    return ckpt_mgr.restore(
        step,
        args=ocp.args.Composite(config=ocp.args.JsonRestore()),
    )["config"]
