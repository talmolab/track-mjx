"""Grafts and resumes refuse a checkpoint trained with other alignment settings.

A checkpoint records the six acme-alignment fields in its saved config
(``checkpoint.record_alignment``). A field it does not record holds its
``LEGACY_ALIGNMENT`` value: the checkpoint predates the fields. Continuing from
a checkpoint whose settings differ from this config's would switch the
learning algorithm partway through a run, so the entries refuse unless the
mismatch is allowed explicitly:

* a graft (``transfer.graft_ff_checkpoint``), override
  ``transfer.graft_allow_algorithm_mismatch``;
* a resume (``checkpoint_dir`` already holds a checkpoint), override
  ``train_config.allow_alignment_change_on_resume``.
"""

import dataclasses
import logging

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest

from track_mjx.agent.dmpo import checkpoint
from track_mjx.agent.dmpo.config import ALIGNMENT_FIELDS, LEGACY_ALIGNMENT, DMPOConfig
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.dmpo.networks import make_dmpo_networks
from track_mjx.agent.dmpo.optim_kl_anchor import make_kl_anchor_optimizers

OBS, ACT = 7, 3
LEGACY_RESOLVED = dict(LEGACY_ALIGNMENT)  # "current" needs no resolving
ALIGNED_RESOLVED = {
    **{name: getattr(DMPOConfig(), name) for name in ALIGNMENT_FIELDS},
    "policy_loss_state": "next",
}


def _write_config(step_dir, config) -> None:
    ocp.Checkpointer(ocp.JsonCheckpointHandler()).save(
        step_dir / "config", args=ocp.args.JsonSave(config)
    )


def _pre_alignment_source(tmp_path, name="DMPONetwork_100"):
    """A checkpoint directory whose saved config predates the fields."""
    step_dir = tmp_path / name
    _write_config(step_dir, {"train_config": {"vmin": 0.0, "n_step": 5}})
    return step_dir


def _aligned_source(tmp_path, cfg: DMPOConfig, name="DMPONetwork_100"):
    step_dir = tmp_path / name
    _write_config(step_dir, checkpoint.record_alignment({"train_config": {}}, cfg))
    return step_dir


# --- saved_alignment / record_alignment ----------------------------------------


def test_saved_alignment_absent_fields_are_legacy(tmp_path):
    assert checkpoint.saved_alignment(_pre_alignment_source(tmp_path)) == (
        LEGACY_RESOLVED
    )


def test_saved_alignment_partial_fields(tmp_path):
    step_dir = tmp_path / "DMPONetwork_1"
    _write_config(step_dir, {"train_config": {"critic_target_mode": "sample_mixture"}})
    saved = checkpoint.saved_alignment(step_dir)
    assert saved == {**LEGACY_RESOLVED, "critic_target_mode": "sample_mixture"}


def test_record_alignment_round_trips(tmp_path):
    assert checkpoint.saved_alignment(_aligned_source(tmp_path, DMPOConfig())) == (
        ALIGNED_RESOLVED
    )


def test_record_alignment_uses_the_imitation_layout(tmp_path):
    """The imitation entry keeps its train config at train_setup.train_config."""
    config = {"train_setup": {"train_config": {"n_step": 5}}}
    checkpoint.record_alignment(config, DMPOConfig())
    assert "train_config" not in config
    step_dir = tmp_path / "DMPONetwork_1"
    _write_config(step_dir, config)
    assert checkpoint.saved_alignment(step_dir) == ALIGNED_RESOLVED


def test_saved_auto_resolves_by_the_saved_learner(tmp_path):
    step_dir = tmp_path / "DMPONetwork_1"
    train = {name: getattr(DMPOConfig(), name) for name in ALIGNMENT_FIELDS}
    _write_config(step_dir, {"train_config": {**train, "rnn_bptt_length": 8}})
    assert checkpoint.saved_alignment(step_dir)["policy_loss_state"] == "current"


def test_unreadable_saved_config_raises(tmp_path):
    step_dir = tmp_path / "DMPONetwork_1"
    (step_dir / "train_state").mkdir(parents=True)
    with pytest.raises(ValueError, match="could not read the saved config"):
        checkpoint.saved_alignment(step_dir)


def test_latest_step_dir_skips_unfinished_saves(tmp_path):
    assert checkpoint.latest_step_dir(tmp_path / "missing") is None
    assert checkpoint.latest_step_dir(tmp_path) is None
    for name in (
        "DMPONetwork_20",
        "DMPONetwork_100",
        "DMPONetwork_300.orbax-checkpoint-tmp-7",
        "other_400",
    ):
        (tmp_path / name).mkdir()
    assert checkpoint.latest_step_dir(tmp_path).name == "DMPONetwork_100"


# --- graft ---------------------------------------------------------------------


def test_graft_from_pre_alignment_checkpoint_refuses(kl_anchor_entry, tmp_path):
    source = _pre_alignment_source(tmp_path)
    with pytest.raises(ValueError) as err:
        kl_anchor_entry["_check_graft_alignment"](str(source), DMPOConfig(), {})
    message = str(err.value)
    for name in ALIGNMENT_FIELDS:  # every field differs from the aligned defaults
        assert f"train_config.{name}:" in message
    assert "checkpoint='mean_action' this config='sample_mixture'" in message
    assert "transfer.graft_allow_algorithm_mismatch" in message


def test_graft_override_allows_with_warning(kl_anchor_entry, tmp_path, caplog):
    source = _pre_alignment_source(tmp_path)
    transfer = {"graft_allow_algorithm_mismatch": True}
    with caplog.at_level(logging.WARNING):
        kl_anchor_entry["_check_graft_alignment"](str(source), DMPOConfig(), transfer)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "the override must log a WARNING"
    text = warnings[-1].getMessage()
    assert "transfer.graft_allow_algorithm_mismatch" in text
    assert "train_config.policy_grad_clip_scope" in text


def test_graft_refuses_a_single_differing_field(kl_anchor_entry, tmp_path):
    source = _aligned_source(tmp_path, DMPOConfig())
    cfg = DMPOConfig(critic_action_input="raw")
    with pytest.raises(ValueError) as err:
        kl_anchor_entry["_check_graft_alignment"](str(source), cfg, {})
    message = str(err.value)
    assert "train_config.critic_action_input:" in message
    others = [n for n in ALIGNMENT_FIELDS if n != "critic_action_input"]
    assert not any(f"train_config.{n}:" in message for n in others)


# --- resume --------------------------------------------------------------------


def _resume(entry, ckpt_dir, cfg, allow):
    return checkpoint.check_resume_alignment(
        ckpt_dir, cfg, override_key=entry["RESUME_OVERRIDE_KEY"], allow=allow
    )


def test_resume_with_mismatched_alignment_refuses(kl_anchor_entry, tmp_path, caplog):
    assert kl_anchor_entry["RESUME_OVERRIDE_KEY"] == (
        "train_config.allow_alignment_change_on_resume"
    )
    _pre_alignment_source(tmp_path, "DMPONetwork_100")
    with pytest.raises(ValueError) as err:
        _resume(kl_anchor_entry, tmp_path, DMPOConfig(), allow=False)
    message = str(err.value)
    assert "DMPONetwork_100" in message
    assert "train_config.bootstrap_on_truncation:" in message
    assert "train_config.allow_alignment_change_on_resume" in message

    with caplog.at_level(logging.WARNING):
        saved = _resume(kl_anchor_entry, tmp_path, DMPOConfig(), allow=True)
    assert saved == LEGACY_RESOLVED
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_resume_reads_the_newest_checkpoint(kl_anchor_entry, tmp_path):
    """An old pre-alignment save does not block a run that now records them."""
    _pre_alignment_source(tmp_path, "DMPONetwork_100")
    _aligned_source(tmp_path, DMPOConfig(), "DMPONetwork_200")
    assert _resume(kl_anchor_entry, tmp_path, DMPOConfig(), False) == (ALIGNED_RESOLVED)


def test_fresh_run_dir_passes(kl_anchor_entry, tmp_path):
    assert _resume(kl_anchor_entry, tmp_path, DMPOConfig(), allow=False) is None


# --- matching settings ---------------------------------------------------------


@pytest.mark.parametrize(
    "saved_cfg, cfg",
    [
        (DMPOConfig(), DMPOConfig()),
        (DMPOConfig(), DMPOConfig(policy_loss_state="next")),  # "auto" on FF
        (DMPOConfig(**LEGACY_ALIGNMENT), DMPOConfig(**LEGACY_ALIGNMENT)),
    ],
    ids=["aligned", "auto_equals_next", "legacy"],
)
def test_matching_fields_pass(kl_anchor_entry, tmp_path, caplog, saved_cfg, cfg):
    source = _aligned_source(tmp_path, saved_cfg)
    with caplog.at_level(logging.WARNING):
        kl_anchor_entry["_check_graft_alignment"](str(source), cfg, {})
        _resume(kl_anchor_entry, tmp_path, cfg, allow=False)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_pre_alignment_checkpoint_matches_the_legacy_preset(kl_anchor_entry, tmp_path):
    source = _pre_alignment_source(tmp_path)
    cfg = DMPOConfig(**LEGACY_ALIGNMENT)
    kl_anchor_entry["_check_graft_alignment"](str(source), cfg, {})
    _resume(kl_anchor_entry, tmp_path, cfg, allow=False)


# --- the optimizer-state structure across policy_grad_clip_scope ----------------


def _kl_anchor_state(cfg: DMPOConfig):
    nets = make_dmpo_networks(OBS, ACT, cfg)
    state = init_training_state(
        jax.random.key(0), nets, {"obs_size": OBS, "action_size": ACT}, cfg
    )
    pol_opt, _, _ = make_kl_anchor_optimizers(cfg)
    return state._replace(policy_opt_state=pol_opt.init(state.policy_params))


def _structure(tree):
    return jax.tree_util.tree_structure(tree)


def test_resume_across_clip_scope(kl_anchor_entry, tmp_path):
    """A pre-alignment checkpoint is refused before anything is restored.

    Its policy optimizer state is per-block; the aligned config's is global,
    so the two trees differ. With the override the checkpoint is restored into
    a template of its own structure and the policy optimizer state is then
    re-initialized for this config's optimizer.
    """
    legacy_cfg = DMPOConfig(**LEGACY_ALIGNMENT)
    saved_state = _kl_anchor_state(legacy_cfg)
    saved_state = saved_state._replace(
        policy_params=jax.tree.map(lambda x: x + 1.0, saved_state.policy_params)
    )
    mgr = checkpoint.make_checkpointer(str(tmp_path))
    checkpoint.save(mgr, 100, saved_state, config={"train_config": {"n_step": 5}})
    mgr.wait_until_finished()

    cfg = DMPOConfig()
    fresh = _kl_anchor_state(cfg)
    assert _structure(fresh.policy_opt_state) != _structure(
        saved_state.policy_opt_state
    )
    with pytest.raises(ValueError, match="policy_grad_clip_scope"):
        _resume(kl_anchor_entry, tmp_path, cfg, allow=False)

    saved = _resume(kl_anchor_entry, tmp_path, cfg, allow=True)
    restored = kl_anchor_entry["_restore_for_resume"](mgr, fresh, cfg, saved, {})
    assert _structure(restored.policy_opt_state) == _structure(fresh.policy_opt_state)
    same = jax.tree.map(
        lambda a, b: bool(jnp.array_equal(a, b)),
        restored.policy_params,
        saved_state.policy_params,
    )
    assert all(jax.tree.leaves(same))

    # The re-initialized state drives this config's optimizer.
    pol_opt, _, _ = make_kl_anchor_optimizers(cfg)
    grads = jax.tree.map(jnp.ones_like, restored.policy_params)
    pol_opt.update(grads, restored.policy_opt_state, restored.policy_params)


def test_restore_for_resume_without_scope_change_is_plain_restore(
    kl_anchor_entry, tmp_path
):
    cfg = DMPOConfig()
    state = _kl_anchor_state(cfg)
    mgr = checkpoint.make_checkpointer(str(tmp_path))
    checkpoint.save(mgr, 5, state, config={})
    mgr.wait_until_finished()
    saved = dataclasses.asdict(cfg)
    restored = kl_anchor_entry["_restore_for_resume"](mgr, state, cfg, saved, {})
    assert int(restored.steps) == int(state.steps)
    assert (
        kl_anchor_entry["_restore_for_resume"](
            checkpoint.make_checkpointer(str(tmp_path / "empty")), state, cfg, None, {}
        )
        is None
    )
