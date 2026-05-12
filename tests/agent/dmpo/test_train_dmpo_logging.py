import re

from track_mjx.agent.dmpo.train_dmpo_logging import make_run_id


def test_make_run_id_format():
    rid = make_run_id(
        config_name="rodent-dmpo-vision-scratch-position",
        seed=0,
        git_sha="aa2abd9c0d1e2f3",
    )
    # Expect: <config_short>_seed<seed>_g<sha7>
    assert re.fullmatch(
        r"rodent-dmpo-vision-scratch-position_seed0_gaa2abd9", rid
    ), f"unexpected run_id: {rid}"


def test_make_run_id_no_git():
    # Falls back to "nogit" suffix when sha is None / empty.
    rid = make_run_id("foo", 7, git_sha=None)
    assert rid == "foo_seed7_gnogit"


def test_make_run_id_truncates_sha():
    rid = make_run_id("foo", 0, git_sha="0123456789abcdef")
    assert rid.endswith("_g0123456")


def test_save_load_wandb_state(tmp_path):
    from track_mjx.agent.dmpo.train_dmpo_logging import (
        save_wandb_state, load_wandb_state,
    )
    save_wandb_state(tmp_path, "myrun_seed0_gabcdef0")
    state = load_wandb_state(tmp_path)
    assert state == {"wandb_run_id": "myrun_seed0_gabcdef0"}


def test_load_wandb_state_missing(tmp_path):
    from track_mjx.agent.dmpo.train_dmpo_logging import load_wandb_state
    assert load_wandb_state(tmp_path) is None
