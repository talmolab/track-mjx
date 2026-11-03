"""Tests for tools/relocate_paths.py (CPU only, fake configs in a tmp dir)."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "relocate_paths.py"
MANIFEST = REPO / "track_mjx" / "config" / "dmpo" / "priors" / "MANIFEST.md"

_spec = importlib.util.spec_from_file_location("relocate_paths", TOOL)
rp = importlib.util.module_from_spec(_spec)
sys.modules["relocate_paths"] = rp
_spec.loader.exec_module(rp)

OLD = "/old/root"
PRIOR = "${oc.env:DMPO_PRIOR_ROOT}"
CKPT = "${oc.env:DMPO_CKPT_ROOT,dmpo_checkpoints}"
ALIASES = {"combined_jump": "260407_183223_056788"}


def _relocate(key: str, value: str) -> str:
    return rp.relocate_value(key, value, [OLD], ALIASES)


def test_prior_path_becomes_prior_root_id():
    assert (
        _relocate("prior_checkpoint_path", f"{OLD}/SCAMPER/model_checkpoints/260827")
        == f"{PRIOR}/260827"
    )


def test_prior_alias_maps_to_the_manifest_id():
    assert (
        _relocate("prior_checkpoint_path", f"{OLD}/data/prior/combined_jump")
        == f"{PRIOR}/260407_183223_056788"
    )


def test_output_dir_keeps_its_basename():
    for key in ("checkpoint_dir", "model_path", "mimic_checkpoint_dir"):
        assert _relocate(key, f"{OLD}/a/b/run_x") == f"{CKPT}/run_x"


def test_graft_source_keeps_run_and_step():
    assert (
        _relocate("graft_ff_checkpoint", f"{OLD}/ckpts/run_a/DMPONetwork_100")
        == f"{CKPT}/run_a/DMPONetwork_100"
    )


@pytest.mark.parametrize("kind", ["final", "warmstart_src", "graftsrc"])
def test_archived_graft_source_maps_back_to_run_and_step(kind):
    value = f"{OLD}/baselines/{kind}_arm_w_big_DMPONetwork_297676800"
    assert (
        _relocate("warm_start_dmpo_checkpoint", value)
        == f"{CKPT}/arm_w_big/DMPONetwork_297676800"
    )


def test_relative_and_already_relocated_values_are_untouched():
    for value in ("./checkpoints/dmpo", f"{CKPT}/run", f"{PRIOR}/x", "null"):
        assert _relocate("checkpoint_dir", value) == value


def test_absolute_value_under_no_old_root_is_an_error():
    with pytest.raises(rp.RelocationError, match="none of"):
        _relocate("checkpoint_dir", "/elsewhere/run")


def test_relocate_text_rewrites_values_and_keeps_comments_and_other_keys():
    text = (
        "# @package _global_\n"
        "transfer:\n"
        f"  prior_checkpoint_path: {OLD}/p/combined_jump  # frozen prior\n"
        "  prior_checkpoint_step: null\n"
        f"checkpoint_dir: {OLD}/c/run1\n"
        f"# checkpoint_dir: {OLD}/commented/out\n"
        f"notes: see {OLD}/not/a/key\n"
    )
    new, changed = rp.relocate_text(text, [OLD], ALIASES)
    assert changed == 2
    assert new == (
        "# @package _global_\n"
        "transfer:\n"
        f"  prior_checkpoint_path: {PRIOR}/260407_183223_056788  # frozen prior\n"
        "  prior_checkpoint_step: null\n"
        f"checkpoint_dir: {CKPT}/run1\n"
        f"# checkpoint_dir: {OLD}/commented/out\n"
        f"notes: see {OLD}/not/a/key\n"
    )


def test_load_aliases_reads_the_committed_manifest():
    assert rp.load_aliases(MANIFEST) == {
        "combined_jump": "260407_183223_056788",
        "combined_clips_gen0": "260414_072827_774853",
    }


def test_main_rewrites_a_directory_and_check_mode_reports_without_writing(
    tmp_path, capsys
):
    arm = tmp_path / "sub" / "arm.yaml"
    arm.parent.mkdir()
    arm.write_text(f"checkpoint_dir: {OLD}/x/run\n")

    assert rp.main(["--old-root", OLD, "--check", str(tmp_path)]) == 1
    assert arm.read_text() == f"checkpoint_dir: {OLD}/x/run\n"

    assert rp.main(["--old-root", OLD, str(tmp_path)]) == 0
    assert arm.read_text() == f"checkpoint_dir: {CKPT}/run\n"
    assert "1 value(s) relocated" in capsys.readouterr().out

    # A second pass finds nothing to do, in both modes.
    assert rp.main(["--old-root", OLD, "--check", str(tmp_path)]) == 0


def test_main_fails_on_an_unrelocatable_path(tmp_path, capsys):
    arm = tmp_path / "arm.yaml"
    arm.write_text("checkpoint_dir: /elsewhere/run\n")
    assert rp.main(["--old-root", OLD, str(tmp_path)]) == 2
    assert "none of" in capsys.readouterr().err
