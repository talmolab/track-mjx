"""Tests for tools/make_prior_manifest.py (CPU only, fake files in a tmp dir)."""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "make_prior_manifest.py"
COMMITTED = REPO / "track_mjx" / "config" / "dmpo" / "priors" / "MANIFEST.md"

_spec = importlib.util.spec_from_file_location("make_prior_manifest", TOOL)
mpm = importlib.util.module_from_spec(_spec)
sys.modules["make_prior_manifest"] = mpm
_spec.loader.exec_module(mpm)


def _make_checkpoint(root: Path, cid: str, step: int, seed: int) -> Path:
    """A tiny directory with the real layout: config/, policy/ (nested), ..."""
    step_dir = root / cid / f"PriorNetwork_{step}"
    files = {
        "_CHECKPOINT_METADATA": f"meta {seed}".encode(),
        "config/metadata": f"cfg {seed}".encode(),
        "policy/_METADATA": f"pol {seed}".encode(),
        "policy/d/shard0": bytes(range(256)) * (seed + 1),
        "train_state/manifest.ocdbt": b"\x00" * 17 + bytes([seed]),
    }
    for rel, data in files.items():
        path = step_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return step_dir


@pytest.fixture
def root(tmp_path):
    base = tmp_path / "priors"
    _make_checkpoint(base, "alpha", 3, seed=1)
    _make_checkpoint(base, "beta", 5, seed=2)
    return base


def _build(root, out, *extra, ids=("alpha:3", "beta:5")):
    return mpm.main(["--root", str(root), "--out", str(out), *extra, *ids])


def _verify(root, manifest, *extra):
    return mpm.main(
        ["--verify", "--root", str(root), "--manifest", str(manifest), *extra]
    )


def test_build_then_verify_roundtrip(root, tmp_path, capsys):
    out = tmp_path / "MANIFEST.md"
    assert _build(root, out) == 0
    text = out.read_text()
    assert str(tmp_path) not in text  # no host path leaks into the manifest
    parsed = mpm.parse(text)
    assert [(c.id, c.step) for c in parsed.checkpoints] == [("alpha", 3), ("beta", 5)]
    assert {f.path for f in parsed.checkpoints[0].files} == {
        "_CHECKPOINT_METADATA",
        "config/metadata",
        "policy/_METADATA",
        "policy/d/shard0",
        "train_state/manifest.ocdbt",
    }
    assert _verify(root, out) == 0
    assert "verify ok" in capsys.readouterr().out


def test_manifest_is_deterministic(root, tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    assert _build(root, a) == 0
    assert _build(root, b) == 0
    assert a.read_text() == b.read_text()


def test_verify_follows_symlinked_ids(root, tmp_path):
    out = tmp_path / "MANIFEST.md"
    assert _build(root, out) == 0
    staged = tmp_path / "staged"
    staged.mkdir()
    for cid in ("alpha", "beta"):
        (staged / cid).symlink_to(root / cid)
    assert _verify(staged, out) == 0


def test_verify_detects_flipped_byte(root, tmp_path, capsys):
    out = tmp_path / "MANIFEST.md"
    assert _build(root, out) == 0
    target = root / "beta" / "PriorNetwork_5" / "policy" / "d" / "shard0"
    data = bytearray(target.read_bytes())
    data[3] ^= 0xFF  # same size, different content
    target.write_bytes(bytes(data))
    assert _verify(root, out) == 1
    err = capsys.readouterr().err
    assert "beta: sha256 mismatch: policy/d/shard0" in err
    assert "alpha" not in err


def test_verify_detects_size_change_missing_and_extra_files(root, tmp_path, capsys):
    out = tmp_path / "MANIFEST.md"
    assert _build(root, out) == 0
    step = root / "alpha" / "PriorNetwork_3"
    (step / "policy" / "d" / "shard0").write_bytes(b"short")
    (step / "config" / "metadata").unlink()
    (step / "stray.bin").write_bytes(b"x")
    assert _verify(root, out) == 1
    err = capsys.readouterr().err
    assert "alpha: size mismatch: policy/d/shard0" in err
    assert "alpha: missing file: config/metadata" in err
    assert "alpha: unexpected file: stray.bin" in err


def test_verify_detects_missing_checkpoint_dir(root, tmp_path, capsys):
    out = tmp_path / "MANIFEST.md"
    assert _build(root, out) == 0
    shutil.rmtree(root / "beta")
    assert _verify(root, out) == 1
    assert "beta: missing PriorNetwork_5" in capsys.readouterr().err


def test_verify_rejects_a_newer_step_than_the_pinned_one(root, tmp_path, capsys):
    out = tmp_path / "MANIFEST.md"
    assert _build(root, out) == 0
    _make_checkpoint(root, "alpha", 4, seed=9)
    assert _verify(root, out) == 1
    assert "highest step on disk is 4, not the pinned 3" in capsys.readouterr().err
    assert _build(root, tmp_path / "again.md") == 1  # build refuses it too


def test_verify_rejects_an_edited_manifest(root, tmp_path, capsys):
    out = tmp_path / "MANIFEST.md"
    assert _build(root, out) == 0
    text = out.read_text()
    digest = mpm.parse(text).checkpoints[0].files[0].sha256
    out.write_text(text.replace(digest, "0" * 64, 1))
    assert _verify(root, out) == 1  # file row no longer matches its directory digest
    assert "directory digest" in capsys.readouterr().err


def test_alias_proof_recorded_and_reverified(root, tmp_path, capsys):
    shutil.copytree(root / "alpha", tmp_path / "alias_alpha")
    out = tmp_path / "MANIFEST.md"
    assert (
        _build(
            root, out, "--alias", "old_alpha", "alpha", str(tmp_path / "alias_alpha")
        )
        == 0
    )
    assert "alias old_alpha -> alpha: byte-identical: yes" in capsys.readouterr().out
    assert mpm.parse(out.read_text()).aliases == [("old_alpha", "alpha")]
    assert "| `old_alpha` | `alpha` | yes |" in out.read_text()
    assert (
        _verify(
            root, out, "--alias", "old_alpha", "alpha", str(tmp_path / "alias_alpha")
        )
        == 0
    )
    # A drifting alias copy is caught on verify ...
    drift = tmp_path / "alias_alpha" / "PriorNetwork_3" / "config" / "metadata"
    drift.write_bytes(b"cfg 9")  # same size as the original, different bytes
    assert (
        _verify(
            root, out, "--alias", "old_alpha", "alpha", str(tmp_path / "alias_alpha")
        )
        == 1
    )
    assert (
        "alias old_alpha: sha256 mismatch: config/metadata" in capsys.readouterr().err
    )
    # ... and an alias the manifest never recorded is refused.
    assert (
        _verify(root, out, "--alias", "other", "alpha", str(tmp_path / "alias_alpha"))
        == 1
    )


def test_non_identical_alias_fails_the_build_and_writes_nothing(root, tmp_path, capsys):
    shutil.copytree(root / "alpha", tmp_path / "alias_alpha")
    (tmp_path / "alias_alpha" / "PriorNetwork_3" / "config" / "metadata").write_bytes(
        b"cfg 9"
    )
    out = tmp_path / "MANIFEST.md"
    assert (
        _build(
            root, out, "--alias", "old_alpha", "alpha", str(tmp_path / "alias_alpha")
        )
        == 1
    )
    assert not out.exists()
    assert (
        "alias old_alpha: sha256 mismatch: config/metadata" in capsys.readouterr().err
    )


def test_alias_with_a_newer_latest_step_is_not_identical(root, tmp_path, capsys):
    shutil.copytree(root / "alpha", tmp_path / "alias_alpha")
    _make_checkpoint(tmp_path, "alias_alpha", 4, seed=7)  # newer step in the alias
    out = tmp_path / "MANIFEST.md"
    assert (
        _build(
            root, out, "--alias", "old_alpha", "alpha", str(tmp_path / "alias_alpha")
        )
        == 1
    )
    assert "alias old_alpha: highest step is 4, expected 3" in capsys.readouterr().err


def test_bad_spec_and_missing_dir_are_reported(root, tmp_path):
    with pytest.raises(SystemExit, match="ID:STEP"):
        _build(root, tmp_path / "m.md", ids=("alpha",))
    with pytest.raises(SystemExit, match="missing checkpoint directory"):
        _build(root, tmp_path / "m.md", ids=("alpha:9",))


def test_cli_runs_as_a_script(root, tmp_path):
    out = tmp_path / "MANIFEST.md"
    build = [sys.executable, str(TOOL), "--root", str(root), "--out", str(out)]
    check = [sys.executable, str(TOOL), "--verify", "--root", str(root)]
    check += ["--manifest", str(out)]
    assert subprocess.run([*build, "alpha:3", "beta:5"], check=False).returncode == 0
    assert subprocess.run(check, check=False).returncode == 0
    (root / "alpha" / "PriorNetwork_3" / "stray.bin").write_bytes(b"x")
    assert subprocess.run(check, check=False).returncode == 1


def test_committed_manifest_is_well_formed_and_path_free():
    text = COMMITTED.read_text()
    parsed = mpm.parse(text)  # also checks each digest against its file rows
    assert [(c.id, c.step) for c in parsed.checkpoints] == [
        ("260407_183223_056788", 14),
        ("260414_072827_774853", 24),
        ("260827_153154_576196", 24),
        ("default", 27),
    ]
    assert parsed.aliases == [
        ("combined_jump", "260407_183223_056788"),
        ("combined_clips_gen0", "260414_072827_774853"),
    ]
    assert not re.search(r"(^|[\s`(])/(?:home|Users|root|scratch|mnt)\b", text)
    assert all(len(c.files) > 0 and c.size > 0 for c in parsed.checkpoints)


@pytest.mark.artifact
def test_staged_priors_match_the_committed_manifest(capsys):
    prior_root = os.environ.get("DMPO_PRIOR_ROOT")
    if not prior_root or not Path(prior_root).is_dir():
        pytest.skip("DMPO_PRIOR_ROOT is not set to a staged prior directory")
    rc = mpm.main(["--verify", "--root", prior_root, "--manifest", str(COMMITTED)])
    assert rc == 0, capsys.readouterr().err
