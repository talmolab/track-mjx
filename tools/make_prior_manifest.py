#!/usr/bin/env python
"""Build or verify the sha256 manifest of the frozen DMPO prior checkpoints.

DMPO configs load their frozen priors from ``${oc.env:DMPO_PRIOR_ROOT}/<id>``.
Each ``<id>`` is a directory that holds the pinned ``PriorNetwork_<step>/``
checkpoint (``config``, ``policy``, ``train_state``, ``_CHECKPOINT_METADATA``).
This tool records one sha256 per checkpoint file in
``track_mjx/config/dmpo/priors/MANIFEST.md`` and checks a local copy against it.

Build the manifest (the root is always a CLI argument)::

    python tools/make_prior_manifest.py --root DIR \\
        [--out FILE] [--alias NAME ID PATH ...] [ID:STEP ...]

With no ``ID:STEP`` arguments the four ids in ``DEFAULT_IDS`` are used. Every
``--alias NAME ID PATH`` compares the directory ``PATH`` (an alternate copy
that older configs used) file-by-file against checkpoint ``ID``; the manifest
only records an alias when the bytes are identical, otherwise the build fails.

Verify a copy (re-hashes every file, exits 1 on any difference)::

    python tools/make_prior_manifest.py --verify --root DIR \\
        [--manifest FILE] [--alias NAME ID PATH ...]

The directory digest of a checkpoint is the sha256 of the lines
``<relative path>:<file sha256>\\n`` over every file under the checkpoint
directory, in sorted path order.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / "track_mjx" / "config" / "dmpo" / "priors" / "MANIFEST.md"
)
DEFAULT_IDS = (
    "260407_183223_056788:14",
    "260414_072827_774853:24",
    "260827_153154_576196:24",
    "default:27",
)
STEP_PREFIX = "PriorNetwork_"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class FileEntry:
    path: str  # posix path relative to the checkpoint directory
    size: int
    sha256: str


@dataclass
class Checkpoint:
    id: str
    step: int
    files: list[FileEntry] = field(default_factory=list)
    digest: str | None = None  # directory digest as recorded in a manifest

    @property
    def step_dir(self) -> str:
        return f"{STEP_PREFIX}{self.step}"

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def dir_digest(self) -> str:
        return directory_digest(self.files)


@dataclass
class Manifest:
    checkpoints: list[Checkpoint]
    aliases: list[tuple[str, str]]  # (alias name, id it is byte-identical to)


def hash_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def hash_tree(step_dir: Path) -> list[FileEntry]:
    """Hash every file under ``step_dir``, sorted by relative posix path."""
    entries = []
    for dirpath, _, names in os.walk(step_dir, followlinks=True):
        for name in names:
            full = Path(dirpath) / name
            digest, size = hash_file(full)
            rel = full.relative_to(step_dir).as_posix()
            entries.append(FileEntry(rel, size, digest))
    return sorted(entries, key=lambda e: e.path)


def directory_digest(files: list[FileEntry]) -> str:
    h = hashlib.sha256()
    for entry in sorted(files, key=lambda e: e.path):
        h.update(f"{entry.path}:{entry.sha256}\n".encode())
    return h.hexdigest()


def latest_step(ckpt_dir: Path) -> int | None:
    steps = [
        int(m.group(1))
        for p in ckpt_dir.iterdir()
        if p.is_dir() and (m := re.fullmatch(rf"{STEP_PREFIX}(\d+)", p.name))
    ]
    return max(steps) if steps else None


def parse_spec(spec: str) -> tuple[str, int]:
    cid, sep, step = spec.rpartition(":")
    if not sep or not cid or not step.isdigit():
        raise SystemExit(f"bad checkpoint spec {spec!r}: expected ID:STEP")
    return cid, int(step)


def diff_files(expected: list[FileEntry], actual: list[FileEntry]) -> list[str]:
    """Human-readable differences between two per-file digest lists."""
    exp = {e.path: e for e in expected}
    act = {e.path: e for e in actual}
    problems = [f"missing file: {p}" for p in sorted(exp.keys() - act.keys())]
    problems += [f"unexpected file: {p}" for p in sorted(act.keys() - exp.keys())]
    for path in sorted(exp.keys() & act.keys()):
        if exp[path].size != act[path].size:
            problems.append(
                f"size mismatch: {path} (expected {exp[path].size}, "
                f"got {act[path].size})"
            )
        elif exp[path].sha256 != act[path].sha256:
            problems.append(
                f"sha256 mismatch: {path} (expected {exp[path].sha256[:12]}..., "
                f"got {act[path].sha256[:12]}...)"
            )
    return problems


def check_latest(ckpt_dir: Path, ckpt: Checkpoint) -> list[str]:
    """A config with no pinned step loads the highest step; it must be ours."""
    newest = latest_step(ckpt_dir)
    if newest is not None and newest != ckpt.step:
        return [
            f"{ckpt.id}: highest step on disk is {newest}, not the pinned {ckpt.step}"
        ]
    return []


def prove_alias(ckpt: Checkpoint, name: str, alias_dir: Path) -> list[str]:
    """Problems that stop ``alias_dir`` being byte-identical to ``ckpt``."""
    step_dir = alias_dir / ckpt.step_dir
    if not step_dir.is_dir():
        return [f"alias {name}: no {ckpt.step_dir} under the given path"]
    problems = [
        f"alias {name}: {p}" for p in diff_files(ckpt.files, hash_tree(step_dir))
    ]
    newest = latest_step(alias_dir)
    if newest != ckpt.step:
        problems.append(f"alias {name}: highest step is {newest}, expected {ckpt.step}")
    return problems


def hash_checkpoint(root: Path, cid: str, step: int) -> Checkpoint:
    step_dir = root / cid / f"{STEP_PREFIX}{step}"
    if not step_dir.is_dir():
        raise SystemExit(f"missing checkpoint directory: {step_dir}")
    return Checkpoint(cid, step, hash_tree(step_dir))


def render(manifest: Manifest) -> str:
    lines = [
        "# DMPO prior checkpoints",
        "",
        "Frozen SCAMPER prior checkpoints that the DMPO configs load as",
        "`${oc.env:DMPO_PRIOR_ROOT}/<id>`. Each id is a directory holding the",
        "pinned `PriorNetwork_<step>/` checkpoint (`config`, `policy`,",
        "`train_state`, `_CHECKPOINT_METADATA`). The pinned step is the highest",
        "step of its id, so configs that leave the step unset resolve to it.",
        "",
        "Check a local copy against this file (exit status 0 means every file",
        "matches):",
        "",
        '    python tools/make_prior_manifest.py --verify --root "$DMPO_PRIOR_ROOT"',
        "",
        "The directory digest is the sha256 of the lines",
        "`<relative path>:<file sha256>` (one per file, sorted by path).",
        "",
        "## Checkpoints",
        "",
        "| id | step dir | files | size (bytes) | directory digest (sha256) |",
        "|---|---|---|---|---|",
    ]
    for c in manifest.checkpoints:
        lines.append(
            f"| `{c.id}` | `{c.step_dir}` | {len(c.files)} | {c.size} | "
            f"`{c.dir_digest}` |"
        )
    if manifest.aliases:
        lines += [
            "",
            "## Aliases",
            "",
            "Some configs named a checkpoint by an alternate directory. Each",
            "alias below was compared file by file against the id it maps to:",
            "",
            "| alias | id | byte-identical |",
            "|---|---|---|",
        ]
        lines += [f"| `{a}` | `{i}` | yes |" for a, i in manifest.aliases]
    lines += ["", "## Files"]
    for c in manifest.checkpoints:
        lines += [
            "",
            f"### `{c.id}/{c.step_dir}`",
            "",
            "| file | size (bytes) | sha256 |",
            "|---|---|---|",
        ]
        lines += [f"| `{f.path}` | {f.size} | `{f.sha256}` |" for f in c.files]
    return "\n".join(lines) + "\n"


_CKPT_ROW = re.compile(
    r"^\| `([^`]+)` \| `"
    + STEP_PREFIX
    + r"(\d+)` \| (\d+) \| (\d+) \| `([0-9a-f]{64})` \|$"
)
_ALIAS_ROW = re.compile(r"^\| `([^`]+)` \| `([^`]+)` \| yes \|$")
_FILES_HEAD = re.compile(r"^### `([^`/]+)/" + STEP_PREFIX + r"(\d+)`$")
_FILE_ROW = re.compile(r"^\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \|$")


def parse(text: str) -> Manifest:
    """Read a manifest written by :func:`render` (raises ValueError if malformed)."""
    by_id: dict[str, Checkpoint] = {}
    aliases: list[tuple[str, str]] = []
    section = ""
    current: Checkpoint | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            section, current = line[3:].strip(), None
        elif section == "Checkpoints" and (m := _CKPT_ROW.match(line)):
            by_id[m.group(1)] = Checkpoint(
                m.group(1), int(m.group(2)), digest=m.group(5)
            )
        elif section == "Aliases" and (m := _ALIAS_ROW.match(line)):
            aliases.append((m.group(1), m.group(2)))
        elif section == "Files":
            if m := _FILES_HEAD.match(line):
                current = by_id.get(m.group(1))
                if current is None or current.step != int(m.group(2)):
                    raise ValueError(f"file table for unknown checkpoint: {line}")
            elif current is not None and (m := _FILE_ROW.match(line)):
                current.files.append(FileEntry(m.group(1), int(m.group(2)), m.group(3)))
    if not by_id:
        raise ValueError("no checkpoint rows found")
    for c in by_id.values():
        if not c.files:
            raise ValueError(f"{c.id}: no file rows")
        if c.digest != c.dir_digest:
            raise ValueError(f"{c.id}: directory digest does not match its file rows")
    return Manifest(list(by_id.values()), aliases)


def build(
    root: Path,
    specs: list[str],
    aliases: list[list[str]],
    out: Path,
) -> int:
    checkpoints = [hash_checkpoint(root, *parse_spec(s)) for s in specs]
    by_id = {c.id: c for c in checkpoints}
    problems = []
    for c in checkpoints:
        problems += check_latest(root / c.id, c)
    proven = []
    for name, cid, path in aliases:
        if cid not in by_id:
            problems.append(f"alias {name}: unknown id {cid}")
            continue
        found = prove_alias(by_id[cid], name, Path(path))
        problems += found
        if not found:
            proven.append((name, cid))
    if problems:
        print("\n".join(problems), file=sys.stderr)
        print("manifest NOT written", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(Manifest(checkpoints, proven)))
    for c in checkpoints:
        print(f"{c.id}/{c.step_dir}: {len(c.files)} files, {c.size} bytes")
    for name, cid in proven:
        print(f"alias {name} -> {cid}: byte-identical: yes")
    print(f"wrote {out}")
    return 0


def verify(root: Path, manifest_path: Path, aliases: list[list[str]]) -> int:
    try:
        manifest = parse(manifest_path.read_text())
    except (OSError, ValueError) as err:
        print(f"cannot read manifest {manifest_path}: {err}", file=sys.stderr)
        return 1
    by_id = {c.id: c for c in manifest.checkpoints}
    problems = []
    for want in manifest.checkpoints:
        step_dir = root / want.id / want.step_dir
        if not step_dir.is_dir():
            problems.append(f"{want.id}: missing {want.step_dir}")
            continue
        found = [f"{want.id}: {p}" for p in diff_files(want.files, hash_tree(step_dir))]
        found += check_latest(root / want.id, want)
        problems += found
        if not found:
            print(f"ok {want.id}/{want.step_dir}: {len(want.files)} files")
    recorded = dict(manifest.aliases)
    for name, cid, path in aliases:
        if recorded.get(name) != cid or cid not in by_id:
            problems.append(f"alias {name} -> {cid} is not recorded in the manifest")
            continue
        found = prove_alias(by_id[cid], name, Path(path))
        problems += found
        if not found:
            print(f"ok alias {name} -> {cid}: byte-identical")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(f"VERIFY FAILED: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("verify ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, type=Path, help="directory of <id>/ dirs")
    ap.add_argument("--verify", action="store_true", help="check, do not write")
    ap.add_argument(
        "--manifest",
        "--out",
        dest="manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="manifest to write (build) or read (verify)",
    )
    ap.add_argument(
        "--alias",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "ID", "PATH"),
        help="alternate copy PATH that must be byte-identical to checkpoint ID",
    )
    ap.add_argument("ids", nargs="*", metavar="ID:STEP", help="default: all four")
    args = ap.parse_args(argv)
    if args.verify:
        if args.ids:
            ap.error("--verify takes its ids from the manifest")
        return verify(args.root, args.manifest, args.alias)
    return build(args.root, args.ids or list(DEFAULT_IDS), args.alias, args.manifest)


if __name__ == "__main__":
    sys.exit(main())
