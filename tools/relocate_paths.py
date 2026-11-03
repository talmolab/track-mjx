#!/usr/bin/env python
"""Relocate absolute machine paths in DMPO arm configs to environment roots.

Research configs name their frozen priors, graft sources and run directories by
absolute paths. This tool rewrites those values so that every path hangs off one
of two environment variables:

* ``prior_checkpoint_path`` becomes ``${oc.env:DMPO_PRIOR_ROOT}/<id>``. ``<id>``
  is the last path component, mapped through the alias table of
  ``track_mjx/config/dmpo/priors/MANIFEST.md`` (an alternate directory name that
  holds a byte-identical copy of a manifest checkpoint becomes that checkpoint's
  id).
* ``graft_ff_checkpoint`` and ``warm_start_dmpo_checkpoint`` become
  ``${oc.env:DMPO_CKPT_ROOT,dmpo_checkpoints}/<run>/<step dir>``. A source that
  was archived as a single directory named ``<kind>_<run>_<step dir>`` (kind is
  ``final``, ``warmstart_src`` or ``graftsrc``) maps back to ``<run>/<step dir>``;
  any other path keeps its last two components.
* ``checkpoint_dir``, ``model_path`` and ``mimic_checkpoint_dir`` become
  ``${oc.env:DMPO_CKPT_ROOT,dmpo_checkpoints}/<last component>``.

Only values that start with one of the ``--old-root`` prefixes are rewritten.
There is no built-in root: the old roots are always CLI arguments. A root is
matched as a literal string prefix, so an interpolated root such as
``'${oc.env:HOME}/work'`` can be passed to relocate values written that way.

Every value of the keys above is either relocated, left alone, or an error:

* a value that already starts with one of the two environment roots is left
  alone;
* a relative value (``./runs/x``) and ``null`` name no machine and are left
  alone;
* a value that starts with ``/``, ``~`` or ``${`` (any other interpolation) and
  is under no old root is an error. Single or double quotes around a value are
  accepted and kept.

A value of one of these keys that is written in a form this tool does not see
(for example on a continuation line, or under a different key) is not checked.

Usage::

    python tools/relocate_paths.py --old-root /path/to/old/root \\
        track_mjx/config/dmpo

Pass ``--old-root`` once per root. ``--check`` rewrites nothing and exits 1 if
any value would change. Trailing comments on a rewritten line are kept.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / "track_mjx" / "config" / "dmpo" / "priors" / "MANIFEST.md"
)

PRIOR_ROOT = "${oc.env:DMPO_PRIOR_ROOT}"
CKPT_ROOT = "${oc.env:DMPO_CKPT_ROOT,dmpo_checkpoints}"

PRIOR_KEYS = ("prior_checkpoint_path",)
GRAFT_KEYS = ("graft_ff_checkpoint", "warm_start_dmpo_checkpoint")
OUTPUT_KEYS = ("checkpoint_dir", "model_path", "mimic_checkpoint_dir")

_ARCHIVE_NAME = re.compile(
    r"^(?:final|warmstart_src|graftsrc)_(?P<run>.+)_(?P<step>[A-Za-z]+_\d+)$"
)
_ALIAS_ROW = re.compile(r"^\|\s*`(?P<alias>[^`]+)`\s*\|\s*`(?P<id>[^`]+)`\s*\|")
_LINE = re.compile(
    r"^(?P<head>\s*(?P<key>"
    + "|".join(PRIOR_KEYS + GRAFT_KEYS + OUTPUT_KEYS)
    + r"):\s*)(?P<value>\"[^\"]*\"|'[^']*'|[^\s#]+)(?P<tail>.*)$"
)


class RelocationError(ValueError):
    """A path value that cannot be relocated."""


def load_aliases(manifest: Path) -> dict[str, str]:
    """Reads the alias table of a prior manifest.

    Args:
        manifest: Path of the ``MANIFEST.md`` written by ``make_prior_manifest``.

    Returns:
        ``{alias directory name: checkpoint id}``.
    """
    aliases: dict[str, str] = {}
    in_aliases = False
    for line in manifest.read_text().splitlines():
        if line.startswith("## "):
            in_aliases = line.strip() == "## Aliases"
        elif in_aliases and (match := _ALIAS_ROW.match(line)):
            aliases[match["alias"]] = match["id"]
    return aliases


def _strip_root(value: str, old_roots: list[str]) -> str | None:
    """Returns ``value`` relative to the first old root that prefixes it."""
    for root in old_roots:
        prefix = root.rstrip("/") + "/"
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def relocate_value(
    key: str, value: str, old_roots: list[str], aliases: dict[str, str]
) -> str:
    """Relocates one config value.

    Args:
        key: The config key (one of ``PRIOR_KEYS``, ``GRAFT_KEYS``, ``OUTPUT_KEYS``).
        value: The unquoted value.
        old_roots: Prefixes that mark a value as relocatable.
        aliases: Prior directory aliases (see ``load_aliases``).

    Returns:
        The relocated value, or ``value`` unchanged if it is relative, null or
        already under an environment root.

    Raises:
        RelocationError: The value starts with ``/``, ``~`` or ``${`` and is
            under no old root.
    """
    if value.startswith((f"{PRIOR_ROOT}/", f"{CKPT_ROOT}/")):
        return value
    if not value.startswith(("/", "~", "${")):
        return value
    rel = _strip_root(value, old_roots)
    if rel is None:
        raise RelocationError(f"{key}: {value!r} is under none of {old_roots}")
    parts = [part for part in rel.split("/") if part]
    if not parts:
        raise RelocationError(f"{key}: {value!r} has no path below the old root")
    if key in PRIOR_KEYS:
        return f"{PRIOR_ROOT}/{aliases.get(parts[-1], parts[-1])}"
    if key in OUTPUT_KEYS:
        return f"{CKPT_ROOT}/{parts[-1]}"
    archived = _ARCHIVE_NAME.match(parts[-1])
    if archived and len(parts) >= 2 and parts[-2] == "baselines":
        return f"{CKPT_ROOT}/{archived['run']}/{archived['step']}"
    if len(parts) < 2:
        raise RelocationError(f"{key}: {value!r} needs a run and a step directory")
    return f"{CKPT_ROOT}/{parts[-2]}/{parts[-1]}"


def relocate_text(
    text: str, old_roots: list[str], aliases: dict[str, str]
) -> tuple[str, int]:
    """Relocates every path value in a YAML document.

    Args:
        text: The YAML source.
        old_roots: Prefixes that mark a value as relocatable.
        aliases: Prior directory aliases.

    Returns:
        The new text and the number of values that changed.

    Raises:
        RelocationError: A path value cannot be relocated (see
            ``relocate_value``).
    """
    changed = 0
    out = []
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped) :]
        match = _LINE.match(stripped)
        if match:
            value = match["value"]
            quote = value[0] if value[0] in "\"'" else ""
            inner = value[1:-1] if quote else value
            new = relocate_value(match["key"], inner, old_roots, aliases)
            if new != inner:
                changed += 1
                stripped = f"{match['head']}{quote}{new}{quote}{match['tail']}"
        out.append(stripped + newline)
    return "".join(out), changed


def _yaml_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.rglob("*.yaml")) if path.is_dir() else [path])
    return files


def main(argv: list[str] | None = None) -> int:
    """Runs the tool; returns the process exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--old-root",
        action="append",
        required=True,
        help="absolute prefix of the values to relocate (repeatable)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="prior manifest whose alias table maps directory names to ids",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if any value would change",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="YAML files or dirs")
    args = parser.parse_args(argv)

    aliases = load_aliases(args.manifest)
    total = 0
    for file in _yaml_files(args.paths):
        try:
            new_text, changed = relocate_text(file.read_text(), args.old_root, aliases)
        except RelocationError as error:
            print(f"{file}: {error}", file=sys.stderr)
            return 2
        if changed:
            total += changed
            if not args.check:
                file.write_text(new_text)
            print(f"{file}: {changed} value(s)")
    print(f"{total} value(s) {'would change' if args.check else 'relocated'}")
    return 1 if args.check and total else 0


if __name__ == "__main__":
    sys.exit(main())
