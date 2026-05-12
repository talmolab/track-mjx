"""Deprecated: use track_mjx.train_dmpo (imitation) or
vnl_playground.train_dmpo (downstream).

This module was the monolithic DMPO entry that handled both imitation
and gap-vision tasks. It was split (Phase 1 of plan
2026-05-04-3-dmpo-script-split):

  - track_mjx.train_dmpo                 — rodent imitation only
  - vnl_playground.train_dmpo            — VNL downstream tasks
                                           (gap, vision, gap-jump-trial)

Both new entries call into ``track_mjx.agent.dmpo.training_loop.run``.

Importing this module emits a DeprecationWarning. Running it as a script
(``python -m track_mjx.agent.dmpo.train_dmpo``) prints the migration
note and exits with status 1.
"""
from __future__ import annotations

import sys
import warnings


_MIGRATION_MSG = (
    "track_mjx.agent.dmpo.train_dmpo is deprecated.\n"
    "  - For imitation:  python -m track_mjx.train_dmpo "
    "--config-name=rodent-dmpo-imitation\n"
    "  - For downstream: python -m vnl_playground.train_dmpo "
    "--config-name=<your-config>\n"
    "See ClaudeCode_PromptHistory/2026-05-04-3-dmpo-script-split/plan.md."
)


warnings.warn(_MIGRATION_MSG, DeprecationWarning, stacklevel=2)


def main(*_args, **_kwargs):
    print(_MIGRATION_MSG, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
