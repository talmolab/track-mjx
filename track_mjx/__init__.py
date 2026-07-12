"""This module exposes all high level APIs for track-mjx"""

from track_mjx.device_utils import enable_jit_cache, patch_brax_pmap_compat
from track_mjx.version import __version__ as __version__

enable_jit_cache()
patch_brax_pmap_compat()
