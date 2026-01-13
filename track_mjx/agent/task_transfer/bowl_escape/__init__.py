"""Bowl escape task transfer training pipeline.

Two training modes:
1. decoder_only: Freeze decoder, train new encoder
2. prior_decoder: Freeze prior + decoder, train residual encoder
"""

from track_mjx.agent.task_transfer.bowl_escape.wrappers import (
    DecoderHighLevelWrapper,
    PriorDecoderHighLevelWrapper,
)
from track_mjx.agent.task_transfer.bowl_escape.checkpoint_utils import (
    load_prior_checkpoint,
    make_decoder_inference_fn,
    make_prior_inference_fn,
)

__all__ = [
    "DecoderHighLevelWrapper",
    "PriorDecoderHighLevelWrapper",
    "load_prior_checkpoint",
    "make_decoder_inference_fn",
    "make_prior_inference_fn",
]
