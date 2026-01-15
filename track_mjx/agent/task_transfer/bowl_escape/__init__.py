"""Bowl escape task transfer training pipeline.

Two training modes:
1. decoder_only: Freeze decoder, train new encoder
2. prior_decoder: Freeze prior + decoder, train residual encoder

Note: Do not add imports here that trigger mujoco imports.
The MUJOCO_GL environment variable must be set before mujoco is imported,
which happens in train.py. Adding imports here would cause mujoco to be
imported before train.py can set the environment variable.
"""

__all__ = [
    "DecoderHighLevelWrapper",
    "PriorDecoderHighLevelWrapper",
    "load_prior_checkpoint",
    "make_decoder_inference_fn",
    "make_prior_inference_fn",
]
