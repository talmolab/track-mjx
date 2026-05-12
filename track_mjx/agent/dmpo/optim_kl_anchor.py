"""Asymmetric-LR optimizer for the DMPO kl-anchor mode.

Builds a `policy_opt` via `optax.multi_transform` that applies different
Adam learning rates to {prior, decoder, policy_head} parameter blocks,
plus a separate critic_opt and dual_opt (mirroring make_optimizers in
learner.py).
"""
from __future__ import annotations

from typing import Tuple

import jax
import optax

from track_mjx.agent.dmpo.config import DMPOConfig

_VALID_LABELS = ("prior", "decoder", "policy_head", "other")


def label_param_tree(params):
    """Map each leaf in the policy param tree to one of {prior, decoder,
    policy_head, other} based on its top-level block name under
    `params["params"]`.
    """
    inner = params["params"] if "params" in params else params

    def _label_block(block_name):
        if block_name in _VALID_LABELS:
            return block_name
        return "other"

    labeled_inner = {}
    for block_name, block in inner.items():
        lbl = _label_block(block_name)
        labeled_inner[block_name] = jax.tree_util.tree_map(lambda _, _lbl=lbl: _lbl, block)
    if "params" in params:
        return {"params": labeled_inner}
    return labeled_inner


def make_kl_anchor_optimizers(
    cfg: DMPOConfig,
    prior_lr_mult: float = 0.1,
    decoder_lr_mult: float = 1.0,
    policy_head_lr_mult: float = 1.0,
    other_lr_mult: float = 1.0,
) -> Tuple[optax.GradientTransformation, optax.GradientTransformation, optax.GradientTransformation]:
    base_lr = cfg.policy_lr

    def _block_chain(lr_mult):
        return optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adam(base_lr * lr_mult),
        )

    transforms = {
        "prior": _block_chain(prior_lr_mult),
        "decoder": _block_chain(decoder_lr_mult),
        "policy_head": _block_chain(policy_head_lr_mult),
        "other": _block_chain(other_lr_mult),
    }

    policy_opt = optax.multi_transform(transforms, label_param_tree)

    critic_opt = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adam(cfg.critic_lr),
    )
    dual_opt = optax.adam(cfg.dual_lr)

    return policy_opt, critic_opt, dual_opt
