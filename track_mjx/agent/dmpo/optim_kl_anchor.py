"""Asymmetric-LR optimizer for the DMPO kl-anchor mode.

Builds a `policy_opt` via `optax.multi_transform` that applies different
Adam learning rates to {prior, decoder, policy_head} parameter blocks,
plus a separate critic_opt and dual_opt (mirroring make_optimizers in
learner.py).

The policy gradient-norm clip follows `cfg.policy_grad_clip_scope`: "global"
(acme) clips the whole trainable policy gradient by one norm before the
per-block Adam; "per_block" clips each block by its own norm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

if TYPE_CHECKING:
    from track_mjx.agent.dmpo.config import DMPOConfig

_VALID_LABELS = ("prior", "decoder", "policy_head", "other")


def label_param_tree(params):
    """Map each leaf in the policy param tree to one of {prior, decoder,
    policy_head, other} based on its top-level block name under
    `params["params"]`.
    """
    inner = params.get("params", params)

    def _label_block(block_name):
        if block_name in _VALID_LABELS:
            return block_name
        return "other"

    labeled_inner = {}
    for block_name, block in inner.items():
        lbl = _label_block(block_name)
        labeled_inner[block_name] = jax.tree.map(lambda _, _lbl=lbl: _lbl, block)
    if "params" in params:
        return {"params": labeled_inner}
    return labeled_inner


def clip_by_trainable_global_norm(
    max_norm: float, trainable_mask_fn
) -> optax.GradientTransformation:
    """``optax.clip_by_global_norm`` over the trainable leaves only.

    ``trainable_mask_fn(updates)`` returns a pytree of Python bools with the
    structure of ``updates`` (True = trainable). The global norm is taken over
    the trainable leaves, and only those are scaled, with the same rule as
    optax (``t / norm * max_norm`` when the norm is not below ``max_norm``).
    Frozen leaves pass through untouched and never enter the norm, so a NaN
    gradient in a frozen block cannot poison the trainable update; the
    frozen block's own transform (``set_to_zero``) then discards it.
    """

    def update_fn(updates, state, params=None):
        del params
        mask = trainable_mask_fn(updates)
        leaves, treedef = jax.tree_util.tree_flatten(updates)
        flags = treedef.flatten_up_to(mask)
        trainable = [u for u, keep in zip(leaves, flags) if keep]
        if not trainable:
            return updates, state
        g_norm = optax.tree.norm(trainable)
        trigger = jnp.squeeze(g_norm < max_norm)

        def clip(t):
            return jax.lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * max_norm)

        out = [clip(u) if keep else u for u, keep in zip(leaves, flags)]
        return jax.tree_util.tree_unflatten(treedef, out), state

    return optax.GradientTransformation(optax.init_empty_state, update_fn)


def make_kl_anchor_optimizers(
    cfg: DMPOConfig,
    prior_lr_mult: float = 0.1,
    decoder_lr_mult: float = 1.0,
    policy_head_lr_mult: float = 1.0,
    other_lr_mult: float = 1.0,
) -> tuple[
    optax.GradientTransformation,
    optax.GradientTransformation,
    optax.GradientTransformation,
]:
    base_lr = cfg.policy_lr
    scope = getattr(cfg, "policy_grad_clip_scope", "global")
    if scope not in ("global", "per_block"):
        raise ValueError(
            f"unknown policy_grad_clip_scope={scope!r}; valid values: "
            "'global', 'per_block'"
        )
    lr_mults = {
        "prior": prior_lr_mult,
        "decoder": decoder_lr_mult,
        "policy_head": policy_head_lr_mult,
        "other": other_lr_mult,
    }

    def _block_chain(lr_mult):
        # A true freeze, not adam(0.0).
        #
        # adam(base_lr * 0.0) zeroes the update by MULTIPLICATION, which is an
        # exact freeze only while every gradient is finite. It is not NaN-safe:
        # clip_by_global_norm computes the global norm over this masked branch
        # alone, so one NaN gradient anywhere in the block makes that norm NaN,
        # divides every leaf by it, and then 0 * NaN = NaN -- silently turning
        # the whole "frozen" prior or decoder into NaN mid-run and destroying
        # the warm-start the design depends on.
        #
        # optax.set_to_zero() emits exact zeros unconditionally, carries no
        # state, and therefore also avoids accumulating Adam mu/nu moments in a
        # block that never moves (2x param memory per frozen block, written
        # into every checkpoint, and a full-magnitude jolt if the multiplier is
        # ever un-zeroed on resume).
        if lr_mult == 0.0:
            return optax.set_to_zero()
        if scope == "global":
            # Clipped once, jointly, before the blocks (below).
            return optax.adam(base_lr * lr_mult)
        return optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adam(base_lr * lr_mult),
        )

    transforms = {label: _block_chain(mult) for label, mult in lr_mults.items()}

    policy_opt = optax.multi_transform(transforms, label_param_tree)
    if scope == "global":
        # One norm over every trainable block, as acme clips the whole policy
        # gradient. Frozen blocks (lr_mult 0) stay out of the norm.
        def _trainable(tree):
            return jax.tree.map(
                lambda label: lr_mults[label] != 0.0, label_param_tree(tree)
            )

        policy_opt = optax.chain(
            clip_by_trainable_global_norm(cfg.grad_clip, _trainable), policy_opt
        )

    critic_opt = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adam(cfg.critic_lr),
    )
    dual_opt = optax.adam(cfg.dual_lr)

    return policy_opt, critic_opt, dual_opt
