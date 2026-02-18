"""Recurrent PPO implementation with VAE-style intention encoding.

This module provides a recurrent variant of the intention-based PPO algorithm
where the decoder uses an RNN (SimpleCell, GRU, or LSTM) instead of a
feedforward MLP. The encoder remains an MLP that maps trajectory observations
to latent intentions.
"""

from track_mjx.agent.recurrent_ppo.losses import (
    RecurrentPPONetworkParams,
    compute_recurrent_ppo_loss,
)
from track_mjx.agent.recurrent_ppo.networks import (
    RecurrentDecoder,
    RecurrentIntentionNetwork,
    RecurrentNetwork,
    RecurrentPPONetworks,
    get_rnn_cell,
    init_hidden_state,
    make_dict_value_network,
    make_inference_fn,
    make_logging_inference_fn,
    make_recurrent_intention_ppo_networks,
    reset_hidden_on_done,
)
from track_mjx.agent.recurrent_ppo.ppo import (
    RecurrentEvaluator,
    TrainingState,
    actor_step_rnn,
    generate_unroll_rnn,
    train,
)
from track_mjx.agent.recurrent_ppo.recurrent_vision_losses import (
    compute_recurrent_shared_vision_ppo_loss,
)
from track_mjx.agent.recurrent_ppo.recurrent_vision_networks import (
    RecurrentSharedVisionModule,
    make_recurrent_vision_highlvl_ppo_networks,
)

__all__ = [
    # losses
    "RecurrentPPONetworkParams",
    "compute_recurrent_ppo_loss",
    # networks
    "RecurrentDecoder",
    "RecurrentIntentionNetwork",
    "RecurrentNetwork",
    "RecurrentPPONetworks",
    "get_rnn_cell",
    "init_hidden_state",
    "make_dict_value_network",
    "make_inference_fn",
    "make_logging_inference_fn",
    "make_recurrent_intention_ppo_networks",
    "reset_hidden_on_done",
    # ppo training
    "RecurrentEvaluator",
    "TrainingState",
    "actor_step_rnn",
    "generate_unroll_rnn",
    "train",
    # recurrent vision losses
    "compute_recurrent_shared_vision_ppo_loss",
    # recurrent vision networks
    "RecurrentSharedVisionModule",
    "make_recurrent_vision_highlvl_ppo_networks",
]
