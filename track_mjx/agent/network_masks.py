import jax
from track_mjx.agent.mlp_ppo.losses import PPONetworkParams
import copy


def create_decoder_mask(params, decoder_name="decoder"):
    """Creates mask where the depth of nodes that contains 'decoder' becomes leaves, and decoder is set to frozen, and the rest to learned."""

    param_mask = copy.deepcopy(params)
    for key in param_mask.policy["params"]:
        if key == decoder_name:
            param_mask.policy["params"][key] = True
        else:
            param_mask.policy["params"][key] = False

    for key in param_mask.value:
        param_mask.value[key] = False

    return param_mask
