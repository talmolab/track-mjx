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


def create_bias_mask(params):
    """Creates boolean mask were any leaves under decoder are set to False."""

    def _mask_fn(path, _):
        def f(key):
            try:
                return key.key
            except:
                return key.name

        # Check if any part of the path contains 'decoder'
        return "frozen" if "bias" in [str(f(part)) for part in path] else "learned"

    # Create mask using tree_map_with_path
    return jax.tree_util.tree_map_with_path(lambda path, _: _mask_fn(path, _), params)
