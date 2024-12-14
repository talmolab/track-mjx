from brax.training.acme import running_statistics
from brax.training.acme import types
from flax import nnx


class NormalizeObservations(nnx.Module):
    """Module for normalizing observations."""

    _state: nnx.Variable[running_statistics.RunningStatisticsState]

    def __init__(self, specs: types.NestedArray):
        """Initializes the module."""
        self._state = nnx.Variable(running_statistics.init_state(specs))

    def __call__(self, x: types.NestedArray) -> types.Nest:
        """Normalize the input."""
        return running_statistics.normalize(x, self._state)

    def update(self, x: types.Nest):
        """Update the running statistics."""
        self._state = nnx.Variable(running_statistics.update(self._state, x))


class Identity(nnx.Module):
    """Module for doing nothing."""

    def __init__(self):
        pass

    def __call__(self, x: types.Nest) -> types.Nest:
        return x

    def update(self, x: types.Nest):
        pass
