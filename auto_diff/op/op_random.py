from typing import Union, Mapping, Sequence
import numpy as np
from .operation import Operation


class OpRandom(Operation):
    """Constant tensor filled with random values."""

    def __init__(self, shape: Union[int, Sequence[int]], **kwargs):
        self.params = {
            'shape': shape,
        }
        if not isinstance(shape, int):
            shape = tuple(shape)
        self.shape = shape
        super(OpRandom, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, Operation], np.ndarray]) -> np.ndarray:
        """Generate and returns the constant."""
        return np.random.random(self.shape)

    def _backward(self, gradient: np.ndarray) -> None:
        """No backward operation needed."""
        pass
