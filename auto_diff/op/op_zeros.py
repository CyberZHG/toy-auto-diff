from typing import Union, Mapping, Sequence
import numpy as np
from .operation import Operation


class OpZeros(Operation):
    """Constant tensor filled with zeros."""

    def __init__(self, shape: Union[int, Sequence[int]], **kwargs):
        self.params = {
            'shape': shape,
        }
        if not isinstance(shape, int):
            shape = tuple(shape)
        self.shape = shape
        super(OpZeros, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, Operation], np.ndarray]) -> np.ndarray:
        """Generate and returns the constant."""
        return np.zeros(self.shape)

    def _backward(self, gradient: Operation) -> None:
        """No backward operation needed."""
        pass
