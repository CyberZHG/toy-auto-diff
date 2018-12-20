from typing import Union, Mapping, Sequence
import numpy as np
from .operation import Operation


class OpZeros(Operation):
    """Constant tensor filled with zeros."""

    def __init__(self, shape: Union[int, Sequence[int]], **kwargs):
        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)
        super(OpZeros, self).__init__(**kwargs)

    def _get_name(self) -> str:
        if len(self.shape) == 1:
            return 'zeros(%d)' % self.shape[0]
        return 'zeros%s' % str(self.shape)

    def _forward(self, feed_dict: Mapping[Union[str, Operation], np.ndarray]) -> np.ndarray:
        """Generate and returns the constant."""
        return np.zeros(self.shape)

    def _backward(self, gradient: Operation) -> None:
        """No backward operation needed."""
        pass
