from typing import Union, Mapping, Sequence
import numpy as np
from .operation import Operation


class OpOnes(Operation):
    """Constant tensor filled with ones."""

    def __init__(self, shape: Union[int, Sequence[int]], **kwargs):
        if isinstance(shape, int):
            self.shape = shape
        else:
            self.shape = tuple(shape)
        super(OpOnes, self).__init__(**kwargs)

    def _get_name(self) -> str:
        if isinstance(self.shape, int):
            return 'ones(%d)' % self.shape
        return 'ones%s' % str(self.shape)

    def _forward(self, feed_dict: Mapping[Union[str, Operation], np.ndarray]) -> np.ndarray:
        """Generate and returns the constant."""
        return np.ones(self.shape)

    def _backward(self, gradient: Operation) -> None:
        """No backward operation needed."""
        pass
