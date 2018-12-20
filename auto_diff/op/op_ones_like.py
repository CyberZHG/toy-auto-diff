from typing import Union, Mapping
import numpy as np
from .operation import Operation


class OpOnesLike(Operation):
    """Constant tensor filled with ones."""

    def __init__(self, x: Operation, **kwargs):
        self.inputs = [x]
        self.shape = x.shape
        super(OpOnesLike, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, Operation], np.ndarray]) -> np.ndarray:
        """Generate and returns the constant."""
        return np.ones_like(self.inputs[0].forward(feed_dict))

    def _backward(self, gradient: Operation) -> None:
        """No backward operation needed."""
        import auto_diff as ad
        self.gradients = [ad.zeros_like(self)]
