from typing import Mapping, Union, Optional, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder
from .op_keepdims import OpKeepdims


class OpSum(OpKeepdims):
    """Sum of elements over a given axis."""

    def __init__(self,
                 x: Operation,
                 axis: Optional[Union[int, Sequence[int]]] = None,
                 keepdims: bool = False,
                 **kwargs):
        super(OpSum, self).__init__(self.__class__, x, axis, keepdims, **kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Sum over axis"""
        if not self.params['keepdims']:
            return self.values[0]
        return np.sum(self.values[0], axis=self.params['axis'], keepdims=True)

    def _backward(self, gradient: np.ndarray) -> None:
        """Expand the dimensions of the gradient."""
        if not self.params['keepdims']:
            self.gradients = [gradient]
            return
        self.gradients = [np.ones_like(self.values[0]) * gradient]
