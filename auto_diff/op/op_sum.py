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
            return self.inputs[0].forward(feed_dict)
        return np.sum(self.inputs[0].forward(feed_dict), axis=self.params['axis'], keepdims=True)

    def _backward(self, gradient: Operation) -> None:
        """Expand the dimensions of the gradient."""
        from .op_ones_like import OpOnesLike
        if not self.params['keepdims']:
            self.gradients = [gradient]
            return
        self.gradients = [OpOnesLike(self.inputs[0]) * gradient]
