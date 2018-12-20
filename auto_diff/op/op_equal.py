from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpEqual(Operation):
    """Element-wise equal."""

    def __init__(self, x: Operation, y: Operation, **kwargs):
        self.inputs = [x, y]
        self._broadcast_shape(x, y)
        super(OpEqual, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return (self.inputs[0].forward(feed_dict) == self.inputs[1].forward(feed_dict)).astype(dtype=np.float64)

    def _backward(self, gradient: Operation) -> None:
        raise NotImplementedError('`equal` is not differentiable')
