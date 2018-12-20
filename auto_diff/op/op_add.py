from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpAdd(Operation):
    """Element-wise addition."""

    def __init__(self, x: Operation, y: Operation, **kwargs):
        self.inputs = [x, y]
        self._broadcast_shape(x, y)
        super(OpAdd, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return self.inputs[0].forward(feed_dict) + self.inputs[1].forward(feed_dict)

    def _backward(self, gradient: Operation) -> None:
        self.gradients = [
            self.inputs[0]._broadcast_backward(gradient),
            self.inputs[1]._broadcast_backward(gradient),
        ]
