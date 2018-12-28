from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpMaximum(Operation):
    """Element-wise maximum."""

    def __init__(self, x: Operation, y: Operation, **kwargs):
        self.inputs = [x, y]
        self._broadcast_shape(x, y)
        super(OpMaximum, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.maximum(self.values[0], self.values[1])

    def _backward(self, gradient: np.ndarray) -> None:
        self.gradients = [
            self._broadcast_backward(np.equal(self.output, self.values[0]) * gradient, np.shape(self.values[0])),
            self._broadcast_backward(np.equal(self.output, self.values[1]) * gradient, np.shape(self.values[1])),
        ]
