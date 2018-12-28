from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpDivide(Operation):
    """Element-wise divide."""

    def __init__(self, x: Operation, y: Operation, **kwargs):
        self.inputs = [x, y]
        self._broadcast_shape(x, y)
        super(OpDivide, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return self.values[0] / self.values[1]

    def _backward(self, gradient: np.ndarray) -> None:
        x, y = self.values
        self.gradients = [
            self._broadcast_backward(gradient / y, np.shape(x)),
            self._broadcast_backward(-gradient * x / np.square(y), np.shape(y)),
        ]
