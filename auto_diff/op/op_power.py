from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpPower(Operation):
    """Element-wise power."""

    def __init__(self, x: Operation, y: Operation, **kwargs):
        self.inputs = [x, y]
        self._broadcast_shape(x, y)
        super(OpPower, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.power(self.values[0], self.values[1])

    def _backward(self, gradient: np.ndarray) -> None:
        gradient_x = self.values[1] * np.power(self.values[0], self.values[1] - 1.0)
        gradient_y = np.log(self.values[0]) * self.output
        self.gradients = [
            self._broadcast_backward(gradient * gradient_x, np.shape(self.values[0])),
            self._broadcast_backward(gradient * gradient_y, np.shape(self.values[1])),
        ]
