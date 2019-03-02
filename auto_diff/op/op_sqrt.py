from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpSqrt(Operation):
    """Element-wise square-root."""

    def __init__(self, x: Operation, **kwargs):
        self.inputs = [x]
        self.shape = x.shape
        super(OpSqrt, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.sqrt(self.values[0])

    def _backward(self, gradient: Operation) -> None:
        self.gradients = [1.0 / (2.0 * self.output) * gradient]
