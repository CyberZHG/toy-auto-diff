from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpTanh(Operation):
    """Element-wise tanh."""

    def __init__(self, x: Operation, **kwargs):
        self.inputs = [x]
        self.shape = x.shape
        super(OpTanh, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.tanh(self.values[0])

    def _backward(self, gradient: np.ndarray) -> None:
        self.gradients = [(1.0 - np.square(self.output)) * gradient]
