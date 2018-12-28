from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpExp(Operation):
    """Element-wise exp."""

    def __init__(self, x: Operation, **kwargs):
        self.inputs = [x]
        self.shape = x.shape
        super(OpExp, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.exp(self.values[0])

    def _backward(self, gradient: np.ndarray) -> None:
        self.gradients = [gradient * np.exp(self.values[0])]
