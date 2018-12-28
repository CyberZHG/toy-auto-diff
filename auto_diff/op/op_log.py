from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpLog(Operation):
    """Element-wise log (ln)."""

    def __init__(self, x: Operation, **kwargs):
        self.inputs = [x]
        self.shape = x.shape
        super(OpLog, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.log(self.values[0])

    def _backward(self, gradient: np.ndarray) -> None:
        self.gradients = [gradient / self.values[0]]
