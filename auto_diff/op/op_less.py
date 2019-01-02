from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpLess(Operation):
    """Element-wise less."""

    def __init__(self, x: Operation, y: Operation, **kwargs):
        self.inputs = [x, y]
        self._broadcast_shape(x, y)
        super(OpLess, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        result = self.values[0] < self.values[1]
        if np.isscalar(result):
            result = float(result)
        return result

    def _backward(self, gradient: np.ndarray) -> None:
        raise NotImplementedError('`less` is not differentiable')
