from typing import Mapping, Union, Optional
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder
from .op_constant import OpConstant


class OpWhere(Operation):
    """Conditional selection."""

    def __init__(self, condition: Operation, x: Optional[Operation] = None, y: Optional[Operation] = None, **kwargs):
        if x is None:
            x = OpConstant(1.0)
        if y is None:
            y = OpConstant(0.0)
        self.inputs = [x, y]
        self.params = {
            'condition': condition,
        }
        self.output_condition = None
        self._broadcast_shape(condition, x, y)
        super(OpWhere, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        self.output_condition = self.params['condition'].forward(feed_dict).astype(np.bool)
        return np.where(self.output_condition, self.values[0], self.values[1])

    def _backward(self, gradient: np.ndarray) -> None:
        self.gradients = [
            self._broadcast_backward(np.where(self.output_condition, gradient, 0.0), np.shape(self.values[0])),
            self._broadcast_backward(np.where(self.output_condition, 0.0, gradient), np.shape(self.values[1])),
        ]
