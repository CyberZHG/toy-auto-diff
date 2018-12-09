from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpSubtract(Operation):
    """Element-wise subtract."""

    def __init__(self, x: Operation, y: Operation, **kwargs):
        self.inputs = [x, y]
        self._broadcast_shape(x, y)
        super(OpSubtract, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return '(%s - %s)' % (self.inputs[0].name, self.inputs[1].name)

    def _get_op_name(self) -> str:
        return 'subtract(%s, %s)' % (self.inputs[0]._op_name, self.inputs[1]._op_name)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return self.inputs[0].forward(feed_dict) - self.inputs[1].forward(feed_dict)

    def _backward(self, gradient: Operation) -> None:
        self.gradient = [
            self.inputs[0]._broadcast_backward(gradient),
            -self.inputs[1]._broadcast_backward(gradient),
        ]
        self.inputs[0].backward(self.gradient[0])
        self.inputs[1].backward(self.gradient[1])
