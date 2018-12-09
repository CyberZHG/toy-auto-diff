from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpNegative(Operation):
    """Element-wise numerical negative."""

    def __init__(self, x: Operation, **kwargs):
        self.inputs = [x]
        self.shape = x.shape
        super(OpNegative, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return '(-%s)' % self.inputs[0].name

    def _get_op_name(self) -> str:
        return 'negative(%s)' % self.inputs[0]._op_name

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return -self.inputs[0].forward(feed_dict)

    def _backward(self, gradient: Operation) -> None:
        self.gradient = -gradient
        self.inputs[0].backward(self.gradient)
