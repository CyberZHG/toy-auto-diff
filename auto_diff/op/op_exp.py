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

    def _get_name(self) -> str:
        return 'exp(%s)' % self.inputs[0].name

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.exp(self.inputs[0].forward(feed_dict))

    def _backward(self, gradient: Operation) -> None:
        self.gradient = OpExp(self.inputs[0]) * gradient
        self.inputs[0].backward(self.gradient)
