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

    def _get_name(self) -> str:
        return 'tanh(%s)' % self.inputs[0].name

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.tanh(self.inputs[0].forward(feed_dict))

    def _backward(self, gradient: Operation) -> None:
        import auto_diff as ad
        self.gradient = (1.0 - ad.square(self)) * gradient
        self.inputs[0].backward(self.gradient)
