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

    def _get_name(self) -> str:
        return 'log(%s)' % self.inputs[0].name

    def _get_op_name(self) -> str:
        return 'log(%s)' % self.inputs[0]._op_name

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.log(self.inputs[0].forward(feed_dict))

    def _backward(self, gradient: Operation) -> None:
        self.gradient = gradient / self.inputs[0]
        self.inputs[0].backward(self.gradient)
