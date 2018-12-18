from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpSetitem(Operation):
    """Get item based on indexing"""

    def __init__(self, x: Operation, key, value: Operation, **kwargs):
        self.inputs = [x, value]
        self.key = key
        self.shape = x.shape
        super(OpSetitem, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return 'setitem(%s, %s, %s)' % (self.inputs[0].name, str(self.key), self.inputs[1].name)

    def _get_op_name(self) -> str:
        return 'setitem(%s, %s, %s)' % (self.inputs[0]._op_name, str(self.key), self.inputs[1]._op_name)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        val = self.inputs[0].forward(feed_dict)
        val[self.key] = self.inputs[1].forward(feed_dict)
        return val

    def _backward(self, gradient: Operation) -> None:
        raise NotImplementedError('`setitem` is not differentiable')
