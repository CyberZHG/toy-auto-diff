from typing import Mapping, Union, Optional
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpArgmax(Operation):
    """Returns the indices of the maximum values along an axis."""

    def __init__(self, x: Operation, axis: Optional[int] = None, **kwargs):
        self.inputs = [x]
        self.axis = axis
        if axis is None:
            self.shape = ()
        else:
            shape = list(x.shape)
            del shape[axis]
            self.shape = tuple(shape)
        super(OpArgmax, self).__init__(**kwargs)

    def _get_name(self) -> str:
        if self.axis is None:
            return 'argmax(%s)' % self.inputs[0].name
        return 'argmax(%s, axis=%d)' % (self.inputs[0].name, self.axis)

    def _get_op_name(self) -> str:
        if self.axis is None:
            return 'argmax(%s)' % self.inputs[0]._op_name
        return 'argmax(%s, axis=%d)' % (self.inputs[0]._op_name, self.axis)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.argmax(self.inputs[0].forward(feed_dict), axis=self.axis)

    def _backward(self, gradient: Operation) -> None:
        raise NotImplementedError('`argmax` is not differentiable')
