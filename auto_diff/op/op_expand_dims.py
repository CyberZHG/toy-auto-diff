from typing import Mapping, Union, Optional
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpExpandDims(Operation):
    """Expand the dimensions of the tensor."""

    def __init__(self, x: Operation, axis: Optional[int] = None, **kwargs):
        self.inputs = [x]
        if axis is None:
            axis = -1
        self.axis = axis
        if axis < 0:
            axis += len(x.shape) + 1
        self.shape = x.shape[:axis] + (1,) + x.shape[axis:]
        super(OpExpandDims, self).__init__(**kwargs)

    def _get_name(self) -> str:
        if self.axis == -1:
            return 'expand_dims(%s)' % self.inputs[0].name
        return 'expand_dims(%s, axis=%d)' % (self.inputs[0].name, self.axis)

    def _get_op_name(self) -> str:
        if self.axis == -1:
            return 'expand_dims(%s)' % self.inputs[0]._op_name
        return 'expand_dims(%s, axis=%d)' % (self.inputs[0]._op_name, self.axis)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Expand shape."""
        return np.expand_dims(self.inputs[0].forward(feed_dict), axis=self.axis)

    def _backward(self, gradient: Operation) -> None:
        """Squeeze the expanded dimension."""
        self.gradient = gradient.squeeze(axis=self.axis)
        self.inputs[0].backward(self.gradient)
