from typing import Mapping, Union, Optional, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpSqueeze(Operation):
    """Flatten the tensor to 1-D array."""

    def __init__(self, x: Operation, axis: Optional[Union[int, Sequence[int]]], **kwargs):
        self.inputs = [x]
        if axis is None:
            axis = -1
        self.axis = axis
        try:
            self.backward_axis = list(sorted(set([(a + len(x.shape)) % len(x.shape) for a in axis])))
        except TypeError:
            self.backward_axis = [axis]
        self.shape = tuple(x.shape[i] for i in range(len(x.shape)) if i not in self.backward_axis)
        super(OpSqueeze, self).__init__(**kwargs)

    def _get_name(self) -> str:
        if self.axis == -1:
            return 'squeeze(%s)' % self.inputs[0].name
        return 'squeeze(%s, axis=%s)' % (self.inputs[0].name, str(self.axis))

    def _get_op_name(self) -> str:
        if self.axis == -1:
            return 'squeeze(%s)' % self.inputs[0]._op_name
        return 'squeeze(%s, axis=%s)' % (self.inputs[0]._op_name, str(self.axis))

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Squeeze the tensor."""
        return self.inputs[0].forward(feed_dict).squeeze(axis=self.axis)

    def _backward(self, gradient: Operation) -> None:
        """Expand the dimensions of the gradient."""
        self.gradient = gradient
        for axis in self.backward_axis:
            self.gradient = self.gradient.expand_dims(axis=axis)
        self.inputs[0].backward(self.gradient)
