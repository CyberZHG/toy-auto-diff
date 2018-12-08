from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpFlatten(Operation):
    """Flatten the tensor to 1-D array."""

    def __init__(self, x: Operation, **kwargs):
        self.x = x
        self.shape = (np.prod(x.shape),)
        self.old_shape = x.shape
        super(OpFlatten, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return 'flatten(%s)' % self.x.name

    def _get_op_name(self) -> str:
        return 'flatten(%s)' % self.x._op_name

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Flatten the tensor."""
        return self.x.forward(feed_dict).flatten()

    def _backward(self, gradient: Operation) -> None:
        """Reshape the gradient to its old shape."""
        self.gradient = gradient.reshape(self.old_shape)
        self.x.backward(self.gradient)
