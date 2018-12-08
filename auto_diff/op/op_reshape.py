from typing import Mapping, Union, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpReshape(Operation):
    """Reshape the tensor to a given shape."""

    def __init__(self, x: Operation, shape: Sequence[int], **kwargs):
        self.x = x
        self.shape = shape
        self.old_shape = x.shape
        super(OpReshape, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return 'reshape(%s, shape=%s)' % (self.x.name, str(self.shape))

    def _get_op_name(self) -> str:
        return 'reshape(%s, shape=%s)' % (self.x._op_name, str(self.shape))

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Reshape the tensor."""
        return np.reshape(self.x.forward(feed_dict), newshape=self.shape)

    def _backward(self, gradient: Operation) -> None:
        """Reshape the gradient to its old shape."""
        self.gradient = gradient.reshape(shape=self.old_shape)
        self.x.backward(self.gradient)
