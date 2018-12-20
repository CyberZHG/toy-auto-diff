from typing import Mapping, Union, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpReshape(Operation):
    """Reshape the tensor to a given shape."""

    def __init__(self, x: Operation, shape: Sequence[int], **kwargs):
        self.inputs = [x]
        rest, fill_index = 1, -1
        for index, dim in enumerate(shape):
            if dim == -1:
                if fill_index != -1:
                    raise ValueError('Only one dimension could be undefined, found %s' % str(shape))
                fill_index = index
            else:
                rest *= dim
        if fill_index != -1:
            shape = list(shape)
            shape[fill_index] = np.prod(x.shape, dtype=np.int) // rest
        self.shape = tuple(shape)
        self.old_shape = x.shape
        super(OpReshape, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return 'reshape(%s, shape=%s)' % (self.inputs[0].name, str(self.shape))

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Reshape the tensor."""
        return np.reshape(self.inputs[0].forward(feed_dict), newshape=self.shape)

    def _backward(self, gradient: Operation) -> None:
        """Reshape the gradient to its old shape."""
        self.gradient = gradient.reshape(shape=self.old_shape)
        self.inputs[0].backward(self.gradient)
