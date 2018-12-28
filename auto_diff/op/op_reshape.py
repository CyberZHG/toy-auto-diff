from typing import Mapping, Union, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpReshape(Operation):
    """Reshape the tensor to a given shape."""

    def __init__(self, x: Operation, shape: Sequence[int], **kwargs):
        self.inputs = [x]
        self.params = {
            'shape': shape,
        }
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

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Reshape the tensor."""
        return np.reshape(self.values[0], newshape=self.shape)

    def _backward(self, gradient: np.ndarray) -> None:
        """Reshape the gradient to its old shape."""
        self.gradients = [np.reshape(gradient, self.old_shape)]
