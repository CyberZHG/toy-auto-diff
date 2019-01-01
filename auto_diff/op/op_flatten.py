from typing import Mapping, Union
import numpy as np
from .operation import Operation


class OpFlatten(Operation):
    """Flatten the tensor to 1-D array."""

    def __init__(self, x: Operation, **kwargs):
        self.inputs = [x]
        if any(map(lambda x: x is None, np.shape(x))):
            self.shape = (None,)
        else:
            self.shape = (int(np.prod(np.shape(x))),)
        super(OpFlatten, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, Operation], np.ndarray]):
        return self.values[0].flatten()

    def _backward(self, gradient: np.ndarray):
        self.gradients = [
            gradient.reshape(self.values[0].shape),
        ]
