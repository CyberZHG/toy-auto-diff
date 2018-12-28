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
        self.params = {
            'axis': axis,
        }
        if axis < 0:
            axis += len(x.shape) + 1
        self.shape = x.shape[:axis] + (1,) + x.shape[axis:]
        super(OpExpandDims, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Expand shape."""
        return np.expand_dims(self.values[0], axis=self.params['axis'])

    def _backward(self, gradient: np.ndarray) -> None:
        """Squeeze the expanded dimension."""
        self.gradients = [np.squeeze(gradient, axis=self.params['axis'])]
