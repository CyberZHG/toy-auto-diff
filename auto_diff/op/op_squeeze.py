from typing import Mapping, Union, Optional, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpSqueeze(Operation):
    """Flatten the tensor to 1-D array."""

    def __init__(self, x: Operation, axis: Optional[Union[int, Sequence[int]]] = None, **kwargs):
        self.inputs = [x]
        if axis is None:
            axis = -1
        self.params = {
            'axis': axis,
        }
        try:
            self.backward_axis = list(sorted(set([(a + x.dim) % x.dim for a in axis])))
        except TypeError:
            self.backward_axis = [(axis + x.dim) % x.dim]
        self.shape = tuple(x.shape[i] for i in range(len(x.shape)) if i not in self.backward_axis)
        super(OpSqueeze, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Squeeze the tensor."""
        return self.values[0].squeeze(axis=self.params['axis'])

    def _backward(self, gradient: np.ndarray) -> None:
        """Expand the dimensions of the gradient."""
        self.gradients = [gradient]
        for axis in self.backward_axis:
            self.gradients = [np.expand_dims(self.gradients[0], axis=axis)]
