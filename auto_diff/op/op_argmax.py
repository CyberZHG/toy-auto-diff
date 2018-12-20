from typing import Mapping, Union, Optional
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpArgmax(Operation):
    """Returns the indices of the maximum values along an axis."""

    def __init__(self, x: Operation, axis: Optional[int] = None, **kwargs):
        self.inputs = [x]
        self.params = {
            'axis': axis,
        }
        if axis is None:
            self.shape = ()
        else:
            shape = list(x.shape)
            del shape[axis]
            self.shape = tuple(shape)
        super(OpArgmax, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.argmax(self.inputs[0].forward(feed_dict), axis=self.params['axis'])

    def _backward(self, gradient: Operation) -> None:
        raise NotImplementedError('`argmax` is not differentiable')
