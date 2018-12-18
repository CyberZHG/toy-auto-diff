from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpShape(Operation):
    """Get shape of the operation."""

    def __init__(self, x: Operation, **kwargs):
        self.inputs = [x]
        self.shape = (x.dim,)
        super(OpShape, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return 'shape(%s)' % self.inputs[0].name

    def _get_op_name(self) -> str:
        return 'shape(%s)' % self.inputs[0]._op_name

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.array(np.shape(self.inputs[0].forward(feed_dict)))

    def _backward(self, gradient: Operation) -> None:
        raise NotImplementedError('`shape` is not differentiable')
