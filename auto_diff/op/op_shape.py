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

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.array(np.shape(self.inputs[0].forward(feed_dict)))

    def _backward(self, gradient: Operation) -> None:
        import auto_diff as ad
        self.gradients = [ad.zeros_like(self.inputs[0])]
