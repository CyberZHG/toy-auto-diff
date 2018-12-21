from typing import Mapping, Union, Optional
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder
from .op_constant import OpConstant


class OpWhere(Operation):
    """Conditional selection."""

    def __init__(self, condition: Operation, x: Optional[Operation] = None, y: Optional[Operation] = None, **kwargs):
        if x is None:
            x = OpConstant(1.0)
        if y is None:
            y = OpConstant(0.0)
        self.inputs = [x, y]
        self.params = {
            'condition': condition,
        }
        self._broadcast_shape(condition, x, y)
        super(OpWhere, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.where(
            self.params['condition'].forward(feed_dict).astype(np.bool),
            self.inputs[0].forward(feed_dict),
            self.inputs[1].forward(feed_dict),
        )

    def _backward(self, gradient: Operation) -> None:
        self.gradients = [
            self.inputs[0]._broadcast_backward(OpWhere(self.params['condition'], gradient, OpConstant(0.0))),
            self.inputs[1]._broadcast_backward(OpWhere(self.params['condition'], OpConstant(0.0), gradient)),
        ]
