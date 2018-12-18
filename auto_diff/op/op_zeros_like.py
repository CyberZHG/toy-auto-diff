from typing import Union, Mapping
import numpy as np
from .operation import Operation


class OpZerosLike(Operation):
    """Constant tensor filled with zeros."""

    def __init__(self, x: Operation, **kwargs):
        self.inputs = [x]
        self.shape = x.shape
        super(OpZerosLike, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return 'zeros_like(%s)' % self.inputs[0].name

    def _get_op_name(self) -> str:
        return 'zeros_like(%s)' % self.inputs[0]._op_name

    def _forward(self, feed_dict: Mapping[Union[str, Operation], np.ndarray]) -> np.ndarray:
        """Generate and returns the constant."""
        return np.zeros_like(self.inputs[0].forward(feed_dict))

    def _backward(self, gradient: Operation) -> None:
        """No backward operation needed."""
        pass
