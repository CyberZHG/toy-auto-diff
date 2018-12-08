from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpConstant(Operation):
    """Contains a constant."""

    def __init__(self, x: Union[int, float, list, np.ndarray], **kwargs):
        """
        :param x: The constant value.
        :param kwargs:
        """
        if isinstance(x, int):
            x = float(x)
        if not np.isscalar(x) and not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float64)
        self.x = x
        if np.isscalar(x):
            self.shape = (1,)
        else:
            self.shape = x.shape
        super(OpConstant, self).__init__(**kwargs)

    def _get_name(self) -> str:
        if np.isscalar(self.x):
            return str(self.x)
        return 'C%s' % str(self.x.shape)

    def _get_op_name(self) -> str:
        return 'c_%d' % self._op_index

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Returns the constant."""
        return self.x

    def _backward(self, gradient: Operation) -> None:
        """No backward operation needed."""
        pass
