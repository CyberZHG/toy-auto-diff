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
        if np.isscalar(x):
            self.x = float(x)
            self.shape = ()
        else:
            self.x = np.array(x, dtype=np.float64)
            self.shape = self.x.shape
            self.params = {
                'shape': self.shape,
            }
        super(OpConstant, self).__init__(**kwargs)

    @property
    def name(self) -> str:
        if np.isscalar(self.x):
            return str(self.x)
        return super(OpConstant, self).name

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Returns the constant."""
        return self.x

    def _backward(self, gradient: np.ndarray) -> None:
        """No backward operation needed."""
        pass
