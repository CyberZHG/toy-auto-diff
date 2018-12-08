from typing import Mapping, Union
import numpy as np
from .operation import Operation


class OpPlaceholder(Operation):
    """The placeholder that represents values to be feed."""

    def __init__(self, shape: tuple, **kwargs):
        """
        :param shape: Shape of the value.
        :param kwargs:
        """
        self.shape = shape
        super(OpPlaceholder, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return 'X%s' % str(self.shape)

    def _get_op_name(self) -> str:
        return 'x_%d' % self._op_index

    def _forward(self, feed_dict: Mapping[Union[str, 'OpPlaceholder'], np.ndarray]):
        """Finds and returns the value in feed dictionary."""
        return feed_dict[self]

    def _backward(self, gradient: Operation) -> None:
        """No backward operation needed."""
        pass
