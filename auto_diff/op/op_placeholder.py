from typing import Mapping, Union, Sequence
import numpy as np
from .operation import Operation


class OpPlaceholder(Operation):
    """The placeholder that represents values to be feed."""

    def __init__(self, shape: Sequence[int], **kwargs):
        """
        :param shape: Shape of the value.
        :param kwargs:
        """
        self.params = {
            'shape': shape,
        }
        self.shape = tuple(shape)
        super(OpPlaceholder, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, 'OpPlaceholder'], np.ndarray]):
        """Finds and returns the value in feed dictionary."""
        return feed_dict[self]

    def _backward(self, gradient: Operation) -> None:
        """No backward operation needed."""
        pass
