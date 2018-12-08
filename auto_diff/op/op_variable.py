from typing import Mapping, Union, Callable
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpVariable(Operation):
    """Contains weights that could be updated."""

    def __init__(self, initializer: Union[Callable, int, float, list, np.ndarray], shape: tuple = None, **kwargs):
        """
        :param initializer: A function that accepts shape as its arguments or a numpy array.
        :param shape: Must be provided if the initializer is a function.
        :param kwargs:
        """
        if callable(initializer):
            self.x = initializer(shape)
        else:
            self.x = np.array(initializer, dtype=np.float64)
        self.shape = self.x.shape
        super(OpVariable, self).__init__(**kwargs)

    def update(self, value: Union[int, float, list, np.ndarray]) -> None:
        value = np.array(value)
        if self.x.shape != value.shape:
            raise ValueError('The shape of two tensors should be equal, '
                             'got %s and %s' % (str(self.x.shape), str(value.shape)))
        self.x = value

    def _get_name(self) -> str:
        return 'W%s' % str(self.x.shape)

    def _get_op_name(self) -> str:
        return 'w_%d' % self._op_index

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Returns the contained weights."""
        return self.x

    def _backward(self, gradient: Operation) -> None:
        """No backward operation needed."""
        pass
