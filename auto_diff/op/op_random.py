from typing import Union, Mapping, Sequence
import numpy as np
from .operation import Operation
from .op_shape import OpShape


class OpRandom(Operation):
    """Constant tensor filled with random values."""

    def __init__(self, shape: Union[int, Sequence[int], OpShape], **kwargs):
        if isinstance(shape, OpShape):
            self.shape = shape.inputs[0].shape
        elif not isinstance(shape, int):
            self.shape = tuple(shape)
        else:
            self.shape = shape
        self.params = {
            'shape': shape,
        }
        super(OpRandom, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, Operation], np.ndarray]) -> np.ndarray:
        """Generate and returns the constant."""
        if isinstance(self.params['shape'], OpShape):
            shape = self.params['shape'].forward(feed_dict)
        else:
            shape = self.shape
        return np.random.random(shape)

    def _backward(self, gradient: np.ndarray) -> None:
        """No backward operation needed."""
        pass
