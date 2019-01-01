from typing import Mapping, Union, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpPad(Operation):
    """Pad with zeros."""

    def __init__(self, x: Operation, pad_width: Union[int, Sequence[int], Sequence[Sequence[int]]], **kwargs):
        self.inputs = [x]
        shape, slices = [], []
        self.params = {
            'pad_width': pad_width,
        }
        if isinstance(pad_width, int):
            pad_width = [(pad_width,)] * x.dim
        elif isinstance(pad_width, tuple):
            if all(map(lambda x: isinstance(x, int), pad_width)):
                pad_width = [pad_width] * x.dim
                # TODO: check length of pad
            else:
                pad_width = list(pad_width)
        for i in range(x.dim):
            if len(pad_width[i]) == 1:
                pad_width[i] = (pad_width[i][0], pad_width[i][0])
            if x.shape[i] is None:
                shape.append(None)
            else:
                shape.append(x.shape[i] + pad_width[i][0] + pad_width[i][1])
            start, stop = None, None
            if pad_width[i][0]:
                start = pad_width[i][0]
            if pad_width[i][1]:
                stop = -pad_width[i][1]
            slices.append(slice(start, stop))
        self.shape = tuple(shape)
        self.slices = tuple(slices)
        super(OpPad, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.pad(self.values[0], self.params['pad_width'], mode='constant', constant_values=0)

    def _backward(self, gradient: np.ndarray) -> None:
        self.gradients = [gradient[self.slices]]
