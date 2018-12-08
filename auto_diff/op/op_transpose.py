from typing import Mapping, Union, Optional, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpTranspose(Operation):
    """Transpose the tensor."""

    def __init__(self, x: Operation, axes: Optional[Sequence[int]] = None, **kwargs):
        self.inputs = [x]
        self.axes = axes
        if axes is None:
            self.reverse_axes = None
            self.shape = tuple(reversed(x.shape))
        else:
            self.reverse_axes = [0] * len(axes)
            for i, axis in enumerate(axes):
                self.reverse_axes[axis] = i
            self.shape = tuple(x.shape[axis] for axis in axes)
        super(OpTranspose, self).__init__(**kwargs)

    def _get_name(self) -> str:
        if self.axes is None:
            return 'transpose(%s)' % self.inputs[0].name
        return 'transpose(%s, axes=%s)' % (self.inputs[0].name, str(self.axes))

    def _get_op_name(self) -> str:
        if self.axes is None:
            return 'transpose(%s)' % self.inputs[0]._op_name
        return 'transpose(%s, axes=%s)' % (self.inputs[0]._op_name, str(self.axes))

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Transpose the tensor."""
        return np.transpose(self.inputs[0].forward(feed_dict), axes=self.axes)

    def _backward(self, gradient: Operation) -> None:
        """Transpose the gradients to its old shape."""
        self.gradient = gradient.transpose(axes=self.reverse_axes)
        self.inputs[0].backward(self.gradient)
