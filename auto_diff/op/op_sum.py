from typing import Mapping, Union, Optional, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpSum(Operation):
    """Sum of elements over a given axis."""

    def __init__(self,
                 x: Operation,
                 axis: Optional[Union[int, Sequence[int]]] = None,
                 keepdims: bool = False,
                 **kwargs):
        self.inputs = [x]
        self.axis = axis
        self.keepdims = keepdims
        if axis is None:
            if self.keepdims:
                self.shape = tuple([1] * len(x.shape))
            else:
                self.shape = (1,)
        elif isinstance(axis, int):
            self.shape = list(x.shape)
            if self.keepdims:
                self.shape[axis] = 1
            else:
                del self.shape[axis]
            self.shape = tuple(self.shape)
        else:
            axis = tuple(list(sorted(set([(a + len(x.shape)) % len(x.shape) for a in axis]))))
            self.shape = list(x.shape)
            for a in reversed(axis):
                if self.keepdims:
                    self.shape[a] = 1
                else:
                    del self.shape[a]
            if len(self.shape) == 0:
                self.shape = (1,)
            else:
                self.shape = tuple(self.shape)

        if not self.keepdims:
            if self.shape == (1,):
                axis = tuple(list(range(1, len(x.shape))))
            self.inputs[0] = self.inputs[0].\
                sum(axis=self.axis, keepdims=True).\
                squeeze(axis=axis, name=self.inputs[0].name)
        super(OpSum, self).__init__(**kwargs)

    def _get_args_str(self, name):
        args = [name]
        if self.axis is not None:
            args.append('axis=%s' % str(self.axis))
        if self.keepdims:
            args.append('keepdims=%s' % str(self.keepdims))
        return '(' + ', '.join(args) + ')'

    def _get_name(self) -> str:
        return 'sum' + self._get_args_str(self.inputs[0].name)

    def _get_op_name(self) -> str:
        return 'sum' + self._get_args_str(self.inputs[0]._op_name)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Sum over axis"""
        if not self.keepdims:
            return self.inputs[0].forward(feed_dict)
        return np.sum(self.inputs[0].forward(feed_dict), axis=self.axis, keepdims=True)

    def _backward(self, gradient: Operation) -> None:
        """Expand the dimensions of the gradient."""
        from .op_ones_like import OpOnesLike
        if not self.keepdims:
            self.inputs[0].backward(gradient)
            return
        self.gradient = OpOnesLike(self.inputs[0]) * gradient
        self.inputs[0].backward(self.gradient)
