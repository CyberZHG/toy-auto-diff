from typing import Mapping, Union, Optional, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpKeepdims(Operation):
    """Common operations for keepdims."""

    def __init__(self,
                 base_op: type,
                 x: Operation,
                 axis: Optional[Union[int, Sequence[int]]] = None,
                 keepdims: bool = False,
                 **kwargs):
        self.inputs = [x]
        self.axis = axis
        self.keepdims = keepdims
        if axis is None:
            if keepdims:
                self.shape = tuple([1] * x.dim)
            else:
                self.shape = ()
            axis = tuple(range(x.dim))
        else:
            if isinstance(axis, int):
                axis = [axis]
            axis = tuple(list(sorted(set([(a + len(x.shape)) % len(x.shape) for a in axis]))))
            shape = list(x.shape)
            for a in reversed(axis):
                if keepdims:
                    shape[a] = 1
                else:
                    del shape[a]
            self.shape = tuple(shape)

        if x.isscalar():
            self.keepdims = False
        elif not self.keepdims:
            self.inputs[0] = base_op(self.inputs[0], axis=self.axis, keepdims=True).\
                squeeze(axis=axis, name=self.inputs[0].name)
        super(OpKeepdims, self).__init__(**kwargs)

    def _get_args_str(self, name):
        args = [name]
        if self.axis is not None:
            args.append('axis=%s' % str(self.axis))
        if self.keepdims:
            args.append('keepdims=%s' % str(self.keepdims))
        return '(' + ', '.join(args) + ')'

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
