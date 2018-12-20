from typing import Union, Optional, Sequence
from .operation import Operation


class OpKeepdims(Operation):
    """Common operations for keepdims."""

    def __init__(self,
                 base_op: type,
                 x: Operation,
                 axis: Optional[Union[int, Sequence[int]]] = None,
                 keepdims: bool = False,
                 **kwargs):
        self.inputs = [x]
        self.params = {
            'axis': axis,
            'keepdims': keepdims,
        }
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
            self.params['keepdims'] = False
        elif not keepdims:
            self.inputs[0] = base_op(self.inputs[0], axis=self.params['axis'], keepdims=True).\
                squeeze(axis=axis, name=self.inputs[0].name)
        super(OpKeepdims, self).__init__(**kwargs)
