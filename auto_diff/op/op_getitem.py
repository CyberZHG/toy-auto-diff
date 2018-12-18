from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpGetitem(Operation):
    """Get item based on indexing"""

    def __init__(self, x: Operation, item, **kwargs):
        self.inputs = [x]
        self.item = item
        shape = []
        if isinstance(item, int):
            item = (item,)
        for i, s in enumerate(item):
            if isinstance(s, slice):
                start, stop, step = s.start, s.stop, s.step
                if step is None:
                    step = 1
                if step > 0:
                    if start is None:
                        start = 0
                    if stop is None:
                        stop = x.shape[i]
                else:
                    if start is None:
                        start = x.shape[i]
                    if stop is None:
                        stop = 0
                    start, stop, step = stop, start, -step
                if start is None or stop is None:
                    shape.append(None)
                else:
                    if start < 0:
                        start = (start + x.shape[i]) % x.shape[i]
                    if stop < 0:
                        stop = (stop + x.shape[i]) % x.shape[i]
                    if start >= stop:
                        shape.append(0)
                    else:
                        shape.append((stop - start - 1) // step + 1)
        self.shape = tuple(shape) + x.shape[len(item):]
        super(OpGetitem, self).__init__(**kwargs)

    def _get_name(self) -> str:
        slice_str = []
        if isinstance(self.item, int):
            slice_str = [str(self.item)]
        else:
            for s in self.item:
                if isinstance(s, slice):
                    part = ''
                    if s.start is not None:
                        part += str(s.start)
                    part += ':'
                    if s.stop is not None:
                        part += str(s.stop)
                    if s.step is not None:
                        part += ':' + str(s.step)
                    slice_str.append(part)
                else:
                    slice_str.append(str(s))
        return '%s[%s]' % (self.inputs[0].name, ', '.join(slice_str))

    def _get_op_name(self) -> str:
        return '%s[%s]' % (self.inputs[0].name, str(self.item))

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return self.inputs[0].forward(feed_dict)[self.item]

    def _backward(self, gradient: Operation) -> None:
        from .op_zeros_like import OpZerosLike
        from .op_setitem import OpSetitem
        self.gradient = OpSetitem(OpZerosLike(self.inputs[0]), self.item, gradient)
        self.inputs[0].backward(self.gradient)
