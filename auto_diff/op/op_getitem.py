from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpGetitem(Operation):
    """Get item based on indexing"""

    def __init__(self, x: Operation, item, **kwargs):
        self.inputs = [x]
        self.params = {
            'item': item,
        }
        shape = []
        if isinstance(item, int):
            item = (item,)
        for i, s in enumerate(item):
            if isinstance(s, slice):
                if x.shape[i] is None:
                    shape.append(None)
                else:
                    shape.append(len(range(*s.indices(x.shape[i]))))
        self.shape = tuple(shape) + x.shape[len(item):]
        super(OpGetitem, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return self.inputs[0].forward(feed_dict)[self.params['item']]

    def _backward(self, gradient: Operation) -> None:
        from .op_zeros_like import OpZerosLike
        from .op_setitem import OpSetitem
        self.gradients = [OpSetitem(OpZerosLike(self.inputs[0]), self.params['item'], gradient)]
