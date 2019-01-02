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
        if isinstance(item, (int, Operation)):
            item = (item,)
        for i, s in enumerate(item):
            if isinstance(s, slice):
                if x.shape[i] is None or any(map(lambda x: isinstance(x, Operation), [s.start, s.stop, s.step])):
                    shape.append(None)
                else:
                    shape.append(len(range(*s.indices(x.shape[i]))))
        self.shape = tuple(shape) + x.shape[len(item):]
        self.item_forward = None
        super(OpGetitem, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        self.item_forward = []
        item = self.params['item']
        if isinstance(item, (int, Operation)):
            item = (item,)
        for s in item:
            if isinstance(s, int):
                self.item_forward.append(s)
            elif isinstance(s, Operation):
                self.item_forward.append(int(s.forward(feed_dict)))
            elif isinstance(s, slice):
                values = [s.start, s.stop, s.step]
                for i, v in enumerate(values):
                    if isinstance(v, Operation):
                        values[i] = int(v.forward(feed_dict))
                self.item_forward.append(slice(*values))
        self.item_forward = tuple(self.item_forward)
        return self.values[0][self.item_forward]

    def _backward(self, gradient: np.ndarray) -> None:
        holder = np.zeros_like(self.values[0])
        holder[self.item_forward] = gradient
        self.gradients = [holder]
