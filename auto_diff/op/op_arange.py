from typing import Mapping, Union, Optional
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder
from .op_constant import OpConstant


class OpArange(Operation):
    """Get evenly spaced values within a given interval."""

    def __init__(self,
                 start: [Union[int, float, Operation]],
                 stop: Optional[Union[int, float, Operation]] = None,
                 step: Optional[Union[int, float, Operation]] = None,
                 **kwargs):
        if stop is None:
            start, stop = None, start
        if isinstance(start, (int, float)):
            start = OpConstant(start)
        if isinstance(stop, (int, float)):
            stop = OpConstant(stop)
        if isinstance(step, (int, float)):
            step = OpConstant(step)
        self.params = {
            'start': start,
            'stop': stop,
            'step': step,
        }
        self.shape = (None,)
        super(OpArange, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        values = [self.params['start'], self.params['stop'], self.params['step']]
        for i, value in enumerate(values):
            if value is not None:
                values[i] = values[i].forward(feed_dict)
        if values[0] is None:
            values[0], values[1] = values[1], None
        return np.arange(*values, dtype=np.float64)

    def _backward(self, gradient: np.ndarray) -> None:
        pass
