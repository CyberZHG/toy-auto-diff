from typing import Mapping, Union, Callable, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpWhileLoop(Operation):
    """While loop."""

    def __init__(self,
                 cond: Callable,
                 body: callable,
                 loop_vars: Sequence[Operation],
                 output_index: int = -1,
                 **kwargs):
        self.inputs = []
        self.params = {
            'cond': cond,
            'body': body,
            'loop_vars': loop_vars,
            'output_index': output_index,
        }
        self.shape = (None,) + tuple(loop_vars[output_index].shape)
        super(OpWhileLoop, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        outputs = []
        cond, body, loop_vars = self.params['cond'], self.params['body'], self.params['loop_vars']
        output_index = self.params['output_index']
        self.inputs = []
        while cond(loop_vars).forward(feed_dict):
            loop_vars = body(loop_vars)
            self.inputs.append(loop_vars[output_index])
            outputs.append(self.inputs[-1].forward(feed_dict))
        return np.stack(outputs)

    def _backward(self, gradient: np.ndarray) -> None:
        gradient = self._broadcast_backward(
            gradient,
            (len(self.inputs),) + tuple(self.params['loop_vars'][self.params['output_index']].shape),
        )
        self.gradients = [gradient[i] for i in range(len(self.inputs))]
