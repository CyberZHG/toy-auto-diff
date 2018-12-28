from typing import Mapping, Union, Callable, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpMapFn(Operation):
    """Mapping with function."""

    def __init__(self, fn: Callable, elems: Union[Operation, Sequence[Operation]], **kwargs):
        self.inputs = []
        self.params = {
            'fn': fn,
            'elems': elems,
        }
        self.is_seq_input = isinstance(elems, (list, tuple))
        if self.is_seq_input:
            first_input = tuple(map(lambda x: x[0], elems))
            elem_dim = elems[0].shape[0]
        else:
            first_input = elems[0]
            elem_dim = elems.shape[0]
        first_output = fn(first_input)
        self.is_seq_output = isinstance(first_output, (list, tuple))
        if self.is_seq_output:
            self.fn_output_num = len(first_output)
            self.shape = (self.fn_output_num, elems.shape[0]) + first_output[0].shape
        else:
            self.fn_output_num = 0
            self.shape = (elem_dim,) + first_output.shape
        super(OpMapFn, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        fn, elems = self.params['fn'], self.params['elems']
        if self.is_seq_input:
            length = elems[0].forward(feed_dict).shape[0]
            self.inputs = [fn(tuple(map(lambda x: x[i], elems))) for i in range(length)]
        else:
            length = elems.forward(feed_dict).shape[0]
            self.inputs = [fn(elems[i]) for i in range(length)]
        if self.is_seq_output:
            self.inputs = [result[i] for i in range(self.fn_output_num) for result in self.inputs]
        results = list(map(lambda x: x.forward(feed_dict), self.inputs))
        result = np.stack(results)
        if self.is_seq_output:
            result = result.reshape((self.fn_output_num, -1) + result.shape[1:])
        return result

    def _backward(self, gradient: Operation) -> None:
        pass
