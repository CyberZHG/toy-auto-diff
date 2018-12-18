from typing import Mapping, Union, Optional, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder
from .op_keepdims import OpKeepdims


class OpMean(OpKeepdims):
    """Calculate the mean of elements."""

    def __init__(self,
                 x: Operation,
                 axis: Optional[Union[int, Sequence[int]]] = None,
                 keepdims: bool = False,
                 **kwargs):
        super(OpMean, self).__init__(self.__class__, x, axis, keepdims, **kwargs)

    def _get_name(self) -> str:
        return 'mean' + self._get_args_str(self.inputs[0].name)

    def _get_op_name(self) -> str:
        return 'mean' + self._get_args_str(self.inputs[0]._op_name)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        if not self.keepdims:
            return self.inputs[0].forward(feed_dict)
        return np.mean(self.inputs[0].forward(feed_dict), axis=self.axis, keepdims=True)

    def _backward(self, gradient: Operation) -> None:
        import auto_diff as ad
        if not self.keepdims:
            self.inputs[0].backward(gradient)
            return
        self.gradient = gradient / (ad.prod(ad.shape(self.inputs[0])) / ad.prod(ad.shape(self)))
        self.inputs[0].backward(self.gradient)
