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

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        if not self.params['keepdims']:
            return self.inputs[0].forward(feed_dict)
        return np.mean(self.inputs[0].forward(feed_dict), axis=self.params['axis'], keepdims=True)

    def _backward(self, gradient: Operation) -> None:
        import auto_diff as ad
        if not self.params['keepdims']:
            self.gradients = [gradient]
            return
        self.gradients = [gradient / (ad.prod(ad.shape(self.inputs[0])) / ad.prod(ad.shape(self)))]
