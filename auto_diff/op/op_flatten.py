import numpy as np
from .operation import Operation
from .op_reshape import OpReshape


class OpFlatten(OpReshape):
    """Flatten the tensor to 1-D array."""

    def __init__(self, x: Operation, **kwargs):
        super(OpFlatten, self).__init__(x, shape=(int(np.prod(np.shape(x))),), **kwargs)
