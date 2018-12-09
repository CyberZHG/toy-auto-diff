import numpy as np
from .op_reshape import Operation, OpReshape


class OpFlatten(OpReshape):
    """Flatten the tensor to 1-D array."""

    def __init__(self, x: Operation, **kwargs):
        super(OpFlatten, self).__init__(x, shape=(int(np.prod(x.shape)),), **kwargs)

    def _get_name(self) -> str:
        return 'flatten(%s)' % self.inputs[0].name

    def _get_op_name(self) -> str:
        return 'flatten(%s)' % self.inputs[0]._op_name
