from typing import Union, Optional, Sequence, Callable
import numpy as np
from .operation import Operation

__all__ = [
    'array', 'constant', 'placeholder', 'variable',
    'transpose', 'reshape', 'flatten', 'expand_dims', 'squeeze',
]


def constant(x: Union[int, float, list, np.ndarray]):
    """See :class:`OpConstant`."""
    from .op_constant import OpConstant
    return OpConstant(x)


array = constant


def placeholder(shape: tuple):
    """See :class:`OpPlaceholder`."""
    from .op_placeholder import OpPlaceholder
    return OpPlaceholder(shape)


def variable(initializer: Union[Callable, int, float, list, np.ndarray], shape: tuple = None):
    """See :class:`OpVariable`."""
    from .op_variable import OpVariable
    return OpVariable(initializer, shape)


def transpose(x: Operation, axes: Optional[Sequence[int]] = None):
    """See :class:`OpTranspose`."""
    from .op_transpose import OpTranspose
    return OpTranspose(x, axes)


def reshape(x: Operation, shape: Sequence[int]):
    """See :class:`OpReshape`."""
    from .op_reshape import OpReshape
    return OpReshape(x, shape)


def flatten(x: Operation):
    """See :class:`OpFlatten`."""
    from .op_flatten import OpFlatten
    return OpFlatten(x)


def expand_dims(x: Operation, axis: Optional[int] = None):
    """See :class:`OpExpandDims`."""
    from .op_expand_dims import OpExpandDims
    return OpExpandDims(x, axis)


def squeeze(x: Operation, axis: Optional[Union[int, Sequence[int]]] = None):
    """See :class:`OpSqueeze`."""
    from .op_squeeze import OpSqueeze
    return OpSqueeze(x, axis)
