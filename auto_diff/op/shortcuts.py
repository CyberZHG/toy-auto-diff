from typing import Union, Optional, Sequence, Callable
import numpy as np
from .operation import Operation

__all__ = [
    'array', 'constant', 'placeholder', 'variable',
    'ones', 'zeros', 'ones_like', 'zeros_like',
    'transpose', 'reshape', 'flatten', 'expand_dims', 'squeeze',
    'sum',
]


def constant(x: Union[int, float, list, np.ndarray], **kwargs) -> Operation:
    """See :class:`OpConstant`."""
    from .op_constant import OpConstant
    return OpConstant(x, **kwargs)


array = constant


def ones(shape: Union[int, Sequence[int]], **kwargs) -> Operation:
    """See :class:`OpOnes`."""
    from .op_ones import OpOnes
    return OpOnes(shape, **kwargs)


def zeros(shape: Union[int, Sequence[int]], **kwargs) -> Operation:
    """See :class:`OpZeros`."""
    from .op_zeros import OpZeros
    return OpZeros(shape, **kwargs)


def ones_like(x: Operation, **kwargs) -> Operation:
    """See :class:`OpOnesLike`."""
    from .op_ones_like import OpOnesLike
    return OpOnesLike(x, **kwargs)


def zeros_like(x: Operation, **kwargs) -> Operation:
    """See :class:`OpZerosLike`."""
    from .op_zeros_like import OpZerosLike
    return OpZerosLike(x, **kwargs)


def placeholder(shape: Sequence[int], **kwargs) -> Operation:
    """See :class:`OpPlaceholder`."""
    from .op_placeholder import OpPlaceholder
    return OpPlaceholder(shape, **kwargs)


def variable(initializer: Union[Callable, int, float, list, np.ndarray], shape: tuple = None, **kwargs) -> 'OpVariable':
    """See :class:`OpVariable`."""
    from .op_variable import OpVariable
    return OpVariable(initializer, shape, **kwargs)


def transpose(x: Operation, axes: Optional[Sequence[int]] = None, **kwargs) -> Operation:
    """See :class:`OpTranspose`."""
    from .op_transpose import OpTranspose
    return OpTranspose(x, axes, **kwargs)


def reshape(x: Operation, shape: Sequence[int], **kwargs) -> Operation:
    """See :class:`OpReshape`."""
    from .op_reshape import OpReshape
    return OpReshape(x, shape, **kwargs)


def flatten(x: Operation, **kwargs) -> Operation:
    """See :class:`OpFlatten`."""
    from .op_flatten import OpFlatten
    return OpFlatten(x, **kwargs)


def expand_dims(x: Operation, axis: Optional[int] = None, **kwargs) -> Operation:
    """See :class:`OpExpandDims`."""
    from .op_expand_dims import OpExpandDims
    return OpExpandDims(x, axis, **kwargs)


def squeeze(x: Operation, axis: Optional[Union[int, Sequence[int]]] = None, **kwargs) -> Operation:
    """See :class:`OpSqueeze`."""
    from .op_squeeze import OpSqueeze
    return OpSqueeze(x, axis, **kwargs)


def sum(x: Operation, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, **kwargs) -> Operation:
    """See :class:`OpSum`."""
    from .op_sum import OpSum
    return OpSum(x, axis, keepdims, **kwargs)
