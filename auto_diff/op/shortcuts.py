from typing import Union, Optional, Sequence, Callable
import numpy as np
from .operation import Operation

__all__ = [
    'array', 'constant', 'placeholder', 'variable', 'setitem',
    'ones', 'zeros', 'ones_like', 'zeros_like', 'random', 'arange',
    'transpose', 'reshape', 'flatten', 'expand_dims', 'squeeze', 'shape', 'pad',
    'sum', 'prod', 'mean', 'max', 'min', 'argmax',
    'square', 'exp', 'log', 'tanh',
    'add', 'subtract', 'multiply', 'divide', 'dot', 'negative', 'equal', 'less', 'where',
    'maximum', 'minimum',
    'map_fn', 'while_loop',
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


def random(shape: Union[int, Sequence[int]], **kwargs) -> Operation:
    """See :class:`OpRandom`."""
    from .op_random import OpRandom
    return OpRandom(shape, **kwargs)


def arange(start: [Union[int, float, Operation]],
           stop: Optional[Union[int, float, Operation]] = None,
           step: Optional[Union[int, float, Operation]] = None,
           **kwargs) -> Operation:
    """See :class:`OpArange`."""
    from .op_arange import OpArange
    return OpArange(start, stop, step, **kwargs)


def placeholder(shape: Sequence[int], **kwargs) -> Operation:
    """See :class:`OpPlaceholder`."""
    from .op_placeholder import OpPlaceholder
    return OpPlaceholder(shape, **kwargs)


def variable(initializer: Union[Callable, int, float, list, np.ndarray], shape: tuple = None, **kwargs) -> 'OpVariable':
    """See :class:`OpVariable`."""
    from .op_variable import OpVariable
    return OpVariable(initializer, shape, **kwargs)


def setitem(x: Operation, key, value: Operation, **kwargs) -> Operation:
    """See :class:`OpSetitem`."""
    from .op_setitem import OpSetitem
    return OpSetitem(x, key, value, **kwargs)


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


def shape(x: Operation, **kwargs) -> Operation:
    """See :class:`OpShape`."""
    from .op_shape import OpShape
    return OpShape(x, **kwargs)


def pad(x: Operation, pad_width: Union[int, Sequence[int], Sequence[Sequence[int]]], **kwargs) -> Operation:
    """See :class:`OpPad`."""
    from .op_pad import OpPad
    return OpPad(x, pad_width, **kwargs)


def sum(x: Operation, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, **kwargs) -> Operation:
    """See :class:`OpSum`."""
    from .op_sum import OpSum
    return OpSum(x, axis, keepdims, **kwargs)


def prod(x: Operation, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, **kwargs) -> Operation:
    """See :class:`OpProd`."""
    from .op_prod import OpProd
    return OpProd(x, axis, keepdims, **kwargs)


def mean(x: Operation, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, **kwargs) -> Operation:
    """See :class:`OpMean`."""
    from .op_mean import OpMean
    return OpMean(x, axis, keepdims, **kwargs)


def max(x: Operation, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, **kwargs) -> Operation:
    """See :class:`OpMax`."""
    from .op_max import OpMax
    return OpMax(x, axis, keepdims, **kwargs)


def min(x: Operation, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, **kwargs) -> Operation:
    """See :class:`OpMin`."""
    from .op_min import OpMin
    return OpMin(x, axis, keepdims, **kwargs)


def argmax(x: Operation, axis: Optional[int] = None, **kwargs) -> Operation:
    """See :class:`OpArgmax`."""
    from .op_argmax import OpArgmax
    return OpArgmax(x, axis, **kwargs)


def square(x: Operation, **kwargs) -> Operation:
    """See :class:`OpSquare`."""
    from .op_square import OpSquare
    return OpSquare(x, **kwargs)


def exp(x: Operation, **kwargs) -> Operation:
    """See :class:`OpExp`."""
    from .op_exp import OpExp
    return OpExp(x, **kwargs)


def log(x: Operation, **kwargs) -> Operation:
    """See :class:`OpLog`."""
    from .op_log import OpLog
    return OpLog(x, **kwargs)


def tanh(x: Operation, **kwargs) -> Operation:
    """See :class:`OpTanh`."""
    from .op_tanh import OpTanh
    return OpTanh(x, **kwargs)


def add(x: Operation, y: Operation, **kwargs) -> Operation:
    """See :class:`OpAdd`."""
    from .op_add import OpAdd
    return OpAdd(x, y, **kwargs)


def subtract(x: Operation, y: Operation, **kwargs) -> Operation:
    """See :class:`OpSubtract`."""
    from .op_subtract import OpSubtract
    return OpSubtract(x, y, **kwargs)


def multiply(x: Operation, y: Operation, **kwargs) -> Operation:
    """See :class:`OpMultiply`."""
    from .op_multiply import OpMultiply
    return OpMultiply(x, y, **kwargs)


def divide(x: Operation, y: Operation, **kwargs) -> Operation:
    """See :class:`OpDivide`."""
    from .op_divide import OpDivide
    return OpDivide(x, y, **kwargs)


def dot(x: Operation, y: Operation, **kwargs) -> Operation:
    """See :class:`OpDot`."""
    from .op_dot import OpDot
    return OpDot(x, y, **kwargs)


def negative(x: Operation, **kwargs) -> Operation:
    """See :class:`OpNegative`."""
    from .op_negative import OpNegative
    return OpNegative(x, **kwargs)


def equal(x: Operation, y: Operation, **kwargs) -> Operation:
    """See :class:`OpEqual`."""
    from .op_equal import OpEqual
    return OpEqual(x, y, **kwargs)


def less(x: Operation, y: Operation, **kwargs) -> Operation:
    """See :class:`OpLess`."""
    from .op_less import OpLess
    return OpLess(x, y, **kwargs)


def where(condition: Operation, x: Optional[Operation] = None, y: Optional[Operation] = None, **kwargs) -> Operation:
    """See :class:`OpWhere`."""
    from .op_where import OpWhere
    return OpWhere(condition, x, y, **kwargs)


def maximum(x: Operation, y: Operation, **kwargs) -> Operation:
    """See :class:`OpMaximum`"""
    from .op_maximum import OpMaximum
    return OpMaximum(x, y, **kwargs)


def minimum(x: Operation, y: Operation, **kwargs) -> Operation:
    """See :class:`OpMinimum`"""
    from .op_minimum import OpMinimum
    return OpMinimum(x, y, **kwargs)


def map_fn(fn: Callable, elems: Union[Operation, Sequence[Operation]], **kwargs) -> Operation:
    """See :class:`OpMapFn`"""
    from .op_map_fn import OpMapFn
    return OpMapFn(fn, elems, **kwargs)


def while_loop(cond: Callable,
               body: callable,
               loop_vars: Sequence[Operation],
               output_index: int = 0,
               **kwargs):
    """See :class:`OpWhileLoop`"""
    from .op_while_loop import OpWhileLoop
    return OpWhileLoop(cond, body, loop_vars, output_index, **kwargs)
