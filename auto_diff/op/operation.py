from typing import Union, Mapping, Optional, Sequence
import numpy as np


class Operation(object):
    """Abstract operation for building computing graph."""

    #: The counter for giving each operation a unique index.
    __op_counter = [0]

    #: The key for extracting step information from session.
    KEY_STEP = '__step__'
    #: The key that indicates whether it is training.
    KEY_TRAINING = '__training__'

    def __init__(self, **kwargs):
        if not hasattr(self, 'name'):
            self._name = None
            if 'name' in kwargs:
                self._name = kwargs['name']
        if not hasattr(self, 'shape'):
            self.shape: Sequence[Optional[int]] = None
            raise NotImplementedError('Shape not defined')
        if not hasattr(self, 'inputs'):
            self.inputs: Sequence['Operation'] = []
        if not hasattr(self, 'params'):
            self.params: dict = {}
        self.values: Sequence[np.ndarray] = []
        self.output: Optional[np.ndarray] = None
        self.gradients: Optional[Sequence[np.ndarray]] = None
        self._op_index = self.__op_counter[0]
        self.__op_counter[0] += 1
        self._last_step = -1
        self._last_forward = None
        self._enable_cache = True

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        class_name = self.__class__.__name__[2:]
        func_name = ''
        for c in class_name:
            if c.isupper():
                if func_name:
                    func_name += '_'
                c = c.lower()
            func_name += c
        args = [str(inp) for inp in self.inputs] +\
               ['%s=%s' % (str(key), str(value)) for key, value in self.params.items() if value is not None]
        return func_name + '(%s)' % ', '.join(args)

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def dim(self) -> int:
        return len(self.shape)

    def isscalar(self) -> bool:
        return self.shape == ()

    def forward(self, feed_dict: Mapping[Union[str, 'Operation'], np.ndarray] = None) -> np.ndarray:
        """Do the calculations to get the output of the operations.

        :param feed_dict: Contains the real values of placeholders, see :class:`OpPlaceholder`.
        :return: A numpy array.
        """
        if feed_dict is None:
            feed_dict = {}
        if self._enable_cache and self.KEY_STEP in feed_dict and feed_dict[self.KEY_STEP] == self._last_step:
            return self._last_forward
        self.values = [inp.forward(feed_dict) for inp in self.inputs]
        self.output = self._forward(feed_dict)
        if self._enable_cache and self.KEY_STEP in feed_dict:
            self._last_step = feed_dict[self.KEY_STEP]
            self._last_forward = self.output
        return self.output

    def _forward(self, feed_dict: Mapping[Union[str, 'Operation'], np.ndarray]) -> np.ndarray:
        """Forward operation to be implemented."""
        raise NotImplementedError('Forward operation not implemented')

    def backward(self) -> None:
        """Update gradients recursively."""
        inputs, degrees, operations = {self: []}, {self: 0}, [self]
        front = 0
        while front < len(operations):
            current = operations[front]
            for grad_index, op in enumerate(current.inputs):
                if op in degrees:
                    inputs[op].append((current, grad_index))
                    degrees[op] += 1
                else:
                    inputs[op] = [(current, grad_index)]
                    degrees[op] = 1
                    operations.append(op)
            front += 1
        inv_degrees = {}
        for op, degree in degrees.items():
            if degree not in inv_degrees:
                while len(inv_degrees) <= degree:
                    inv_degrees[len(inv_degrees)] = set()
            inv_degrees[degree].add(op)
        while len(inv_degrees[0]):
            for op in inv_degrees[0]:
                inv_degrees[0].remove(op)
                if op == self:
                    self._backward(np.ones_like(self.output))
                else:
                    gradient = 0.0
                    for input_op, grad_index in inputs[op]:
                        gradient += input_op.gradients[grad_index]
                    op._backward(gradient)
                for input_op in op.inputs:
                    inv_degrees[degrees[input_op]].remove(input_op)
                    degrees[input_op] -= 1
                    inv_degrees[degrees[input_op]].add(input_op)
                break

    def _backward(self, gradient: np.ndarray) -> None:
        """Backward operation to be implemented."""
        raise NotImplementedError('Backward operation not implemented')

    def _broadcast_shape(self, *args: Union[int, float, 'Operation']):
        self.shape = ()
        for x in args:
            if self.isscalar():
                self.shape = x.shape
                continue
            if x.isscalar():
                continue
            min_dim = min(len(self.shape), len(x.shape))
            shape = []
            for i in range(1, min_dim + 1):
                if self.shape[-i] is None or x.shape[-i] is None:
                    shape.append(None)
                    continue
                if self.shape[-i] != 1 and x.shape[-i] != 1 and self.shape[-i] != x.shape[-i]:
                    raise ValueError('Cannot broadcast with shape %s and %s' % (str(self.shape), str(x.shape)))
                shape.append(max(self.shape[-i], x.shape[-i]))
            self.shape = tuple(list(self.shape[:-min_dim]) + list(x.shape[:-min_dim]) + list(reversed(shape)))

    @staticmethod
    def _broadcast_backward(gradient: np.ndarray, target_shape):
        gradient_shape = np.shape(gradient)
        if target_shape == gradient_shape:
            return gradient
        if target_shape == ():
            gradient = gradient.sum()
            return gradient
        expand_dim = len(gradient_shape) - len(target_shape)
        axis = list(range(expand_dim))
        for i, dim in enumerate(target_shape):
            if target_shape[i] == 1 and (gradient_shape[i + expand_dim] is None or gradient_shape[i + expand_dim] > 1):
                axis.append(expand_dim + i)
        if len(axis) == 1:
            axis = axis[0]
        else:
            axis = tuple(axis)
        gradient = gradient.sum(axis=axis, keepdims=True)
        if expand_dim:
            gradient = gradient.squeeze(axis=tuple(list(range(expand_dim))))
        return gradient

    def transpose(self, axes: Optional[Sequence[int]] = None, **kwargs) -> 'Operation':
        """See :class:`OpTranspose`."""
        from .op_transpose import OpTranspose
        return OpTranspose(self, axes, **kwargs)

    def reshape(self, shape: Sequence[int], **kwargs) -> 'Operation':
        """See :class:`OpReshape`."""
        from .op_reshape import OpReshape
        return OpReshape(self, shape, **kwargs)

    def flatten(self, **kwargs) -> 'Operation':
        """See :class:`OpFlatten`."""
        from .op_flatten import OpFlatten
        return OpFlatten(self, **kwargs)

    def expand_dims(self, axis: Optional[int] = None, **kwargs) -> 'Operation':
        """See :class:`OpExpandDims`."""
        from .op_expand_dims import OpExpandDims
        return OpExpandDims(self, axis, **kwargs)

    def squeeze(self, axis=None, **kwargs) -> 'Operation':
        """See :class:`OpSqueeze`."""
        from .op_squeeze import OpSqueeze
        return OpSqueeze(self, axis, **kwargs)

    def sum(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, **kwargs) -> 'Operation':
        """See :class:`OpSum`."""
        from .op_sum import OpSum
        return OpSum(self, axis, keepdims, **kwargs)

    def prod(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, **kwargs) -> 'Operation':
        """See :class:`OpProd`."""
        from .op_prod import OpProd
        return OpProd(self, axis, keepdims, **kwargs)

    def mean(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, **kwargs) -> 'Operation':
        """See :class:`OpMean`."""
        from .op_mean import OpMean
        return OpMean(self, axis, keepdims, **kwargs)

    def max(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, **kwargs) -> 'Operation':
        """See :class:`OpMax`."""
        from .op_max import OpMax
        return OpMax(self, axis, keepdims, **kwargs)

    def min(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, **kwargs) -> 'Operation':
        """See :class:`OpMin`."""
        from .op_min import OpMin
        return OpMin(self, axis, keepdims, **kwargs)

    def dot(self, x: 'Operation', **kwargs) -> 'Operation':
        """See :class:`OpDot`."""
        from .op_dot import OpDot
        return OpDot(self, x, **kwargs)

    def __add__(self, other) -> 'Operation':
        """See :class:`OpAdd`."""
        from .op_add import OpAdd
        from .op_constant import OpConstant
        if not isinstance(other, Operation):
            other = OpConstant(other)
        return OpAdd(self, other)

    def __radd__(self, other) -> 'Operation':
        """See :class:`OpAdd`."""
        from .op_add import OpAdd
        from .op_constant import OpConstant
        if not isinstance(other, Operation):
            other = OpConstant(other)
        return OpAdd(other, self)

    def __sub__(self, other) -> 'Operation':
        """See :class:`OpSubtract`."""
        from .op_subtract import OpSubtract
        from .op_constant import OpConstant
        if not isinstance(other, Operation):
            other = OpConstant(other)
        return OpSubtract(self, other)

    def __rsub__(self, other) -> 'Operation':
        """See :class:`OpSubtract`."""
        from .op_subtract import OpSubtract
        from .op_constant import OpConstant
        if not isinstance(other, Operation):
            other = OpConstant(other)
        return OpSubtract(other, self)

    def __mul__(self, other) -> 'Operation':
        """See :class:`OpMultiply`."""
        from .op_multiply import OpMultiply
        from .op_constant import OpConstant
        if not isinstance(other, Operation):
            other = OpConstant(other)
        return OpMultiply(self, other)

    def __rmul__(self, other) -> 'Operation':
        """See :class:`OpMultiply`."""
        from .op_multiply import OpMultiply
        from .op_constant import OpConstant
        if not isinstance(other, Operation):
            other = OpConstant(other)
        return OpMultiply(other, self)

    def __truediv__(self, other) -> 'Operation':
        """See :class:`OpDivide`."""
        from .op_divide import OpDivide
        from .op_constant import OpConstant
        if not isinstance(other, Operation):
            other = OpConstant(other)
        return OpDivide(self, other)

    def __rtruediv__(self, other) -> 'Operation':
        """See :class:`OpDivide`."""
        from .op_divide import OpDivide
        from .op_constant import OpConstant
        if not isinstance(other, Operation):
            other = OpConstant(other)
        return OpDivide(other, self)

    def __floordiv__(self, other) -> 'Operation':
        return self.__truediv__(other)

    def __rfloordiv__(self, other) -> 'Operation':
        return self.__rtruediv__(other)

    def __neg__(self) -> 'Operation':
        """See :class:`OpNegative`."""
        from .op_negative import OpNegative
        return OpNegative(self)

    def __lt__(self, other):
        """See :class:`OpLess`."""
        from .op_less import OpLess
        from .op_constant import OpConstant
        if not isinstance(other, Operation):
            other = OpConstant(other)
        return OpLess(self, other)

    def __gt__(self, other):
        """See :class:`OpGreater`."""
        from .op_greater import OpGreater
        from .op_constant import OpConstant
        if not isinstance(other, Operation):
            other = OpConstant(other)
        return OpGreater(self, other)

    def __getitem__(self, item) -> 'Operation':
        """See :class:`OpGetitem`."""
        from .op_getitem import OpGetitem
        return OpGetitem(self, item)

    def simplify(self) -> 'Operation':
        from ..simple import simplify
        return simplify(self)

    def __hash__(self):
        return hash(self._op_index)

    def __eq__(self, other: 'Operation'):
        return self._op_index == other._op_index

    def __str__(self):
        return self.name

    def __unicode__(self):
        return self.__str__()
