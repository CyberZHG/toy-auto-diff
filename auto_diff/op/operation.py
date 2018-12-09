from typing import Union, Mapping, Optional, Sequence
import numpy as np


class Operation(object):
    """Abstract operation for building computing graph."""

    #: The counter for giving each operation a unique index.
    __op_counter = [0]
    #: Collection of existing operations.
    __op_collection = {}

    #: The key for extracting step information from session.
    STEP_KEY = '__step__'

    def __init__(self, **kwargs):
        if not hasattr(self, 'name'):
            if 'name' in kwargs:
                self.name: str = kwargs['name']
            else:
                self.name: str = self._get_name()
        if not hasattr(self, 'shape'):
            self.shape: Sequence = None
            raise NotImplementedError('Shape not defined')
        if not hasattr(self, 'inputs'):
            self.inputs: Sequence['Operation'] = []
        self.gradient: Optional['Operation'] = None
        self._op_index = self.__op_counter[0]
        self.__op_counter[0] += 1
        self._op_name = self._get_op_name()
        self.__op_collection[self] = self
        self._last_step = -1
        self._last_forward = None

    def _get_name(self) -> str:
        """Get the name for display."""
        raise NotImplementedError('Get name not implemented')

    def _get_op_name(self) -> str:
        """Get the name for indexing."""
        raise NotImplementedError('Get operation name not implemented')

    def forward(self, feed_dict: Mapping[Union[str, 'Operation'], np.ndarray] = None) -> np.ndarray:
        """Do the calculations to get the output of the operations.

        :param feed_dict: Contains the real values of placeholders, see :class:`OpPlaceholder`.
        :return: A numpy array.
        """
        if feed_dict is None:
            feed_dict = {}
        if self.STEP_KEY in feed_dict and feed_dict[self.STEP_KEY] == self._last_step:
            return self._last_forward
        output = self._forward(feed_dict)
        if self.STEP_KEY in feed_dict:
            self._last_step = feed_dict[self.STEP_KEY]
            self._last_forward = output
        return output

    def _forward(self, feed_dict: Mapping[Union[str, 'Operation'], np.ndarray]) -> np.ndarray:
        """Forward operation to be implemented."""
        raise NotImplementedError('Forward operation not implemented')

    def backward(self, gradient: 'Operation' = None) -> None:
        """Update gradients recursively.

        :param gradient: Current gradient.
        """
        if gradient is None:
            from .op_constant import OpConstant
            gradient = OpConstant(np.ones(self.shape), name='ones%s' % str(self.shape))
        self.gradient = gradient
        self._backward(gradient)

    def _backward(self, gradient: 'Operation') -> None:
        """Backward operation to be implemented."""
        raise NotImplementedError('Backward operation not implemented')

    def _broadcast_shape(self, x: 'Operation', y: 'Operation'):
        min_dim = min(len(x.shape), len(y.shape))
        shape = []
        for i in range(1, min_dim + 1):
            if x.shape[-i] != 1 and y.shape[-i] != 1 and x.shape[-i] != y.shape[-i]:
                raise ValueError('Cannot broadcast with shape %s and %s' % (str(x.shape), str(y.shape)))
            shape.append(max(x.shape[-i], y.shape[-i]))
        self.shape = tuple(list(x.shape[:-min_dim]) + list(y.shape[:-min_dim]) + list(reversed(shape)))

    def _broadcast_backward(self, gradient: 'Operation'):
        if self.shape == gradient.shape:
            return gradient
        expand_dim = len(gradient.shape) - len(self.shape)
        axis = list(range(expand_dim))
        for i, dim in enumerate(self.shape):
            if self.shape[i] == 1 and gradient.shape[i + expand_dim] > 1:
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

    def __mul__(self, other) -> 'Operation':
        """See :class:`OpMultiply`."""
        from .op_multiply import OpMultiply
        return OpMultiply(self, other)

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
