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
            self.shape: tuple = None
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

    def transpose(self, axes: Optional[Sequence[int]] = None) -> 'Operation':
        """See :class:`OpTranspose`."""
        from .op_transpose import OpTranspose
        return OpTranspose(self, axes)

    def reshape(self, shape: Sequence[int]) -> 'Operation':
        """See :class:`OpReshape`."""
        from .op_reshape import OpReshape
        return OpReshape(self, shape)

    def flatten(self) -> 'Operation':
        """See :class:`OpFlatten`."""
        from .op_flatten import OpFlatten
        return OpFlatten(self)

    def expand_dims(self, axis: Optional[int] = None) -> 'Operation':
        """See :class:`OpExpandDims`."""
        from .op_expand_dims import OpExpandDims
        return OpExpandDims(self, axis)

    def squeeze(self, axis=None) -> 'Operation':
        """See :class:`OpExpandDims`."""
        from .op_squeeze import OpSqueeze
        return OpSqueeze(self, axis)

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
