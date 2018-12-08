from typing import Union, Mapping, Callable, Optional, Sequence
import numpy as np

__all__ = ['Operation', 'OpConstant', 'OpPlaceholder', 'OpVariable', 'OpTranspose']


class Operation(object):

    __op_counter = [0]
    __op_collection = {}

    STEP_KEY = '__step__'

    def __init__(self, **kwargs):
        if not hasattr(self, 'name'):
            if 'name' in kwargs:
                self.name = kwargs['name']
            else:
                self.name = self._get_name()
        if not hasattr(self, 'shape'):
            self.shape = None
        self.gradient = None
        self._op_index = self.__op_counter[0]
        self.__op_counter[0] += 1
        self._op_name = self._get_op_name()
        self.__op_collection[self] = self
        self._last_step = -1
        self._last_forward = None

    def _get_name(self) -> str:
        raise NotImplementedError('Get name not implemented')

    def _get_op_name(self) -> str:
        raise NotImplementedError('Get operation name not implemented')

    def forward(self, feed_dict: Mapping[Union[str, 'OpPlaceholder'], np.ndarray] = None) -> np.ndarray:
        if feed_dict is None:
            feed_dict = {}
        if self.STEP_KEY in feed_dict and feed_dict[self.STEP_KEY] == self._last_step:
            return self._last_forward
        output = self._forward(feed_dict)
        if self.STEP_KEY in feed_dict:
            self._last_step = feed_dict[self.STEP_KEY]
            self._last_forward = output
        return output

    def _forward(self, feed_dict: Mapping[Union[str, 'OpPlaceholder'], np.ndarray]) -> np.ndarray:
        raise NotImplementedError('Forward operation not implemented')

    def backward(self, gradient: 'Operation' = None) -> str:
        if gradient is None:
            gradient = OpConstant(np.ones(self.shape), name='ones%s' % str(self.shape))
        self.gradient = gradient
        self._backward(gradient)

    def _backward(self, gradient: 'Operation') -> None:
        raise NotImplementedError('Backward operation not implemented')

    def __hash__(self):
        return hash(self._op_index)

    def __eq__(self, other: 'Operation'):
        return self._op_index == other._op_index

    def __str__(self):
        return self.name

    def __unicode__(self):
        return self.__str__()


class OpConstant(Operation):

    def __init__(self, x: Union[int, float, list, np.ndarray], **kwargs):
        if isinstance(x, int):
            x = float(x)
        if not np.isscalar(x) and not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float64)
        self.x = x
        if np.isscalar(x):
            self.shape = (1,)
        else:
            self.shape = x.shape
        super(OpConstant, self).__init__(**kwargs)

    def _get_name(self) -> str:
        if np.isscalar(self.x):
            return str(self.x)
        return 'C%s' % str(self.x.shape)

    def _get_op_name(self) -> str:
        return 'c_%d' % self._op_index

    def _forward(self, feed_dict: Mapping[Union[str, 'OpPlaceholder'], np.ndarray]) -> np.ndarray:
        return self.x

    def _backward(self, gradient: 'Operation') -> None:
        pass


class OpPlaceholder(Operation):

    def __init__(self, shape: tuple, **kwargs):
        self.shape = shape
        super(OpPlaceholder, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return 'X%s' % str(self.shape)

    def _get_op_name(self) -> str:
        return 'x_%d' % self._op_index

    def _forward(self, feed_dict: Mapping[Union[str, 'OpPlaceholder'], np.ndarray]):
        return feed_dict[self]

    def _backward(self, gradient: 'Operation') -> None:
        pass


class OpVariable(Operation):

    def __init__(self, initializer: Union[Callable, int, float, list, np.ndarray], shape: tuple = None, **kwargs):
        if callable(initializer):
            self.x = initializer(shape)
        else:
            self.x = np.array(initializer, dtype=np.float64)
        self.shape = self.x.shape
        super(OpVariable, self).__init__(**kwargs)

    def update(self, value: Union[int, float, list, np.ndarray]) -> None:
        value = np.array(value)
        if self.x.shape != value.shape:
            raise ValueError('The shape of two tensors should be equal, '
                             'got %s and %s' % (str(self.x.shape), str(value.shape)))
        self.x = value

    def _get_name(self) -> str:
        return 'W%s' % str(self.x.shape)

    def _get_op_name(self) -> str:
        return 'w_%d' % self._op_index

    def _forward(self, feed_dict: Mapping[Union[str, 'OpPlaceholder'], np.ndarray]) -> np.ndarray:
        return self.x

    def _backward(self, gradient: 'Operation') -> None:
        pass


class OpTranspose(Operation):

    def __init__(self, x: Operation, axes: Optional[Sequence[int]] = None, **kwargs):
        self.x = x
        self.axes = axes
        if axes is None:
            self.reverse_axes = None
            self.shape = tuple(reversed(x.shape))
        else:
            self.reverse_axes = [0] * len(axes)
            for i, axis in enumerate(axes):
                self.reverse_axes[axis] = i
            self.shape = tuple(x.shape[axis] for axis in axes)
        super(OpTranspose, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return '(%s)^T' % str(self.x)

    def _get_op_name(self) -> str:
        if self.axes is None:
            return 'transpose(%s)' % self.x._get_op_name()
        return 'tranpose(%s, axes=%s)' % (self.x._get_op_name(), str(self.axes))

    def _forward(self, feed_dict: Mapping[Union[str, 'OpPlaceholder'], np.ndarray]) -> np.ndarray:
        return np.transpose(self.x.forward(feed_dict), axes=self.axes)

    def _backward(self, gradient: 'Operation') -> None:
        self.gradient = OpTranspose(gradient, axes=self.reverse_axes)
        self.x.backward(self.gradient)
