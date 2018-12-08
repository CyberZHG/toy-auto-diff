import numpy as np

__all__ = ['Operation', 'OpConstant']


class Operation(object):

    __op_counter = 0
    __op_collection = {}

    STEP_KEY = '__step__'

    def __init__(self, **kwargs):
        if not hasattr(self, 'name'):
            if 'name' in kwargs:
                self.name = kwargs['name']
            else:
                self.name = self._get_name()
        self._op_index = self.__op_counter
        self.__op_counter += 1
        self._op_name = self._get_op_name()
        self.__op_collection[self] = self
        self._last_step = -1
        self._last_forward = None

    def _get_name(self):
        raise NotImplementedError('Get name not implemented')

    def _get_op_name(self):
        raise NotImplementedError('Get operation name not implemented')

    def forward(self, feed_dict: dict = None):
        if feed_dict is None:
            feed_dict = {}
        if self.STEP_KEY in feed_dict and feed_dict[self.STEP_KEY] == self._last_step:
            return self._last_forward
        output = self._forward(feed_dict)
        if self.STEP_KEY in feed_dict:
            self._last_step = feed_dict[self.STEP_KEY]
            self._last_forward = output
        return output

    def _forward(self, feed_dict: dict):
        raise NotImplementedError('Forward operation not implemented')

    def backward(self, gradient: 'Operation' = None):
        if gradient is None:
            gradient = OpConstant(1.0)
        self._backward(gradient)

    def _backward(self, gradient: 'Operation'):
        raise NotImplementedError('Backward operation not implemented')

    def __hash__(self):
        return hash(self._op_index)

    def __eq__(self, other: 'Operation'):
        return self._op_index == other._op_index

    def __str__(self):
        return self.name

    def __unicode__(self):
        return self.name


class OpConstant(Operation):

    def __init__(self, x, **kwargs):
        if not np.isscalar(x) and not isinstance(x, np.ndarray):
            x = np.asarray(x)
        self.x = x
        super(OpConstant, self).__init__(**kwargs)

    def _get_name(self):
        if np.isscalar(self.x):
            return str(self.x)
        return 'C%s' % str(self.x.shape)

    def _get_op_name(self):
        return 'c_%d' % self._op_index

    def _forward(self, feed_dict: dict):
        return self.x

    def _backward(self, gradient: 'Operation'):
        pass
