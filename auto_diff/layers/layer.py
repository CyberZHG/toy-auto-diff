from typing import Union, List
import auto_diff as ad


class Layer(object):

    def __init__(self, **kwargs):
        self._built = False
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._inputs = None
        self._outputs = None
        self._input_shapes = None
        self._output_shapes = None

    def build(self, input_shape):
        self._built = True

    def compute_output_shape(self, input_shape):
        raise NotImplementedError('`compute_output_shape` not implemented')

    def call(self, inputs, **kwargs):
        raise NotImplementedError('`call` not implemented')

    def __call__(self, inputs: Union['Layer', List['Layer']], **kwargs):
        if self._outputs is None:
            self._inputs = inputs
            self._input_shapes = inputs.output_shapes
            self.build(self._input_shapes)
            self._output_shapes = self.compute_output_shape(self._input_shapes)
            self._outputs = self.call(inputs.outputs, **kwargs)
        return self

    @property
    def trainable_weights(self) -> List[ad.OpVariable]:
        return self._trainable_weights

    @property
    def non_trainable_weights(self) -> List[ad.OpVariable]:
        return self._non_trainable_weights

    def get_weights(self) -> List[ad.OpVariable]:
        return self.trainable_weights + self.non_trainable_weights

    def add_weight(self, name, shape, initializer=None, trainable=True) -> ad.OpVariable:
        var = ad.variable(initializer, shape=shape, name=name)
        if trainable:
            self._trainable_weights.append(var)
        else:
            self._non_trainable_weights.append(var)
        return var

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def input_shapes(self):
        return self._input_shapes

    @property
    def output_shapes(self):
        return self._output_shapes
