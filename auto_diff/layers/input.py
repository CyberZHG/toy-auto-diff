import auto_diff as ad
from .layer import Layer


class Input(Layer):

    def __init__(self, shape, **kwargs):
        super(Input, self).__init__(**kwargs)
        self.shape = shape
        self.placeholder = ad.placeholder(shape)
        self._outputs = self.call(None)
        self._output_shapes = self.compute_output_shape(None)

    def compute_output_shape(self, input_shape):
        return self.shape

    def call(self, inputs, **kwargs):
        return self.placeholder
