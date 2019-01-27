import auto_diff as ad
from .layer import Layer


class Dense(Layer):

    def __init__(self,
                 output_dim,
                 use_bias=True,
                 kernel_initializer=ad.inits.glorot_normal,
                 bias_initializer=ad.inits.zeros,
                 activation=None,
                 **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.w, self.b = None, None

    def build(self, input_shape):
        if not self._built:
            self.w = self.add_weight(
                name='W',
                shape=(input_shape[-1], self.output_dim),
                initializer=self.kernel_initializer,
                trainable=True,
            )
            if self.use_bias:
                self.b = self.add_weight(
                    name='b',
                    shape=(self.output_dim,),
                    initializer=self.bias_initializer,
                    trainable=True,
                )
        super(Dense, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)

    def call(self, inputs, **kwargs):
        y = ad.dot(inputs, self.w)
        if self.use_bias:
            y += self.b
        if self.activation is not None:
            y = self.activation(y)
        return y
