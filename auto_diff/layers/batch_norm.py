import auto_diff as ad
from .layer import Layer


class BatchNorm(Layer):

    def __init__(self,
                 momentum=0.99,
                 epsilon=1e-3,
                 scale=True,
                 center=True,
                 beta_initializer=ad.inits.zeros,
                 gamma_initializer=ad.inits.ones,
                 **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = scale
        self.center = center
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.gamma, self.beta = None, None
        self.moving_mean, self.moving_var = None, None

    def build(self, input_shape):
        if not self._built:
            if self.scale:
                self.gamma = self.add_weight(
                    name='gamma',
                    shape=(input_shape[-1],),
                    initializer=self.gamma_initializer,
                    trainable=True,
                )
            if self.center:
                self.beta = self.add_weight(
                    name='beta',
                    shape=(input_shape[-1],),
                    initializer=self.beta_initializer,
                    trainable=True,
                )
            self.moving_mean = self.add_weight(
                name='moving_mean',
                shape=(input_shape[-1],),
                initializer=self.gamma_initializer,
                trainable=False,
            )
            self.moving_var = self.add_weight(
                name='moving_var',
                shape=(input_shape[-1],),
                initializer=self.beta_initializer,
                trainable=False,
            )
        super(BatchNorm, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call_moving(self, inputs, moving_mean, moving_var):
        normal = (inputs - moving_mean) / ad.sqrt(moving_var + self.epsilon)
        if self.scale:
            normal *= self.gamma
        if self.center:
            normal += self.beta
        return normal

    def call(self, inputs, **kwargs):
        sum_axis = tuple(range(len(inputs.shape) - 1))
        mean = ad.mean(inputs, axis=sum_axis, keepdims=True)
        var = ad.mean(ad.square(inputs - mean), axis=sum_axis, keepdims=True)
        moving_mean = self.momentum * self.moving_mean + (1.0 - self.momentum) * mean
        moving_var = self.momentum * self.moving_var + (1.0 - self.momentum) * var
        self.add_update(self.moving_mean, ad.squeeze(moving_mean, axis=sum_axis))
        self.add_update(self.moving_var, ad.squeeze(moving_var, axis=sum_axis))
        return ad.where(
            ad.in_train_phase(),
            self.call_moving(inputs, moving_mean, moving_var),
            self.call_moving(inputs, self.moving_mean, self.moving_var),
        )
