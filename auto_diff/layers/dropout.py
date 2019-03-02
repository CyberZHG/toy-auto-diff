import auto_diff as ad
from .layer import Layer


class Dropout(Layer):

    def __init__(self,
                 rate,
                 noise_shape=None,
                 **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.in_train_phase = ad.in_train_phase()

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        if 0.0 < self.rate < 1.0:
            if self.noise_shape is not None:
                noise_shape = self.noise_shape
            else:
                noise_shape = ad.shape(inputs)
            return ad.where(self.in_train_phase,
                            inputs * (ad.random(noise_shape) > self.rate),
                            inputs)

        return inputs
