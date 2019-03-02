from typing import Union, Sequence
import auto_diff as ad
from .layer import Layer


class Conv2D(Layer):

    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Sequence],
                 strides: Union[int, Sequence] = 1,
                 dilation_rate: Union[int, Sequence] = 1,
                 padding='valid',
                 kernel_initializer=ad.inits.glorot_normal,
                 bias_initializer=ad.inits.zeros,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.filters = filters
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = tuple(strides)
        if isinstance(dilation_rate, int):
            self.dilation_rate = (dilation_rate, dilation_rate)
        else:
            self.dilation_rate = tuple(dilation_rate)
        self.dilated_kernel_size = (
            self.dilation_rate[0] * (self.kernel_size[0] - 1) + 1,
            self.dilation_rate[1] * (self.kernel_size[1] - 1) + 1,
        )
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        if padding == 'valid':
            self.pad_width = (0, 0)
        elif padding == 'same':
            self.pad_width = (
                self.dilated_kernel_size[0] // 2,
                self.dilated_kernel_size[1] // 2,
            )
        else:
            raise NotImplementedError('Unknown padding: %s' % str(padding))
        self.activation = activation
        self.w, self.b = None, None

    def build(self, input_shape):
        if not self._built:
            self.w = self.add_weight(
                name='W',
                shape=(self.kernel_size[0] * self.kernel_size[1] * input_shape[-1], self.filters),
                initializer=self.kernel_initializer,
                trainable=True,
            )
            if self.use_bias:
                self.b = self.add_weight(
                    name='b',
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    trainable=True,
                )
        super(Conv2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape
        if height is None:
            new_height = None
        else:
            new_height = (height + 2 * self.pad_width[0] - self.dilated_kernel_size[0]) // self.strides[0] + 1
        if width is None:
            new_width = None
        else:
            new_width = (width + 2 * self.pad_width[1] - self.dilated_kernel_size[1]) // self.strides[1] + 1
        return batch_size, new_height, new_width, self.filters

    def call(self, inputs, **kwargs):
        padded = ad.pad(inputs, ((0,), (self.pad_width[0],), (self.pad_width[1],), (0,)))
        batch_size = ad.shape(inputs)[0]
        reshaped = ad.map_fn(lambda i: self.call_batch(padded, i), ad.arange(batch_size))
        y = ad.dot(reshaped, self.w)
        if self.use_bias:
            y += self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

    def call_batch(self, padded: ad.Operation, i: int):
        height = ad.shape(padded)[1]
        new_height = (height - self.dilated_kernel_size[0]) // self.strides[0] + 1
        return ad.map_fn(lambda r: self.call_row(padded, i, r), ad.arange(new_height))

    def call_row(self, padded: ad.Operation, i: int, r: int):
        width = ad.shape(padded)[2]
        new_width = (width - self.dilated_kernel_size[1]) // self.strides[1] + 1
        return ad.map_fn(lambda c: self.call_column(padded, i, r, c), ad.arange(new_width))

    def call_column(self, padded: ad.Operation, i: int, r: int, c: int):
        block = padded[
            i,
            r * self.strides[0]:r * self.strides[0] + self.dilated_kernel_size[0]:self.dilation_rate[0],
            c * self.strides[1]:c * self.strides[1] + self.dilated_kernel_size[1]:self.dilation_rate[1],
            :,
        ]
        return block.flatten()
