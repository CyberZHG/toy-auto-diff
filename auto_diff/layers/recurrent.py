import auto_diff as ad
import numpy as np
from .layer import Layer


class LSTM(Layer):

    def __init__(self,
                 units: int,
                 return_sequences=False,
                 use_bias: bool = True,
                 **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.use_bias = use_bias
        self.wx, self.wh, self.b = None, None, None

    def build(self, input_shape):
        if not self._built:
            self.wx = self.add_weight(
                name='Wx',
                shape=(input_shape[-1], self.units * 4),
                initializer=np.random.random,
                trainable=True,
            )
            self.wh = self.add_weight(
                name='Wh',
                shape=(self.units, self.units * 4),
                initializer=np.random.random,
                trainable=True,
            )
            if self.use_bias:
                self.b = self.add_weight(
                    name='b',
                    shape=(self.units * 4,),
                    initializer=np.zeros,
                    trainable=True,
                )
        super(LSTM, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape[:2] + (self.units,)
        return input_shape[0], self.units

    def call(self, inputs, **kwargs):
        initial_val = ad.dot(ad.zeros_like(inputs)[:, 0, :], ad.zeros_like(self.wx[:, :self.units]))
        outputs = ad.while_loop(
            lambda body_inputs: ad.less(body_inputs[0], ad.shape(inputs)[1]),
            lambda x: self.step(inputs, x),
            [ad.variable(0.0), initial_val, initial_val],
            output_index=-1,
        )
        if self.return_sequences:
            return outputs.transpose(axes=[1, 0, 2])
        return outputs[-1]

    def step(self, inputs, body_inputs):
        index, cell_state, output = body_inputs
        linear_sum = ad.dot(inputs[:, index], self.wx) + ad.dot(cell_state, self.wh)
        if self.use_bias:
            linear_sum += self.b
        forget_gate = ad.acts.sigmoid(linear_sum[:, :self.units])
        input_gate = ad.acts.sigmoid(linear_sum[:, self.units:self.units * 2])
        output_gate = ad.acts.sigmoid(linear_sum[:, self.units * 2:self.units * 3])
        cell_state_inner = ad.tanh(linear_sum[:, self.units * 3:])
        new_cell_state = forget_gate * cell_state + input_gate * cell_state_inner
        new_output = output_gate * ad.tanh(new_cell_state)
        return index + 1.0, new_cell_state, new_output


class GRU(Layer):

    def __init__(self,
                 units: int,
                 return_sequences=False,
                 use_bias: bool = True,
                 **kwargs):
        super(GRU, self).__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.use_bias = use_bias
        self.wx, self.wh, self.b = None, None, None

    def build(self, input_shape):
        if not self._built:
            self.wx = self.add_weight(
                name='Wx',
                shape=(input_shape[-1], self.units * 3),
                initializer=np.random.random,
                trainable=True,
            )
            self.wh = self.add_weight(
                name='Wh',
                shape=(self.units, self.units * 3),
                initializer=np.random.random,
                trainable=True,
            )
            if self.use_bias:
                self.b = self.add_weight(
                    name='b',
                    shape=(self.units * 3,),
                    initializer=np.zeros,
                    trainable=True,
                )
        super(GRU, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape[:2] + (self.units,)
        return input_shape[0], self.units

    def call(self, inputs, **kwargs):
        initial_val = ad.dot(ad.zeros_like(inputs)[:, 0, :], ad.zeros_like(self.wx[:, :self.units]))
        outputs = ad.while_loop(
            lambda body_inputs: ad.less(body_inputs[0], ad.shape(inputs)[1]),
            lambda x: self.step(inputs, x),
            [ad.variable(0.0), initial_val],
            output_index=-1,
        )
        if self.return_sequences:
            return outputs.transpose(axes=[1, 0, 2])
        return outputs[-1]

    def step(self, inputs, body_inputs):
        index, output = body_inputs
        linear_sum = ad.dot(inputs[:, index], self.wx[:, :self.units * 2]) + ad.dot(output, self.wh[:, :self.units * 2])
        if self.use_bias:
            linear_sum += self.b[:self.units * 2]
        update_gate = ad.acts.sigmoid(linear_sum[:, :self.units])
        reset_gate = ad.acts.sigmoid(linear_sum[:, self.units:self.units * 2])
        output_inner = ad.dot(inputs[:, index], self.wx[:, self.units * 2:]) +\
            ad.dot(reset_gate * output, self.wh[:, self.units * 2:])
        if self.use_bias:
            output_inner += self.b[self.units * 2:]
        new_output = (1.0 - update_gate) * output + update_gate * ad.tanh(output_inner)
        return index + 1.0, new_output
