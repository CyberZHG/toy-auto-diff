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
                shape=(4, input_shape[-1], self.units),
                initializer=np.random.random,
                trainable=True,
            )
            self.wh = self.add_weight(
                name='Wh',
                shape=(4, self.units, self.units),
                initializer=np.random.random,
                trainable=True,
            )
            if self.use_bias:
                self.b = self.add_weight(
                    name='b',
                    shape=(4, self.units),
                    initializer=np.zeros,
                    trainable=True,
                )
        super(LSTM, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape[:2] + (self.units,)
        return input_shape[0], self.units

    def call(self, inputs, **kwargs):
        initial_val = ad.dot(ad.zeros_like(inputs)[:, 0, :], ad.zeros_like(self.wx[0]))
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
        forget_gate = ad.dot(inputs[:, index], self.wx[0]) + ad.dot(cell_state, self.wh[0])
        input_gate = ad.dot(inputs[:, index], self.wx[1]) + ad.dot(cell_state, self.wh[1])
        output_gate = ad.dot(inputs[:, index], self.wx[2]) + ad.dot(cell_state, self.wh[2])
        cell_state_hat = ad.dot(inputs[:, index], self.wx[3]) + ad.dot(cell_state, self.wh[3])
        if self.use_bias:
            forget_gate += self.b[0]
            input_gate += self.b[1]
            output_gate += self.b[2]
            cell_state_hat += self.b[3]
        forget_gate = ad.acts.sigmoid(forget_gate)
        input_gate = ad.acts.sigmoid(input_gate)
        output_gate = ad.acts.sigmoid(output_gate)
        cell_state_hat = ad.tanh(cell_state_hat)
        new_cell_state = forget_gate * cell_state + input_gate * cell_state_hat
        new_output = output_gate * ad.tanh(new_cell_state)
        return index + 1.0, new_cell_state, new_output
