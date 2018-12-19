from typing import Union, List
import numpy as np
import auto_diff as ad


class Model(ad.layers.Layer):

    def __init__(self,
                 inputs: Union[ad.layers.Input, List[ad.layers.Input]],
                 outputs: Union[ad.layers.Layer, List[ad.layers.Layer]],
                 **kwargs):
        super(Model, self).__init__(**kwargs)
        self._inputs = inputs
        self._outputs = outputs
        self._session = ad.sess.Session()

    def compute_output_shape(self, input_shape):
        if isinstance(self._outputs, list):
            return [output.output_shapes for output in self._outputs]
        return self._outputs.output_shapes

    def build(self, input_shape=None):
        if not self._built:
            pass
        super(Model, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.outputs

    def predict_on_batch(self, x: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        feed_dict = {}
        if isinstance(x, list):
            for i, input_val in enumerate(x):
                feed_dict[self._inputs[i].placeholder] = input_val
        else:
            feed_dict[self._inputs.placeholder] = x
        self._session.prepare()
        if isinstance(self._outputs, list):
            outputs = [self._session.run(output.outputs, feed_dict=feed_dict) for output in self._outputs]
        else:
            outputs = self._session.run(self._outputs.outputs, feed_dict=feed_dict)
        return outputs
