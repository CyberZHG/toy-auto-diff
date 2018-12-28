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
        self._optimizer = None
        self._losses = None
        self._loss = None
        self._layers = None
        self._output_placeholders = None
        self._session = ad.sess.Session()

    def compute_output_shape(self, input_shape):
        if isinstance(self._outputs, list):
            return [output.output_shapes for output in self._outputs]
        return self._outputs.output_shapes

    def build(self, optimizer: ad.optims.Optimizer, losses):
        if not self._built:
            self._optimizer = optimizer
            self._losses = losses
            self._layers = {}

            def _collect_all_layers(layer):
                if layer is None:
                    return
                if layer in self._layers:
                    return
                self._layers[layer] = layer
                if isinstance(layer.inputs, list):
                    for input_layer in layer.inputs:
                        _collect_all_layers(input_layer)
                else:
                    _collect_all_layers(layer.inputs)

            if isinstance(self.outputs, list):
                for output in self.outputs:
                    _collect_all_layers(output)
            else:
                _collect_all_layers(self.outputs)

            for layer in self._layers.values():
                self._trainable_weights += layer.trainable_weights
                self._non_trainable_weights += layer.non_trainable_weights

            self._loss = 0.0
            if isinstance(self.outputs, list):
                self._output_placeholders = []
                for i, output in enumerate(self.outputs):
                    output_shapes = output.output_shapes
                    if isinstance(output_shapes, list):
                        self._output_placeholders.append([])
                        for j, output_shape in enumerate(output_shapes):
                            output_placeholder = ad.OpPlaceholder(output_shape)
                            self._output_placeholders[-1].append(output_placeholder)
                            self._loss = self._loss + losses(output_placeholder, self.outputs[i].outputs[j])
                    else:
                        output_placeholder = ad.OpPlaceholder(output_shapes)
                        self._output_placeholders.append(output_placeholder)
                        self._loss = self._loss + losses(output_placeholder, self.outputs[i].outputs)
            else:
                output_shapes = self.outputs.output_shapes
                if isinstance(output_shapes, list):
                    self._output_placeholders = []
                    for i, output_shape in enumerate(output_shapes):
                        output_placeholder = ad.OpPlaceholder(output_shapes)
                        self._output_placeholders.append(output_placeholder)
                        self._loss = self._loss + losses(output_placeholder, self.outputs.outputs[i])
                else:
                    output_placeholder = ad.OpPlaceholder(output_shapes)
                    self._output_placeholders = output_placeholder
                    self._loss = self._loss + losses(output_placeholder, self.outputs.outputs)

        super(Model, self).build(None)

    def call(self, inputs, **kwargs):
        return self.outputs

    def fit_on_batch(self,
                     x: Union[np.ndarray, List[np.ndarray]],
                     y: Union[np.ndarray, List[np.ndarray]]):
        # TODO: Multiple outputs
        feed_dict = {ad.Operation.KEY_TRAINING: True}
        if isinstance(x, list):
            for i, input_val in enumerate(x):
                feed_dict[self._inputs[i].placeholder] = input_val
        else:
            feed_dict[self._inputs.placeholder] = x
        feed_dict[self._output_placeholders] = y
        self._session.prepare()
        self._session.run(self._loss, feed_dict=feed_dict)
        for weight in self.trainable_weights:
            weight.clear_gradient()
        self._loss.backward()
        self._optimizer.update(self.trainable_weights, self._session)

    def predict_on_batch(self, x: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        feed_dict = {ad.Operation.KEY_TRAINING: False}
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
