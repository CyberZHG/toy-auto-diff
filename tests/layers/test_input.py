from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestInput(TestCase):

    def test_output(self):
        input_layer = ad.layers.Input(shape=(3, 4))
        model = ad.models.Model(inputs=input_layer, outputs=input_layer)
        val = np.random.random((3, 4))
        self.assertTrue(np.allclose(val, model.predict_on_batch(val)))

    def test_multi_input(self):
        input_layer_1 = ad.layers.Input(shape=(3, 4))
        input_layer_2 = ad.layers.Input(shape=(2, 3))
        model = ad.models.Model(inputs=[input_layer_1, input_layer_2], outputs=[input_layer_1, input_layer_2])
        val_1 = np.random.random((3, 4))
        val_2 = np.random.random((2, 3))
        outputs = model.predict_on_batch([val_1, val_2])
        self.assertTrue(np.allclose(val_1, outputs[0]))
        self.assertTrue(np.allclose(val_2, outputs[1]))
