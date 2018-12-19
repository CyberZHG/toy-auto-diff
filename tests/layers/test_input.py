from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestInput(TestCase):

    def test_output(self):
        input_layer = ad.layers.Input(shape=(3, 4))
        model = ad.models.Model(inputs=input_layer, outputs=input_layer)
        val = np.random.random((3, 4))
        self.assertTrue(np.allclose(val, model.predict_on_batch(val)))
