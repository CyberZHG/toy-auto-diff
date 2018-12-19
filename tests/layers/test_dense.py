from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestDense(TestCase):

    def test_output(self):
        input_layer = ad.layers.Input(shape=(None, 4))
        dense_layer = ad.layers.Dense(output_dim=3, activation=ad.acts.relu)(input_layer)
        model = ad.models.Model(inputs=input_layer, outputs=dense_layer)
        val = np.random.random((3, 4))
        output = model.predict_on_batch(val)
        self.assertEqual((3, 3), output.shape)
