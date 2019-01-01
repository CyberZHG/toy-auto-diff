from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestOptim(TestCase):

    def test_update_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            input_layer = ad.layers.Input(shape=(None, 5))
            dense_layer = ad.layers.Dense(output_dim=2, activation=ad.acts.softmax)(input_layer)
            model = ad.models.Model(inputs=input_layer, outputs=dense_layer)
            model.build(
                optimizer=ad.optims.Optimizer(lr=1e-3),
                losses=ad.losses.cross_entropy,
            )
            input_vals = np.random.random((2, 5))
            output_vals = np.array([[0.0, 1.0], [1.0, 0.0]])
            for _ in range(5000):
                model.fit_on_batch(input_vals, output_vals)
