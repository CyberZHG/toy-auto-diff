from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestBatchNorm(TestCase):

    def test_no_moving(self):
        input_layer = ad.layers.Input(shape=(None, 5))
        normal_layer = ad.layers.BatchNorm()(input_layer)
        dense_layer = ad.layers.Dense(output_dim=2, activation=ad.acts.softmax)(normal_layer)
        model = ad.models.Model(inputs=input_layer, outputs=dense_layer)
        model.build(
            optimizer=ad.optims.Adam(),
            losses=ad.losses.cross_entropy,
        )

        input_vals = np.random.random((2, 5))
        first = model.predict_on_batch(input_vals)
        second = model.predict_on_batch(input_vals)
        self.assertTrue(np.allclose(first, second))

    def test_fit(self):
        np.random.seed(0xcafe)
        input_layer = ad.layers.Input(shape=(None, 5))
        normal_layer = ad.layers.BatchNorm()(input_layer)
        dense_layer = ad.layers.Dense(output_dim=2, activation=ad.acts.softmax)(normal_layer)
        model = ad.models.Model(inputs=input_layer, outputs=dense_layer)
        model.build(
            optimizer=ad.optims.Adam(),
            losses=ad.losses.cross_entropy,
        )

        input_vals = np.random.random((2, 5))
        output_vals = np.array([[0.0, 1.0], [1.0, 0.0]])
        for _ in range(5000):
            model.fit_on_batch(input_vals, output_vals)
        actual = np.argmax(model.predict_on_batch(input_vals), axis=-1).tolist()
        self.assertEqual([1.0, 0.0], actual)
