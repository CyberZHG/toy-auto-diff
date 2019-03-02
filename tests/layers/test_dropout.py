from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestDropout(TestCase):

    def test_predict_phase(self):
        input_layer = ad.layers.Input(shape=(None, 4))
        dense_layer = ad.layers.Dropout(rate=0.5)(input_layer)
        model = ad.models.Model(inputs=input_layer, outputs=dense_layer)
        x = np.ones((10, 10))
        y = model.predict_on_batch(x)
        self.assertTrue(np.allclose(x, y))

    def test_fit_half(self):
        input_layer = ad.layers.Input(shape=(None, 5))
        drop_layer = ad.layers.Dropout(rate=0.5)(input_layer)
        dense_layer = ad.layers.Dense(output_dim=2, activation=ad.acts.softmax)(drop_layer)
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

    def test_fit_zero(self):
        input_layer = ad.layers.Input(shape=(None, 5))
        drop_layer = ad.layers.Dropout(rate=0.0)(input_layer)
        dense_layer = ad.layers.Dense(output_dim=2, activation=ad.acts.softmax)(drop_layer)
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

    def test_fit_noise_shape(self):
        input_layer = ad.layers.Input(shape=(None, 5))
        drop_layer = ad.layers.Dropout(rate=0.5, noise_shape=(1, 5))(input_layer)
        dense_layer = ad.layers.Dense(output_dim=2, activation=ad.acts.softmax)(drop_layer)
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
