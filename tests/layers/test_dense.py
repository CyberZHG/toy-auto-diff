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

    def test_fit(self):
        input_layer = ad.layers.Input(shape=(None, 5))
        dense_layer = ad.layers.Dense(output_dim=2, activation=ad.acts.softmax)(input_layer)
        model = ad.models.Model(inputs=input_layer, outputs=dense_layer)
        model.build(
            optimizer=ad.optims.SGD(momentum=0.9, decay=1e-3, lr=1e-3, nesterov=True),
            losses=ad.losses.cross_entropy,
        )

        input_vals = np.random.random((2, 5))
        output_vals = np.array([[0.0, 1.0], [1.0, 0.0]])
        for _ in range(5000):
            model.fit_on_batch(input_vals, output_vals)
        actual = np.argmax(model.predict_on_batch(input_vals), axis=-1).tolist()
        self.assertEqual([1.0, 0.0], actual)
