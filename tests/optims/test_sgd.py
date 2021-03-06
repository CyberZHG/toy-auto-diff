from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestSGD(TestCase):

    @staticmethod
    def _create_model():
        input_layer = ad.layers.Input(shape=(None, 5))
        dense_layer = ad.layers.Dense(output_dim=2, activation=ad.acts.softmax)(input_layer)
        model = ad.models.Model(inputs=input_layer, outputs=dense_layer)
        return model

    def _test_fitting(self, model):
        input_vals = np.random.random((2, 5))
        output_vals = np.array([[0.0, 1.0], [1.0, 0.0]])
        for _ in range(5000):
            model.fit_on_batch(input_vals, output_vals)
        actual = np.argmax(model.predict_on_batch(input_vals), axis=-1).tolist()
        self.assertEqual([1.0, 0.0], actual)

    def test_default(self):
        np.random.seed(0xcafe)
        model = self._create_model()
        model.build(
            optimizer=ad.optims.SGD(lr=1e-3),
            losses=ad.losses.cross_entropy,
        )
        self._test_fitting(model)

    def test_momentum(self):
        np.random.seed(0xcafe)
        model = self._create_model()
        model.build(
            optimizer=ad.optims.SGD(momentum=0.9, lr=1e-3),
            losses=ad.losses.cross_entropy,
        )
        self._test_fitting(model)

    def test_decay(self):
        np.random.seed(0xcafe)
        model = self._create_model()
        model.build(
            optimizer=ad.optims.SGD(decay=1e-3, lr=1e-3),
            losses=ad.losses.cross_entropy,
        )
        self._test_fitting(model)

    def test_nesterov(self):
        np.random.seed(0xcafe)
        model = self._create_model()
        model.build(
            optimizer=ad.optims.SGD(lr=1e-3, nesterov=True),
            losses=ad.losses.cross_entropy,
        )
        self._test_fitting(model)

    def test_all(self):
        np.random.seed(0xcafe)
        model = self._create_model()
        model.build(
            optimizer=ad.optims.SGD(momentum=0.9, decay=1e-3, lr=1e-3, nesterov=True),
            losses=ad.losses.cross_entropy,
        )
        self._test_fitting(model)
