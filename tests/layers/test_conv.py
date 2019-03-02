from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestConv2D(TestCase):

    def test_output_shape_valid(self):
        input_layer = ad.layers.Input(shape=(None, None, None, 3))
        conv_layer = ad.layers.Conv2D(kernel_size=3, filters=5, padding='valid', activation=ad.acts.relu)(input_layer)
        model = ad.models.Model(inputs=input_layer, outputs=conv_layer)
        val = np.random.random((1, 5, 5, 3))
        output = model.predict_on_batch(val)
        self.assertEqual((1, 3, 3, 5), output.shape)

    def test_output_shape_same(self):
        input_layer = ad.layers.Input(shape=(None, 5, 5, 3))
        conv_layer = ad.layers.Conv2D(
            kernel_size=(3, 5),
            strides=(1, 1),
            filters=4,
            padding='same',
            activation=ad.acts.relu,
        )(input_layer)
        model = ad.models.Model(inputs=input_layer, outputs=conv_layer)
        val = np.random.random((2, 5, 5, 3))
        output = model.predict_on_batch(val)
        self.assertEqual((2, 5, 5, 4), output.shape)

    def test_invalid_padding(self):
        with self.assertRaises(NotImplementedError):
            input_layer = ad.layers.Input(shape=(None, 5, 5, 3))
            ad.layers.Conv2D(
                kernel_size=(3, 5),
                strides=(1, 1),
                filters=4,
                padding='invalid',
                activation=ad.acts.relu,
            )(input_layer)

    def test_dilation_same_shape(self):
        input_layer = ad.layers.Input(shape=(None, 5, 5, 3))
        conv_layer = ad.layers.Conv2D(
            kernel_size=(3, 5),
            strides=(1, 1),
            filters=4,
            dilation_rate=(2, 3),
            padding='same',
            activation=ad.acts.relu,
        )(input_layer)
        model = ad.models.Model(inputs=input_layer, outputs=conv_layer)
        val = np.random.random((2, 5, 5, 3))
        output = model.predict_on_batch(val)
        self.assertEqual((2, 5, 5, 4), output.shape)

    def test_dilation_valid_shape(self):
        input_layer = ad.layers.Input(shape=(None, 7, 7, 3))
        conv_layer = ad.layers.Conv2D(
            kernel_size=3,
            strides=2,
            filters=4,
            dilation_rate=2,
            padding='same',
            activation=ad.acts.relu,
        )(input_layer)
        model = ad.models.Model(inputs=input_layer, outputs=conv_layer)
        val = np.random.random((2, 5, 5, 3))
        output = model.predict_on_batch(val)
        self.assertEqual((2, 3, 3, 4), output.shape)

    def test_fit(self):
        np.random.seed(0xcafe)
        input_layer = ad.layers.Input(shape=(None, None, None, 2))
        conv_layer = ad.layers.Conv2D(kernel_size=3, filters=2, padding='same', activation=ad.acts.relu)(input_layer)
        model = ad.models.Model(inputs=input_layer, outputs=conv_layer)
        model.build(
            optimizer=ad.optims.Adam(lr=1e-3),
            losses=ad.losses.mean_square_error,
        )

        input_vals = np.random.random((1, 3, 3, 2))
        outputs_vals = input_vals * 0.3 + 0.5
        for _ in range(500):
            model.fit_on_batch(input_vals, outputs_vals)
        actual_0 = model.predict_on_batch(input_vals)
        diff_0 = np.abs(outputs_vals - actual_0)
        for _ in range(500):
            model.fit_on_batch(input_vals, outputs_vals)
        actual_1 = model.predict_on_batch(input_vals)
        diff_1 = np.abs(outputs_vals - actual_1)
        self.assertTrue(np.mean(diff_0 - diff_1) > 0, (diff_0, diff_1, diff_0 - diff_1))
