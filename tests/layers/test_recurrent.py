from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestLSTM(TestCase):

    def test_output_last(self):
        input_layer = ad.layers.Input(shape=(None, None, 3))
        lstm_layer = ad.layers.LSTM(units=5)(input_layer)
        model = ad.models.Model(inputs=input_layer, outputs=lstm_layer)
        val = np.random.random((2, 7, 3))
        output = model.predict_on_batch(val)
        self.assertEqual((2, 5), output.shape)

    def test_output_seq(self):
        input_layer = ad.layers.Input(shape=(None, None, 3))
        lstm_layer = ad.layers.LSTM(units=5, return_sequences=True)(input_layer)
        model = ad.models.Model(inputs=input_layer, outputs=lstm_layer)
        val = np.random.random((2, 7, 3))
        output = model.predict_on_batch(val)
        self.assertEqual((2, 7, 5), output.shape)

    def test_fit(self):
        input_layer = ad.layers.Input(shape=(None, None, 3))
        lstm_layer = ad.layers.LSTM(units=7, return_sequences=True)(input_layer)
        lstm_layer = ad.layers.LSTM(units=5)(lstm_layer)
        model = ad.models.Model(inputs=input_layer, outputs=lstm_layer)
        model.build(
            optimizer=ad.optims.SGD(lr=1e-3),
            losses=ad.losses.mean_square_error,
        )

        input_vals = np.random.random((1, 4, 3))
        outputs_vals = np.mean(input_vals, axis=1).dot(np.random.random((3, 5)))
        for _ in range(1):
            model.fit_on_batch(input_vals, outputs_vals)
        actual_0 = model.predict_on_batch(input_vals)
        diff_0 = np.abs(outputs_vals - actual_0)
        for _ in range(1):
            model.fit_on_batch(input_vals, outputs_vals)
        actual_1 = model.predict_on_batch(input_vals)
        diff_1 = np.abs(outputs_vals - actual_1)
        self.assertTrue(np.mean(diff_0 - diff_1) > 0, (diff_0, diff_1, diff_0 - diff_1))
