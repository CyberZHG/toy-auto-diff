from unittest import TestCase
import auto_diff as ad


class TestLayer(TestCase):

    def test_compute_output_shape_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            ad.layers.Layer().compute_output_shape(input_shape=(1, 2, 5))

    def test_call_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            ad.layers.Layer().call([])
