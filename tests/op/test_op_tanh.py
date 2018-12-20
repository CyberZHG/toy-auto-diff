import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpTanh(NumGradCheck):

    def test_forward(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val)
        y = ad.tanh(x)
        actual = y.forward()
        expect = np.tanh(x_val)
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val)
        y = ad.tanh(x)
        self.numeric_gradient_check(y, {}, [x])
