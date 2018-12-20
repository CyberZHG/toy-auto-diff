import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpNegative(NumGradCheck):

    def test_forward(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val)
        y = -x
        actual = y.forward()
        expect = -x_val
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val)
        y = -x
        self.numeric_gradient_check(y, {}, [x])
