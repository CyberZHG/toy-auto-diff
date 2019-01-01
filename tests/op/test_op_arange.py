import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpArange(NumGradCheck):

    def test_forward(self):
        x = ad.arange(3)
        actual = x.forward()
        expect = np.array([0, 1, 2])
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

        x = ad.arange(3, 7)
        actual = x.forward()
        expect = np.array([3, 4, 5, 6])
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

        x = ad.arange(3, 7, 2)
        actual = x.forward()
        expect = np.array([3, 5])
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        x_val = np.random.random((3,))
        x = ad.variable(x_val)
        y = ad.arange(3)
        z = x * y
        self.numeric_gradient_check(z, {}, [x])
