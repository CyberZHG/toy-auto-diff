import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpLog(NumGradCheck):

    def test_forward(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val)
        y = ad.log(x)
        actual = y.forward()
        expect = np.log(x_val)
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val)
        y = ad.log(x)
        self.numeric_gradient_check(y, {}, [x])

    def test_name(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val, name='X')
        y = ad.log(x)
        self.assertEqual('log(X)', y.__unicode__())
