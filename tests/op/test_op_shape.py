import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpShape(NumGradCheck):

    def test_forward(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val)
        y = ad.shape(x)
        actual = y.forward()
        expect = np.array([3, 4])
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        with self.assertRaises(NotImplementedError):
            x_val = np.random.random((3, 4))
            x = ad.variable(x_val)
            y = ad.shape(x)
            self.numeric_gradient_check(y, {}, [x])

    def test_name(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val, name='X')
        y = ad.shape(x)
        self.assertEqual('shape(X)', y.__unicode__())
