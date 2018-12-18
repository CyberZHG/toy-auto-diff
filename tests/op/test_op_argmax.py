import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpArgmax(NumGradCheck):

    def test_forward(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val)
        y = ad.argmax(x)
        actual = y.forward()
        expect = np.argmax(x_val)
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = ad.argmax(x, axis=0)
        actual = y.forward()
        expect = np.argmax(x_val, axis=0)
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = ad.argmax(x, axis=-1)
        actual = y.forward()
        expect = np.argmax(x_val, axis=-1)
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        with self.assertRaises(NotImplementedError):
            x_val = np.random.random((3, 4))
            x = ad.variable(x_val)
            y = ad.argmax(x)
            self.numeric_gradient_check(y, {}, [x])

    def test_name(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val, name='X')
        y = ad.argmax(x)
        self.assertEqual('argmax(X)', y.__unicode__())
        y = ad.argmax(x, axis=-1)
        self.assertEqual('argmax(X, axis=-1)', y.__unicode__())
