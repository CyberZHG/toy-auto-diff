import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpGetItem(NumGradCheck):

    def test_forward(self):
        val = np.random.random((1, 2, 3))
        x = ad.variable(val)
        y = x[0, 1, 0]
        actual = y.forward()
        expect = val[0, 1, 0]
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = x[0]
        actual = y.forward()
        expect = val[0]
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = x[0, 1:]
        actual = y.forward()
        expect = val[0, 1:]
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = x[0, 1:, :2]
        actual = y.forward()
        expect = val[0, 1:, :2]
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = x[0, :, :2:]
        actual = y.forward()
        expect = val[0, :, :2:]
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = x[0, :-2:-1, ::-2]
        actual = y.forward()
        expect = val[0, :-2:-1, ::-2]
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        val = np.random.random((1, 2, 3))
        x = ad.variable(val)
        y = x[0, 1, 0]
        self.numeric_gradient_check(y, {}, [x])
        y = x[0]
        self.numeric_gradient_check(y, {}, [x])
        y = x[0, 1:]
        self.numeric_gradient_check(y, {}, [x])
        y = x[0, 1:, :2]
        self.numeric_gradient_check(y, {}, [x])
        y = x[0, :, :2:]
        self.numeric_gradient_check(y, {}, [x])
        y = x[0, :-2:-1, ::-2]
        self.numeric_gradient_check(y, {}, [x])

    def test_name(self):
        val = np.random.random((1, 2, 3))
        x = ad.variable(val)
        y = x[0, 1:, :-1:-1]
        self.assertEqual('W(1, 2, 3)[0, 1:, :-1:-1]', y.__unicode__())
        y = x[0, :, -2:]
        self.assertEqual('W(1, 2, 3)[0, :, -2:]', y.__unicode__())
