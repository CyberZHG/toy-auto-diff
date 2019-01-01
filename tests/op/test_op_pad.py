import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpPad(NumGradCheck):

    def test_forward(self):
        val = np.random.random((3, 5))
        x = ad.variable(val)
        y = ad.pad(x, 2)
        actual = y.forward()
        expect = np.pad(val, 2, mode='constant')
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

        val = np.random.random((3, 5))
        x = ad.variable(val)
        y = ad.pad(x, (1, 2))
        actual = y.forward()
        expect = np.pad(val, (1, 2), mode='constant')
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

        val = np.random.random((3, 5))
        x = ad.variable(val)
        y = ad.pad(x, ((1,), (2,)))
        actual = y.forward()
        expect = np.pad(val, ((1,), (2,)), mode='constant')
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

        val = np.random.random((3, 5))
        x = ad.variable(val)
        y = ad.pad(x, ((1, 2), (3, 4)))
        actual = y.forward()
        expect = np.pad(val, ((1, 2), (3, 4)), mode='constant')
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

        val = np.random.random((3, 5))
        x = ad.placeholder(shape=(None, 5))
        y = ad.pad(x, 1)
        actual = y.forward({x: val})
        expect = np.pad(val, 1, mode='constant')
        self.assertEqual((None, 7), y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        val = np.random.random((3, 5))
        x = ad.variable(val)
        y = ad.pad(x, 2)
        self.numeric_gradient_check(y, {}, [x])

        y = ad.pad(x, (1, 2))
        self.numeric_gradient_check(y, {}, [x])

        y = ad.pad(x, ((1,), (2,)))
        self.numeric_gradient_check(y, {}, [x])

        y = ad.pad(x, ((1, 2), (3, 4)))
        self.numeric_gradient_check(y * y, {}, [x])
