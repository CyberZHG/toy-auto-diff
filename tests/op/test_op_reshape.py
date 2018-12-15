import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpReshape(NumGradCheck):

    def test_forward(self):
        val = np.arange(6)
        wr = ad.variable(val).reshape(shape=(1, 2, 3))
        actual = wr.forward()
        expect = np.array([[[0, 1, 2], [3, 4, 5]]])
        self.assertEqual((1, 2, 3), wr.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        wr = ad.variable(val).reshape(shape=(-1,))
        actual = wr.forward()
        expect = np.array([0, 1, 2, 3, 4, 5])
        self.assertEqual((6,), wr.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_forward_undefined(self):
        val = ad.variable(np.arange(6))
        with self.assertRaises(ValueError):
            val.reshape(shape=(-1, 2, -1))

    def test_backward(self):
        val = np.arange(6)
        w = ad.variable(val)
        wr = ad.reshape(w, shape=(1, 2, 3))
        self.numeric_gradient_check(wr, {}, [w])

    def test_name(self):
        val = np.arange(6)
        wr = ad.variable(val).reshape(shape=(1, 2, 3))
        self.assertEqual('reshape(W(6,), shape=(1, 2, 3))', wr.__unicode__())
