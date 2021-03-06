import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpTranspose(NumGradCheck):

    def test_forward_default(self):
        val = np.array([[1, 2, 3], [4, 5, 6]])
        wt = ad.variable(val).transpose()
        actual = wt.forward()
        expect = np.array([[1, 4], [2, 5], [3, 6]])
        self.assertEqual((3, 2), wt.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_forward_axes(self):
        val = np.arange(6).reshape((1, 2, 3))
        wt = ad.transpose(ad.variable(val), axes=(1, 0, 2))
        actual = wt.forward()
        expect = np.array([[[0, 1, 2]], [[3, 4, 5]]])
        self.assertEqual((2, 1, 3), wt.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward_default(self):
        val = np.array([[1, 2, 3], [4, 5, 6]])
        w = ad.variable(val)
        wt = w.transpose()
        self.numeric_gradient_check(wt, {}, [w])

    def test_backward_axes(self):
        val = np.arange(6).reshape((1, 2, 3))
        w = ad.variable(val)
        wt = w.transpose(axes=(1, 0, 2))
        self.numeric_gradient_check(wt, {}, [w])
