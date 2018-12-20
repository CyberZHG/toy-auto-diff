import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpOnesLike(NumGradCheck):

    def test_forward(self):
        w = ad.array([[1, 2, 3], [4, 5, 6]])
        ones = ad.ones_like(w).transpose()
        actual = ones.forward()
        expect = np.ones((3, 2))
        self.assertEqual((3, 2), ones.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        w = ad.array([[1, 2, 3], [4, 5, 6]])
        ones = ad.ones_like(w).transpose()
        self.numeric_gradient_check(ones, {}, [])
