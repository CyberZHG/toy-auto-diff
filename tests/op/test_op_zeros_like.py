import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpZerosLike(NumGradCheck):

    def test_forward(self):
        w = ad.array([[1, 2, 3], [4, 5, 6]])
        zeros = ad.zeros_like(w).transpose()
        actual = zeros.forward()
        expect = np.zeros((3, 2))
        self.assertEqual((3, 2), zeros.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        w = ad.array([[1, 2, 3], [4, 5, 6]])
        zeros = ad.zeros_like(w).transpose()
        self.numeric_gradient_check(zeros, {}, [])

    def test_name(self):
        w = ad.array([[1, 2, 3], [4, 5, 6]], name='W')
        zeros = ad.zeros_like(w).transpose()
        self.assertEqual('transpose(zeros_like(W))', zeros.__unicode__())
