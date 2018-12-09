import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpZeros(NumGradCheck):

    def test_forward(self):
        zeros = ad.zeros(5)
        actual = zeros.forward()
        self.zeros = np.zeros(5)
        expect = self.zeros
        self.assertEqual((5,), zeros.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        zeros = ad.zeros((3, 5)).transpose()
        actual = zeros.forward()
        expect = np.zeros((5, 3))
        self.assertEqual((5, 3), zeros.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        zeros = ad.zeros((3, 5)).transpose()
        self.numeric_gradient_check(zeros, {}, [])

    def test_name(self):
        zeros = ad.zeros(5)
        self.assertEqual('zeros(5)', zeros.__unicode__())
        zeros = ad.zeros((3, 5)).transpose()
        self.assertEqual('transpose(zeros(3, 5))', zeros.__unicode__())
