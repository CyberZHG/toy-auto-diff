import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpOnes(NumGradCheck):

    def test_forward(self):
        ones = ad.ones(5)
        actual = ones.forward()
        expect = np.ones(5)
        self.assertEqual((5,), ones.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        ones = ad.ones((3, 5)).transpose()
        actual = ones.forward()
        expect = np.ones((5, 3))
        self.assertEqual((5, 3), ones.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        ones = ad.ones((3, 5)).transpose()
        self.numeric_gradient_check(ones, {}, [])

    def test_name(self):
        ones = ad.ones(5)
        self.assertEqual('ones(5)', ones.__unicode__())
        ones = ad.ones((3, 5)).transpose()
        self.assertEqual('transpose(ones(3, 5))', ones.__unicode__())
