import numpy as np
from unittest import TestCase
import auto_diff as ad


class TestOpConstant(TestCase):

    def test_forward(self):
        x = np.array([[1, 2]])
        actual = ad.array(x).forward()
        expect = x
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        ad.array(np.array(0.0)).backward()

    def test_name(self):
        self.assertEqual('0.0', str(ad.array(0)))
        self.assertEqual('constant(shape=(1,))', ad.array([1.0]).__str__())
        self.assertEqual('constant(shape=(1, 1))', ad.array([[1.0]]).__unicode__())
