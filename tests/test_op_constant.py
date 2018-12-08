import numpy as np
from unittest import TestCase
from auto_diff import OpConstant


class TestOpConstant(TestCase):

    def test_forward(self):
        x = np.array([[1, 2]])
        actual = OpConstant(x).forward()
        expect = x
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        OpConstant(np.array(0.0)).backward()

    def test_name(self):
        self.assertEqual('0.0', str(OpConstant(0)))
        self.assertEqual('C(1,)', OpConstant([1.0]).__str__())
        self.assertEqual('C(1, 1)', OpConstant([[1.0]]).__unicode__())
