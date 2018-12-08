import numpy as np
from auto_diff import OpVariable, OpTranspose
from .util import NumGradCheck


class TestOpVariable(NumGradCheck):

    def test_forward_default(self):
        val = np.array([[1, 2, 3], [4, 5, 6]])
        w = OpVariable(val)
        wt = OpTranspose(w)
        actual = wt.forward()
        expect = np.array([[1, 4], [2, 5], [3, 6]])
        self.assertEqual((3, 2), wt.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_forward_axes(self):
        val = np.arange(6).reshape((1, 2, 3))
        w = OpVariable(val)
        wt = OpTranspose(w, axes=(1, 0, 2))
        actual = wt.forward()
        expect = np.array([[[0, 1, 2]], [[3, 4, 5]]])
        self.assertEqual((2, 1, 3), wt.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward_default(self):
        val = np.array([[1, 2, 3], [4, 5, 6]])
        w = OpVariable(val)
        wt = OpTranspose(w)
        self.numeric_gradient_check(wt, {}, [w])

    def test_backward_axes(self):
        val = np.arange(6).reshape((1, 2, 3))
        w = OpVariable(val)
        wt = OpTranspose(w, axes=(1, 0, 2))
        self.numeric_gradient_check(wt, {}, [w])

    def test_name(self):
        val = np.array([[1, 2, 3], [4, 5, 6]])
        w = OpVariable(val)
        wt = OpTranspose(w)
        self.assertEqual('(W(2, 3))^T', wt.__unicode__())
        val = np.arange(6).reshape((1, 2, 3))
        w = OpVariable(val)
        wt = OpTranspose(w, axes=(1, 0, 2))
        self.assertEqual('transpose(W(1, 2, 3), axes=(1, 0, 2))', wt.__unicode__())