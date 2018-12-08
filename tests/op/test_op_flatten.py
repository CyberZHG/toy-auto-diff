import numpy as np
from auto_diff import OpVariable
from .util import NumGradCheck


class TestOpFlatten(NumGradCheck):

    def test_forward(self):
        val = np.arange(6).reshape((1, 2, 3))
        wf = OpVariable(val).flatten()
        actual = wf.forward()
        expect = np.array([0, 1, 2, 3, 4, 5])
        self.assertEqual((6,), wf.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        val = np.arange(6).reshape((1, 2, 3))
        w = OpVariable(val)
        wf = w.flatten()
        self.numeric_gradient_check(wf, {}, [w])

    def test_name(self):
        val = np.arange(6).reshape((1, 2, 3))
        wf = OpVariable(val).flatten()
        self.assertEqual('flatten(W(1, 2, 3))', wf.__unicode__())
