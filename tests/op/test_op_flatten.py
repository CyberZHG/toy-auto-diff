import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpFlatten(NumGradCheck):

    def test_forward(self):
        val = np.arange(6).reshape((1, 2, 3))
        wf = ad.variable(val).flatten()
        actual = wf.forward()
        expect = np.array([0, 1, 2, 3, 4, 5])
        self.assertEqual((6,), wf.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        val = np.arange(6).reshape((1, 2, 3))
        w = ad.variable(val)
        wf = ad.flatten(w)
        self.numeric_gradient_check(wf, {}, [w])
