import numpy as np
from auto_diff import OpVariable
from .util import NumGradCheck


class TestOpExpandDims(NumGradCheck):

    def test_forward(self):
        val = np.ones((2, 3))
        we = OpVariable(val).expand_dims()
        actual = we.forward()
        expect = np.ones((2, 3, 1))
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        we = we.expand_dims(axis=1)
        actual = we.forward()
        expect = np.ones((2, 1, 3, 1))
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        we = we.expand_dims(axis=-2)
        actual = we.forward()
        expect = np.ones((2, 1, 1, 3, 1))
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        val = np.ones((2, 3))
        w = OpVariable(val)
        we = w.expand_dims().expand_dims(axis=1).expand_dims(axis=-2)
        self.numeric_gradient_check(we, {}, [w])

    def test_name(self):
        val = np.ones((2, 3))
        we = OpVariable(val).expand_dims()
        self.assertEqual('expand_dims(W(2, 3))', we.__unicode__())
        we = we.expand_dims(axis=1)
        self.assertEqual('expand_dims(expand_dims(W(2, 3)), axis=1)', we.__unicode__())
