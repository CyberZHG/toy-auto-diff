import numpy as np
from auto_diff import OpVariable
from .util import NumGradCheck


class TestOpSqueeze(NumGradCheck):

    def test_forward(self):
        val = np.ones((2, 1, 1, 3, 1))
        we = OpVariable(val).squeeze()
        actual = we.forward()
        expect = np.ones((2, 1, 1, 3))
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        we = we.expand_dims(axis=1)
        actual = we.forward()
        expect = np.ones((2, 1, 3))
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        we = we.expand_dims(axis=-2)
        actual = we.forward()
        expect = np.ones((2, 3))
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_forward_multi(self):
        val = np.ones((2, 1, 1, 3, 1))
        we = OpVariable(val).squeeze(axis=(-1, -3, -4))
        actual = we.forward()
        expect = np.ones((2, 3))
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        we = OpVariable(val).squeeze(axis=(-4, -1, -3))
        actual = we.forward()
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        we = OpVariable(val).squeeze(axis=(1, -1, 2))
        actual = we.forward()
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        we = OpVariable(val).squeeze(axis=(1, 2, 4))
        actual = we.forward()
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        val = np.ones((2, 1, 1, 3, 1))
        w = OpVariable(val)
        we = w.squeeze().squeeze(axis=1).squeeze(axis=-2)
        self.numeric_gradient_check(we, {}, [w])

    def test_backward_multi(self):
        val = np.ones((2, 1, 1, 3, 1))
        w = OpVariable(val)
        we = w.squeeze(axis=(-1, -3, -4))
        self.numeric_gradient_check(we, {}, [w])
        we = w.squeeze(axis=(-4, -1, -3))
        self.numeric_gradient_check(we, {}, [w])
        we = w.squeeze(axis=(1, -1, 2))
        self.numeric_gradient_check(we, {}, [w])
        we = w.squeeze(axis=(1, 2, 4))
        self.numeric_gradient_check(we, {}, [w])

    def test_name(self):
        val = np.ones((2, 1, 1, 3, 1))
        w = OpVariable(val)
        we = w.squeeze()
        self.assertEqual('squeeze(W(2, 1, 1, 3, 1))', we.__unicode__())
        we = w.squeeze(axis=1)
        self.assertEqual('squeeze(W(2, 1, 1, 3, 1), axis=1)', we.__unicode__())
        we = w.squeeze(axis=(1, -1, 2))
        self.assertEqual('squeeze(W(2, 1, 1, 3, 1), axis=(1, -1, 2))', we.__unicode__())
