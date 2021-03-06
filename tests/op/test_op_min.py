import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpMin(NumGradCheck):

    def test_forward(self):
        val = np.random.random((3, 5))
        w = ad.array(val)
        y = w.transpose().min()
        actual = y.forward()
        expect = np.min(val)
        self.assertEqual((), y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = w.transpose().min(axis=-1)
        actual = y.forward()
        expect = np.min(val, axis=0)
        self.assertEqual((5,), y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = w.transpose().min(axis=0)
        actual = y.forward()
        expect = np.min(val, axis=-1)
        self.assertEqual((3,), y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = w.transpose().min(axis=(0, -1))
        actual = y.forward()
        expect = np.min(val)
        self.assertEqual((), y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_forward_keepdims(self):
        val = np.random.random((3, 5))
        w = ad.array(val)
        y = ad.min(w.transpose(), keepdims=True)
        actual = y.forward()
        expect = np.min(val, keepdims=True)
        self.assertEqual((1, 1), y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = w.transpose().min(axis=-1, keepdims=True)
        actual = y.forward()
        expect = np.transpose(np.min(val, axis=0, keepdims=True))
        self.assertEqual((5, 1), y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = w.transpose().min(axis=0, keepdims=True)
        actual = y.forward()
        expect = np.transpose(np.min(val, axis=-1, keepdims=True))
        self.assertEqual((1, 3), y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = w.transpose().min(axis=(0, -1), keepdims=True)
        actual = y.forward()
        expect = np.min(val, keepdims=True)
        self.assertEqual((1, 1), y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        val = np.random.random((3, 5))
        w = ad.variable(val, name='W')
        y = w.transpose().min()
        self.numeric_gradient_check(y, {}, [w])
        y = w.transpose().min(axis=-1)
        self.numeric_gradient_check(y, {}, [w])
        y = w.transpose().min(axis=0)
        self.numeric_gradient_check(y, {}, [w])
        y = w.transpose().min(axis=(0, -1))
        self.numeric_gradient_check(y, {}, [w])
        val = np.random.random((3, 4, 5))
        w = ad.variable(val, name='W')
        y = w.transpose().min()
        self.numeric_gradient_check(y, {}, [w])
        y = w.transpose().min(axis=(0, 2)).min(axis=0)
        self.numeric_gradient_check(y, {}, [w])

    def test_backward_keepdims(self):
        val = np.random.random((3, 5))
        w = ad.variable(val, name='W')
        y = ad.min(w.transpose(), keepdims=True)
        self.numeric_gradient_check(y, {}, [w])
        y = w.transpose().min(axis=-1, keepdims=True)
        self.numeric_gradient_check(y, {}, [w])
        y = w.transpose().min(axis=0, keepdims=True)
        self.numeric_gradient_check(y, {}, [w])
        y = w.transpose().min(axis=(0, -1), keepdims=True)
        self.numeric_gradient_check(y, {}, [w])
        val = np.random.random((3, 4, 5))
        w = ad.variable(val, name='W')
        y = w.transpose().min(keepdims=True)
        self.numeric_gradient_check(y, {}, [w])
        y = w.transpose().min(axis=(0, 2), keepdims=True).min(axis=1, keepdims=True)
        self.numeric_gradient_check(y, {}, [w])
