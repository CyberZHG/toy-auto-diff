import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpVariable(NumGradCheck):

    def test_forward(self):
        sess = ad.Session()
        val = np.random.random((2, 3))
        w = ad.variable(val)
        actual = sess.run(w)
        expect = val
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        sess.prepare()
        val = np.random.random((2, 3))
        w.update(val)
        actual = sess.run(w)
        expect = val
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        w = ad.variable(np.random.random((2, 3)))
        self.numeric_gradient_check(w, {}, [w])

    def test_callable_initializer(self):
        w = ad.variable(initializer=lambda shape: np.ones(shape), shape=(2, 3))
        actual = w.forward()
        expect = np.ones((2, 3))
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_update_scalar(self):
        w = ad.variable(1.2)
        w.update(2.4)
        w.update_add(-3.6)
        self.assertAlmostEqual(-1.2, w.forward())
        with self.assertRaises(ValueError):
            w.update(np.array([1.0]))

    def test_update_tensor_shape_not_fit(self):
        w = ad.variable(np.zeros(shape=(2, 3)))
        with self.assertRaises(ValueError):
            w.update(np.array([1.0]))
