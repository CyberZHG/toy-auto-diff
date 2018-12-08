import numpy as np
from auto_diff import Session, OpVariable
from .util import NumGradCheck


class TestOpVariable(NumGradCheck):

    def test_forward(self):
        sess = Session()
        val = np.random.random((2, 3))
        w = OpVariable(val)
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
        w = OpVariable(np.random.random((2, 3)))
        self.numeric_gradient_check(w, {}, [w])

    def test_name(self):
        self.assertEqual('W(3, 1, 2)', str(OpVariable(np.random.random((3, 1, 2)))))
        self.assertEqual('W()', OpVariable(2.0).__str__())
        self.assertEqual('W(1, 1)', OpVariable([[2.0]]).__unicode__())

    def test_callable_initializer(self):
        w = OpVariable(initializer=lambda shape: np.ones(shape), shape=(2, 3))
        actual = w.forward()
        expect = np.ones((2, 3))
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_update_scalar(self):
        w = OpVariable(1.2)
        w.update(2.4)
        with self.assertRaises(ValueError):
            w.update(np.array([1.0]))

    def test_update_tensor_shape_not_fit(self):
        w = OpVariable(np.zeros(shape=(2, 3)))
        with self.assertRaises(ValueError):
            w.update(np.array([1.0]))
