import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpEqual(NumGradCheck):

    @staticmethod
    def _gen_random_and_result(x_shape, y_shape):
        x_val = np.random.random(x_shape)
        y_val = np.random.random(y_shape)
        x = ad.variable(x_val, name='X%s' % str(x_shape))
        y = ad.variable(y_val, name='Y%s' % str(y_shape))
        z = ad.equal(x, y)
        expect = (x_val == y_val).astype(dtype=np.float64)
        return z, [x, y], expect

    def test_forward(self):
        z, _, expect = self._gen_random_and_result((3, 4), (3, 4))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result(4, (3, 4))
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        with self.assertRaises(NotImplementedError):
            z, variables, _ = self._gen_random_and_result((3, 4), (3, 4))
            self.numeric_gradient_check(z, {}, variables)

    def test_name(self):
        z, _, _ = self._gen_random_and_result((1, 3, 1, 4), (5, 1))
        self.assertEqual('equal(X(1, 3, 1, 4), Y(5, 1))', z.__unicode__())

    def test_broadcast_failed(self):
        with self.assertRaises(ValueError):
            self._gen_random_and_result((1, 3, 4), (1, 4, 1))
