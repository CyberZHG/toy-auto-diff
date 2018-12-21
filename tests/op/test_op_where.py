import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpWhere(NumGradCheck):

    @staticmethod
    def _gen_random_and_result(cond_shape, x_shape, y_shape):
        cond_val = np.random.randint(0, 2, cond_shape) == np.random.randint(0, 2, cond_shape)
        x_val = np.random.random(x_shape)
        y_val = np.random.random(y_shape)
        cond = ad.variable(cond_val.astype(np.float64), name='C%s' % str(cond_shape))
        x = ad.variable(x_val, name='X%s' % str(x_shape))
        y = ad.variable(y_val, name='Y%s' % str(y_shape))
        z = ad.where(cond, x, y)
        expect = np.where(cond_val, x_val, y_val)
        return z, [x, y], expect

    def test_forward(self):
        for _ in range(10):
            z, _, expect = self._gen_random_and_result((1,), (3, 4), (4,))
            actual = z.forward()
            self.assertEqual(expect.shape, z.shape)
            self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        for _ in range(10):
            z, variables, _ = self._gen_random_and_result((3, 1), (4,), (3, 4))
            self.numeric_gradient_check(z, {}, variables)
