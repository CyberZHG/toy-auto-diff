import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpSubtract(NumGradCheck):

    @staticmethod
    def _gen_random_and_result(x_shape, y_shape, call_type=True):
        x_val = np.random.random(x_shape)
        y_val = np.random.random(y_shape)
        x = ad.variable(x_val, name='X%s' % str(x_shape))
        y = ad.variable(y_val, name='Y%s' % str(y_shape))
        if call_type:
            z = x - y
        else:
            z = ad.subtract(x, y)
        expect = x_val - y_val
        return z, [x, y], expect

    def test_forward(self):
        z, _, expect = self._gen_random_and_result((3, 4), (4,), True)
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        z, variables, _ = self._gen_random_and_result((4,), (3, 4), False)
        self.numeric_gradient_check(z, {}, variables)
        x = ad.variable(np.random.random((2, 3)), name='X')
        z = 1.0 - x
        self.numeric_gradient_check(z, {}, [x])
        z = x - 1.0
        self.numeric_gradient_check(z, {}, [x])
