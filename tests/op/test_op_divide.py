import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpDivide(NumGradCheck):

    @staticmethod
    def _gen_random_and_result(x_shape, y_shape, call_type=True):
        x_val = np.random.random(x_shape)
        y_val = np.random.random(y_shape)
        x = ad.variable(x_val, name='X%s' % str(x_shape))
        y = ad.variable(y_val, name='Y%s' % str(y_shape))
        if call_type:
            z = x // y
        else:
            z = ad.divide(x, y)
        expect = x_val / y_val
        return z, [x, y], expect

    def test_forward(self):
        z, _, expect = self._gen_random_and_result((3, 1), (3, 4), True)
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        z, variables, _ = self._gen_random_and_result((3, 4), (3, 1), False)
        self.numeric_gradient_check(z, {}, variables, atol=1e-5)
        x = ad.variable(np.random.random((2, 3)), name='X')
        z = 1.0 // x
        self.numeric_gradient_check(z, {}, [x], atol=1e-5)
        z = x // 1.0
        self.numeric_gradient_check(z, {}, [x], atol=1e-5)
