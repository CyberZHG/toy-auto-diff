import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpMultiply(NumGradCheck):

    @staticmethod
    def _gen_random_and_result(x_shape, y_shape):
        x_val = np.random.random(x_shape)
        y_val = np.random.random(y_shape)
        x = ad.variable(x_val, name='X%s' % str(x_shape))
        y = ad.variable(y_val, name='Y%s' % str(y_shape))
        z = x * y
        expect = x_val * y_val
        return z, [x, y], expect

    def test_forward(self):
        z, _, expect = self._gen_random_and_result((3, 4), (3, 4))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result((3, 1), (3, 4))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result((3, 4), (3, 1))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result((4,), (3, 4))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result((3, 4), (4,))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result((1, 3, 1, 4), (5, 1))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        z, variables, _ = self._gen_random_and_result((3, 4), (3, 4))
        self.numeric_gradient_check(z, {}, variables)
        z, variables, _ = self._gen_random_and_result((3, 1), (3, 4))
        self.numeric_gradient_check(z, {}, variables)
        z, variables, _ = self._gen_random_and_result((3, 4), (3, 1))
        self.numeric_gradient_check(z, {}, variables)
        z, variables, _ = self._gen_random_and_result((4,), (3, 4))
        self.numeric_gradient_check(z, {}, variables)
        z, variables, _ = self._gen_random_and_result((3, 4), (4,))
        self.numeric_gradient_check(z, {}, variables)
        z, variables, _ = self._gen_random_and_result((1, 3, 1, 4), (5, 1))
        self.numeric_gradient_check(z, {}, variables)

    def test_name(self):
        z, _, _ = self._gen_random_and_result((1, 3, 1, 4), (5, 1))
        self.assertEqual('X(1, 3, 1, 4) * Y(5, 1)', z.__unicode__())
        x = ad.placeholder(shape=(3, 4), name='X')
        y = ad.placeholder(shape=(1, 1), name='Y')
        z = ad.multiply(x, y)
        self.assertEqual('X * Y', z.__unicode__())
