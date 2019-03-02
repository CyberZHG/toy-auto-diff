import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpDot(NumGradCheck):

    @staticmethod
    def _gen_random_and_result(x_shape, y_shape, call_type=True):
        x_val = np.random.random(x_shape)
        y_val = np.random.random(y_shape)
        x = ad.variable(x_val, name='X%s' % str(x_shape))
        y = ad.variable(y_val, name='Y%s' % str(y_shape))
        if call_type:
            z = ad.dot(x, y)
        else:
            z = x.dot(y)
        expect = np.dot(x_val, y_val)
        return z, [x, y], expect

    def test_forward_scalar(self):
        z, _, expect = self._gen_random_and_result(None, None, True)
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result(None, (3, 4), False)
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result((3, 4), None)
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward_scalar(self):
        z, variables, _ = self._gen_random_and_result(None, None)
        self.numeric_gradient_check(z, {}, variables)
        z, variables, _ = self._gen_random_and_result((3, 4), None)
        self.numeric_gradient_check(z, {}, variables)

    def test_forward_1d(self):
        z, _, expect = self._gen_random_and_result((3,), (3,))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_forward_1d_invalid(self):
        with self.assertRaises(ValueError):
            self._gen_random_and_result((3,), (4,))

    def test_backward_1d(self):
        z, variables, _ = self._gen_random_and_result((3,), (3,))
        self.numeric_gradient_check(z, {}, variables)

    def test_forward_2d(self):
        z, _, expect = self._gen_random_and_result((3, 4), (4, 5))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_forward_2d_invalid(self):
        with self.assertRaises(ValueError):
            self._gen_random_and_result((3, 4), (5, 6))

    def test_backward_2d(self):
        z, variables, _ = self._gen_random_and_result((3, 4), (4, 5))
        self.numeric_gradient_check(z, {}, variables)

    def test_forward_nd_1d(self):
        z, _, expect = self._gen_random_and_result((3, 4), (4,))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_forward_nd_1d_invalid(self):
        with self.assertRaises(ValueError):
            self._gen_random_and_result((3, 4), (5,))

    def test_backward_nd_1d(self):
        z, variables, _ = self._gen_random_and_result((3, 4), (4,))
        self.numeric_gradient_check(z, {}, variables)

    def test_forward_nd_md(self):
        z, _, expect = self._gen_random_and_result((3, 4, 5), (6, 5, 7))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result((5,), (6, 5, 7))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result((5,), (5, 7))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result((3, 4), (5, 6, 4, 7))
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_forward_nd_md_invalid(self):
        with self.assertRaises(ValueError):
            self._gen_random_and_result((3, 4, 5), (6, 2, 7))

    def test_backward_nd_md(self):
        z, variables, _ = self._gen_random_and_result((3, 4, 5), (6, 5, 7))
        self.numeric_gradient_check(z, {}, variables)
        z, variables, _ = self._gen_random_and_result((5,), (6, 5, 7))
        self.numeric_gradient_check(z, {}, variables)
        z, variables, _ = self._gen_random_and_result((5,), (5, 7))
        self.numeric_gradient_check(z, {}, variables)
        z, variables, _ = self._gen_random_and_result((3, 4), (5, 6, 4, 7))
        self.numeric_gradient_check(z, {}, variables)
