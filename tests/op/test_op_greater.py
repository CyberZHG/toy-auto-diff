import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpGreater(NumGradCheck):

    @staticmethod
    def _gen_random_and_result(x_shape, y_shape, call_type=True):
        x_val = np.random.random(x_shape)
        y_val = np.random.random(y_shape)
        x = ad.variable(x_val, name='X%s' % str(x_shape))
        y = ad.variable(y_val, name='Y%s' % str(y_shape))
        if call_type:
            z = x > y
        else:
            z = ad.greater(x, y)
        expect = (x_val > y_val).astype(dtype=np.float64)
        return z, [x, y], expect

    def test_forward(self):
        z, _, expect = self._gen_random_and_result((3, 4), (3, 4), True)
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        z, _, expect = self._gen_random_and_result(4, (3, 4), False)
        actual = z.forward()
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

        x_val = np.random.random()
        x = ad.variable(x_val)
        y = x < 0.5
        actual = y.forward()
        expect = x_val < 0.5
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        y = 0.5 < x
        actual = y.forward()
        expect = 0.5 < x_val
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        z, _, _ = self._gen_random_and_result((3, 4), (3, 4))
        z.forward()
        z.backward()

    def test_broadcast_failed(self):
        with self.assertRaises(ValueError):
            self._gen_random_and_result((1, 3, 4), (1, 4, 1))
