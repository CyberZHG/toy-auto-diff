import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpMapFn(NumGradCheck):

    def test_forward(self):
        x = ad.variable([1, 2, 3, 4, 5, 6])
        y = ad.map_fn(lambda x: x * x, x)
        actual = y.forward()
        expect = np.array([1, 4, 9, 16, 25, 36])
        self.assertEqual(expect.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

        x = ad.variable([1, 2, 3])
        y = ad.variable([-1, 1, -1])
        z = ad.map_fn(lambda x: x[0] * x[1], (x, y))
        actual = z.forward()
        expect = np.array([-1, 2, -3])
        self.assertEqual(expect.shape, z.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

        x = ad.variable([1, 2, 3])
        y = ad.map_fn(lambda x: (x, -x), x)
        actual = (y[0].forward(), y[1].forward())
        expect = (np.array([1, 2, 3]), np.array([-1, -2, -3]))
        for i in range(2):
            self.assertEqual(expect[i].shape, y[i].shape)
            self.assertTrue(np.allclose(expect[i], actual[i]), (i, expect[i], actual[i]))

    def test_backward(self):
        x = ad.variable([1, 2, 3, 4, 5, 6])
        y = ad.map_fn(lambda x: x * x, x)
        self.numeric_gradient_check(y, {}, [x])

        x = ad.variable([1, 2, 3])
        y = ad.variable([-1, 1, -1])
        z = ad.map_fn(lambda x: x[0] * x[1], (x, y))
        self.numeric_gradient_check(z, {}, [x, y])

        x = ad.variable([1, 2, 3])
        y = ad.map_fn(lambda x: (x, -x), x)
        z = y[0] * y[1]
        self.numeric_gradient_check(z, {}, [x])
