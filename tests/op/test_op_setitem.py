import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpSetItem(NumGradCheck):

    def test_forward(self):
        x_val = np.random.random((3, 4))
        x = ad.variable(x_val)
        y = ad.setitem(x, (1, 2), ad.constant(5.0))
        actual = y.forward()[1, 2]
        expect = 5.0
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        with self.assertRaises(NotImplementedError):
            x_val = np.random.random((3, 4))
            x = ad.variable(x_val)
            y = ad.setitem(x, (1, 2), ad.constant(5.0))
            self.numeric_gradient_check(y, {}, [x])
