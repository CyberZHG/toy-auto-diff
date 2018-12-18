import numpy as np
from ..op.util import NumGradCheck
import auto_diff as ad


class TestLeakyReLU(NumGradCheck):

    def test_leaky_relu(self):
        alpha = 1e-2
        x = ad.variable([1.0, -1.2, 0.0], name='X')
        y = ad.acts.leaky_relu(x, alpha=alpha)
        actual = y.forward()
        expect = np.array([1.0, -0.012, 0.0])
        self.assertTrue(np.allclose(actual, expect), (actual, expect))
        self.numeric_gradient_check(y, {}, [x])
        for _ in range(100):
            alpha = np.random.random()
            x = ad.variable(np.random.random((np.random.randint(1, 11), np.random.randint(1, 11))) - 0.5, name='X')
            y = ad.acts.leaky_relu(x, alpha=alpha)
            self.numeric_gradient_check(y, {}, [x])
