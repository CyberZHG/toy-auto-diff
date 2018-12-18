import numpy as np
from ..op.util import NumGradCheck
import auto_diff as ad


class TestSigmoid(NumGradCheck):

    def test_relu(self):
        x = ad.variable([1.0, -1.2, 0.0], name='X')
        y = ad.acts.sigmoid(x)
        actual = y.forward()
        expect = np.array([0.73105858, 0.23147522, 0.5])
        self.assertTrue(np.allclose(actual, expect), (actual, expect))
        self.numeric_gradient_check(y, {}, [x])
        for _ in range(100):
            x = ad.variable(np.random.random((np.random.randint(1, 11), np.random.randint(1, 11))) - 0.5, name='X')
            y = ad.acts.sigmoid(x)
            self.numeric_gradient_check(y, {}, [x])
