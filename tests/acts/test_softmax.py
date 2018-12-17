import numpy as np
from ..op.util import NumGradCheck
import auto_diff as ad


class TestSoftmax(NumGradCheck):

    def test_softmax_vector(self):
        for _ in range(100):
            x = ad.variable(np.random.random((np.random.randint(1, 11))), name='X')
            y = ad.acts.softmax(x)
            s = y.sum(axis=-1).forward()
            self.assertTrue(np.allclose(np.ones_like(s), s), (s,))
            self.numeric_gradient_check(y, {}, [x])

    def _test_softmax_matrix(self):
        for _ in range(100):
            x = ad.variable(np.random.random((np.random.randint(1, 11), np.random.randint(1, 11))), name='X')
            y = ad.acts.softmax(x)
            s = y.sum(axis=-1).forward()
            self.assertTrue(np.allclose(np.ones_like(s), s), (s,))
            self.numeric_gradient_check(y, {}, [x])
