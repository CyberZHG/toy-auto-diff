import numpy as np
from ..op.util import NumGradCheck
import auto_diff as ad


class TestMeanSquareError(NumGradCheck):

    def test_mean_square_error(self):
        for _ in range(100):
            n, m = np.random.randint(1, 11), np.random.randint(1, 11)
            x_val = np.random.random((n, m))
            y_val = np.zeros((n, m))
            classes = np.random.randint(0, m, (n,))
            y_val[np.arange(n), classes] = 1.0
            x = ad.variable(x_val, name='X')
            y_pred = ad.acts.softmax(x)
            y_true = ad.variable(y_val, name='Y')
            loss = ad.losses.mean_square_error(y_true, y_pred)
            self.assertEqual((n,), loss.shape)
            self.numeric_gradient_check(loss, {}, [x])
