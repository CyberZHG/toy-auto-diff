import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpWhereLoop(NumGradCheck):

    def test_forward(self):
        y = ad.while_loop(
            cond=lambda inputs: ad.less(inputs[0], ad.constant(10)),
            body=lambda inputs: [inputs[0] + 1],
            loop_vars=[ad.variable(0.0)],
        )
        actual = y.forward()
        expect = np.arange(1, 11)
        self.assertEqual((None,), y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

        y = ad.while_loop(
            cond=lambda inputs: ad.less(inputs[0], ad.constant(5)),
            body=lambda inputs: [inputs[0] + 1, (inputs[0] + 1) * inputs[1]],
            loop_vars=[ad.variable(0.0), ad.variable(1.0)],
            output_index=1,
        )
        actual = y.forward()
        expect = np.array([1, 2, 6, 24, 120])
        self.assertEqual((None,), y.shape)
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

        y = ad.while_loop(
            cond=lambda inputs: ad.less(inputs[0], ad.constant(64)),
            body=lambda inputs: [inputs[0] * 2, ad.dot(inputs[1], ad.variable([[1, 1], [1, 0]]))],
            loop_vars=[ad.variable(1), ad.variable([[1, 0], [0, 1]])],
            output_index=1,
        )
        actual = y.forward()
        expect = np.array([1, 2, 3, 5, 8, 13])
        self.assertEqual((None, 2, 2), y.shape)
        self.assertTrue(np.allclose(expect, actual[:, 0, 0]), (expect, actual))

    def test_backward(self):
        x = ad.variable([[1, 1], [1, 0]])
        y = ad.while_loop(
            cond=lambda inputs: ad.less(inputs[0], ad.constant(64)),
            body=lambda inputs: [inputs[0] * 2, ad.dot(inputs[1], x)],
            loop_vars=[ad.variable(1), ad.variable([[1, 0], [0, 1]])],
            output_index=1,
        )
        self.numeric_gradient_check(y, {}, [x])
