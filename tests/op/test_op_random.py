import numpy as np
import auto_diff as ad
from .util import NumGradCheck


class TestOpRandom(NumGradCheck):

    def test_forward(self):
        random = ad.random(5)
        random.forward()
        self.assertEqual(5, random.shape)
        random = ad.random((3, 5)).transpose()
        random.forward()
        self.assertEqual((5, 3), random.shape)

    def test_backward(self):
        random = ad.random((3, 5)).transpose()
        self.numeric_gradient_check(random, {}, [])

    def test_forward_variable_shape(self):
        x = ad.placeholder(shape=(None, 3))
        y = ad.random(ad.shape(x))
        z = y.forward({x: np.random.random((5, 3))})
        self.assertEqual((5, 3), z.shape)
