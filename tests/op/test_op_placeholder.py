import numpy as np
from unittest import TestCase
import auto_diff as ad


class TestOpPlaceholder(TestCase):

    def test_forward(self):
        x = ad.placeholder(shape=(2, 3))
        feed_dict = {x: np.random.random((2, 3))}
        actual = ad.Session().run(x, feed_dict=feed_dict)
        expect = feed_dict[x]
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
        feed_dict = {x: np.random.random((2, 3))}
        actual = ad.Session().run(x, feed_dict=feed_dict)
        expect = feed_dict[x]
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_backward(self):
        ad.placeholder(shape=(2, 3)).backward()
