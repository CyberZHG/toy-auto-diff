from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestRandom(TestCase):

    def test_random_normal(self):
        weights = ad.inits.random_normal(mean=0.1, stddev=0.05)(shape=(3, 5))
        self.assertEqual((3, 5), weights.shape)
        weights = ad.inits.random_normal(mean=0.1, stddev=0.05)(shape=(300, 500))
        weights = weights.flatten()
        mean = np.mean(weights)
        stddev = np.std(weights)
        self.assertTrue(abs(mean - 0.1) < 1e-3)
        self.assertTrue(abs(stddev - 0.05) < 1e-3)

    def test_random_uniform(self):
        weights = ad.inits.random_uniform(low=0.1, high=0.3)(shape=(3, 5))
        self.assertEqual((3, 5), weights.shape)
        weights = ad.inits.random_uniform(low=0.1, high=0.3)(shape=(300, 500))
        weights = weights.flatten()
        self.assertTrue(0.1 <= np.min(weights))
        self.assertTrue(np.max(weights) <= 0.3)
