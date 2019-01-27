from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestGlorot(TestCase):

    def test_glorot_normal(self):
        weights = ad.inits.glorot_normal(shape=(3, 5))
        self.assertEqual((3, 5), weights.shape)
        weights = ad.inits.glorot_normal(shape=(300, 500))
        weights = weights.flatten()
        mean = np.mean(weights)
        stddev = np.std(weights)
        self.assertTrue(abs(mean) < 1e-3)
        self.assertTrue(abs(stddev - 0.05) < 1e-3)

    def test_glorot_uniform(self):
        weights = ad.inits.glorot_uniform(shape=(3, 5))
        self.assertEqual((3, 5), weights.shape)
        weights = ad.inits.glorot_uniform(shape=(300, 500))
        weights = weights.flatten()
        self.assertTrue(0.08 < np.max(np.abs(weights)) < 0.08661)
