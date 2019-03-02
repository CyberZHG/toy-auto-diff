from unittest import TestCase
import numpy as np
import auto_diff as ad


class TestConstants(TestCase):

    def test_zeros(self):
        weights = ad.inits.zeros(shape=(3, 5))
        self.assertEqual((3, 5), weights.shape)
        self.assertEqual(0.0, np.max(weights))
        self.assertEqual(0.0, np.min(weights))

        weights = ad.variable(ad.inits.zeros, shape=3)
        self.assertEqual((3,), weights.shape)

    def test_ones(self):
        weights = ad.inits.ones(shape=(3, 5))
        self.assertEqual((3, 5), weights.shape)
        self.assertEqual(1.0, np.max(weights))
        self.assertEqual(1.0, np.min(weights))

    def test_constants(self):
        weights = ad.inits.constants(2.5)(shape=(3, 5))
        self.assertEqual((3, 5), weights.shape)
        self.assertEqual(2.5, np.max(weights))
        self.assertEqual(2.5, np.min(weights))
