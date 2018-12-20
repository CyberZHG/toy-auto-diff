import numpy as np
from unittest import TestCase
import auto_diff as ad
from auto_diff import OpPlaceholder


class TestSpConstSubTree(TestCase):

    def test_constants(self):
        x = ad.array(np.arange(12)).reshape((3, 4)).transpose().simplify()
        self.assertEqual('constant(shape=(4, 3))', x.name)

    def test_placeholder(self):
        x = OpPlaceholder(shape=(12,), name='X').reshape((3, 4)).transpose().simplify()
        self.assertEqual('transpose(reshape(X, shape=(3, 4)))', x.name)
