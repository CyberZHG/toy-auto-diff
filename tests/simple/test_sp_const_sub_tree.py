import numpy as np
from unittest import TestCase
from auto_diff import OpConstant, OpPlaceholder


class TestSpConstSubTree(TestCase):

    def test_constants(self):
        x = OpConstant(np.arange(12)).reshape((3, 4)).transpose().simplify()
        self.assertEqual('C(4, 3)', x.name)

    def test_placeholder(self):
        x = OpPlaceholder(shape=(12,), name='X').reshape((3, 4)).transpose().simplify()
        self.assertEqual('transpose(reshape(X, shape=(3, 4)))', x.name)
