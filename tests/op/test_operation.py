import numpy as np
import auto_diff as ad
from unittest import TestCase


class DummyOpWithNames(ad.Operation):

    def __init__(self, shape=None, **kwargs):
        if shape is not None:
            self.shape = shape
        super(DummyOpWithNames, self).__init__(**kwargs)

    def _get_name(self):
        return 'test'

    def _get_op_name(self):
        return 'test'


class DummyOpWithoutOpName(ad.Operation):

    def __init__(self, shape=None, **kwargs):
        if shape is not None:
            self.shape = shape
        super(DummyOpWithoutOpName, self).__init__(**kwargs)

    def _get_name(self):
        return 'test'


class TestOperation(TestCase):

    def test_no_shape(self):
        with self.assertRaises(NotImplementedError):
            ad.Operation()

    def test_name_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            ad.Operation(shape=(1, 2))
        with self.assertRaises(NotImplementedError):
            ad.Operation(shape=(1, 2), name='test')
        with self.assertRaises(NotImplementedError):
            DummyOpWithoutOpName(shape=(1, 2))

    def test_forward_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            DummyOpWithNames(shape=(1, 2)).forward()

    def test_backward_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            DummyOpWithNames(shape=(1, 2)).backward()

    def test_twice(self):
        sess = ad.Session()
        op = ad.constant(np.array(1.0))
        sess.run(op)
        sess.run(op)

    def test_equal(self):
        self.assertNotEqual(DummyOpWithNames(shape=(1, 2)), DummyOpWithNames(shape=(1, 2)))
