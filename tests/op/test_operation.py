import numpy as np
from auto_diff import Session, Operation, OpConstant
from unittest import TestCase


class DummaryOperation(Operation):

    def __init__(self, shape=None, **kwargs):
        if shape is not None:
            self.shape = shape
        super(DummaryOperation, self).__init__(**kwargs)

    def _get_name(self):
        return 'test'

    def _get_op_name(self):
        return 'test'


class TestOperation(TestCase):

    def test_no_shape(self):
        with self.assertRaises(NotImplementedError):
            Operation()

    def test_name_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Operation(shape=(1, 2))
        with self.assertRaises(NotImplementedError):
            Operation(shape=(1, 2), name='test')

    def test_forward_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            DummaryOperation(shape=(1, 2)).forward()

    def test_backward_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            DummaryOperation(shape=(1, 2)).backward()

    def test_twice(self):
        sess = Session()
        op = OpConstant(np.array(1.0))
        sess.run(op)
        sess.run(op)

    def test_equal(self):
        self.assertNotEqual(DummaryOperation(shape=(1, 2)), DummaryOperation(shape=(1, 2)))
