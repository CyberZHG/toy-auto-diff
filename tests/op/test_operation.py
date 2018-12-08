import numpy as np
from auto_diff import Session, Operation, OpConstant
from unittest import TestCase


class DummaryOperation(Operation):

    def _get_name(self):
        return 'test'

    def _get_op_name(self):
        return 'test'


class TestOperation(TestCase):

    def test_name_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Operation()
        with self.assertRaises(NotImplementedError):
            Operation(name='test')

    def test_forward_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            DummaryOperation().forward()

    def test_backward_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            DummaryOperation().backward()

    def test_twice(self):
        sess = Session()
        op = OpConstant(np.array(1.0))
        sess.run(op)
        sess.run(op)

    def test_equal(self):
        self.assertNotEqual(DummaryOperation(), DummaryOperation())
