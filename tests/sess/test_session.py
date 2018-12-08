import numpy as np
from unittest import TestCase
from auto_diff import Session, OpConstant


class TestSession(TestCase):

    def setUp(self):
        self.session = Session()

    def test_run_unknown_type(self):
        with self.assertRaises(NotImplementedError):
            self.session.prepare()
            self.session.run(set([OpConstant]))

    def test_run_single(self):
        x = np.array([[1, 2]])
        actual = self.session.run(OpConstant(x))
        expect = x
        self.assertTrue(np.allclose(expect, actual), (expect, actual))

    def test_run_list(self):
        x, y = np.array([[1, 2]]), np.array([2, 1])
        actual = self.session.run([OpConstant(x), OpConstant(y)])
        expect = [x, y]
        self.assertTrue(np.allclose(expect[0], actual[0]), (expect[0], actual[0]))
        self.assertTrue(np.allclose(expect[1], actual[1]), (expect[1], actual[1]))

    def test_run_dict(self):
        x, y = np.array([[1, 2]]), np.array([2, 1])
        actual = self.session.run({'x': OpConstant(x), 'y': OpConstant(y)})
        expect = {'x': x, 'y': y}
        self.assertTrue(np.allclose(expect['x'], actual['x']), (expect['x'], actual['x']))
        self.assertTrue(np.allclose(expect['y'], actual['y']), (expect['y'], actual['y']))
