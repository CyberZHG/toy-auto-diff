from unittest import TestCase
from demos.basic.linear_regression.linear import main as linear_main


class TestLinearRegression(TestCase):

    def test_linear(self):
        linear_main(verbose=False)
        linear_main(base_config={'input_len': 1}, verbose=False)
