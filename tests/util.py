from typing import Mapping, Union, List
import numpy as np
from unittest import TestCase
from auto_diff import Operation, OpPlaceholder, OpVariable


class NumGradCheck(TestCase):

    def numeric_gradient_check(self,
                               func: Operation,
                               feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray],
                               variables: List[OpVariable]):
        eps = 1e-8
        for variable in variables:
            values = variable.forward(feed_dict)
            gradient = variable.gradient.forward(feed_dict)
            numeric_gradient = np.zeros_like(values)
            it = np.nditer(values, flags=['multi_index'])
            while not it.finished:
                i = it.multi_index
                origin = values[i]
                values[i] = origin - eps
                variable.update(values)
                yl = func.forward(feed_dict)
                values[i] = origin + eps
                variable.update(values)
                yu = func.forward(feed_dict)
                values[i] = origin
                variable.update(values)
                numeric_gradient[i] = np.sum(yu - yl) / (eps * 2)
                it.iternext()
            self.assertTrue(np.allclose(numeric_gradient, gradient), (variable, numeric_gradient, gradient))
