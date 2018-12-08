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
        func.backward()
        for variable in variables:
            values = variable.forward(feed_dict)
            shape = values.shape
            flattened = values.flatten()
            gradient = variable.gradient.forward(feed_dict)
            numeric_gradient = np.zeros_like(flattened, dtype=np.float64)
            for i in range(flattened.shape[0]):
                origin = float(flattened[i])
                flattened[i] = origin - eps
                variable.update(flattened.reshape(shape))
                yl = func.forward(feed_dict)
                flattened[i] = origin + eps
                variable.update(flattened.reshape(shape))
                yu = func.forward(feed_dict)
                flattened[i] = origin
                variable.update(flattened.reshape(shape))
                numeric_gradient[i] = np.sum(yu - yl) / (eps * 2)
            numeric_gradient = numeric_gradient.reshape(shape)
            self.assertTrue(np.allclose(numeric_gradient, gradient), (
                str(variable),
                str(variable.gradient),
                numeric_gradient,
                gradient,
            ))
