from typing import Mapping, Union, List
import numpy as np
from unittest import TestCase
from auto_diff import Operation, OpPlaceholder, OpVariable


class NumGradCheck(TestCase):

    def numeric_gradient_check(self,
                               func: Operation,
                               feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray],
                               variables: List[OpVariable],
                               atol=1e-6):
        eps = 1e-8
        for variable in variables:
            variable.clear_gradient()
        func.forward(feed_dict)
        func.backward()
        for variable in variables:
            values = variable.forward(feed_dict)
            gradient = variable.gradient
            if np.isscalar(values):
                origin = values
                variable.update(origin - eps)
                yl = func.forward(feed_dict)
                variable.update(origin + eps)
                yu = func.forward(feed_dict)
                variable.update(origin)
                numeric_gradient = np.sum(yu - yl) / (eps * 2)
            else:
                shape = values.shape
                flattened = values.flatten()
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
            self.assertTrue(np.alltrue(numeric_gradient - gradient < atol), '\n'.join(list(map(str, [
                '',
                '\tInput:\t\t' + str(variable),
                '\tShape:\t\t' + str(np.shape(gradient)),
                '\tNumerical:',
                numeric_gradient,
                '\tActual:',
                gradient,
                '\tDiff:',
                numeric_gradient - gradient,
            ]))))
