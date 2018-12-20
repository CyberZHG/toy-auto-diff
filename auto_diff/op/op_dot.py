from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpDot(Operation):
    """Dot product of two tensors.

    **If** either `x` or `y` is a scalar, then it is equivalent to :class:`OpMultiply`.

    **If** both `x` and `y` are 1-D arrays, it is the inner product of vectors, the result is a scalar:

    .. math::
       z = \\sum_k x_{k} \\cdot y_{k}

    Partial derivatives of a single element:

    .. math::
       \\frac{\\partial L}{\\partial x_{i}} =
       \\frac{\\partial L}{\\partial z} \\cdot \\frac{\\partial z}{\\partial x_i} =
       \\frac{\\partial L}{\\partial z} \\cdot y_i

    .. math::
       \\frac{\\partial L}{\\partial y_{i}} =
       \\frac{\\partial L}{\\partial z} \\cdot \\frac{\\partial z}{\\partial y_i} =
       \\frac{\\partial L}{\\partial z} \\cdot x_i

    Vector derivatives:

    .. math::
       \\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial z} \\cdot y

    .. math::
       \\frac{\\partial L}{\\partial y} = \\frac{\\partial L}{\\partial z} \\cdot x

    Note that since `z` is a scalar, the calculation of vector derivatives in this case is the dot operation.

    **If** both `x` and `y` are 2-D arrays, it is the matrix multiplication, the result is a 2-D array:

    .. math::
       z_{ij} = \\sum_{k} x_{ik} \\cdot y_{kj}

    Partial derivative of a single element:

    .. math::
       \\begin{array}{rcl}
       \\displaystyle \\frac{\\partial L}{\\partial x_{ij}}
       &=& \\displaystyle \\sum_{a,b} \\frac{\\partial L}{\\partial z_{ab}} \\cdot
       \\frac{\\partial z_{ab}}{\\partial x_{ij}} \\\\
       &=& \\displaystyle \\sum_{a,b} \\frac{\\partial L}{\\partial z_{ab}} \\cdot
       \\frac{\\partial \\left ( \\sum_k x_{ak} \\cdot y_{kb} \\right )}{\\partial x_{ij}} \\\\
       &=& \\displaystyle \\sum_{b} \\frac{\\partial L}{\\partial z_{ib}} \\cdot
       \\frac{\\partial \\left ( x_{ij} \\cdot y_{jb} \\right )}{\\partial x_{ij}} \\\\
       &=& \\displaystyle \\sum_{k} \\frac{\\partial L}{\\partial z_{ik}} \\cdot
       y_{jk} \\\\
       &=& \\displaystyle \\sum_{k} \\left ( \\frac{\\partial L}{\\partial Z} \\right )_{ik} \\cdot (Y^T)_{kj} \\\\
       \\end{array}

    .. math::
       \\frac{\\partial L}{\\partial y_{ij}} = \\sum_{k} (X^T)_{ik} \\cdot
       \\left ( \\frac{\\partial L}{\\partial Z} \\right )_{kj}

    The results of partial derivatives are the same as the definition of the dot operation, therefore the matrix
    derivatives are:

    .. math::
       \\frac{\\partial L}{\\partial X} = \\frac{\\partial L}{\\partial Z} \\cdot Y^T

    .. math::
       \\frac{\\partial L}{\\partial Y} = X^T \\cdot \\frac{\\partial L}{\\partial Z}

    **If** `x` is an N-D tensor and `y` is a 1-D array, it is a sum product over the last axis of `x` and `y`.

    **If** `x` is an N-D tensor and `y` is an M-D tensor (M >= 2), it is a sum product over the last axis of `x` and
    second-to-last axis of `y`.

    """

    def __init__(self, x: Operation, y: Operation, **kwargs):
        self.inputs = [x, y]
        if x.isscalar():
            self.shape = y.shape
        elif y.isscalar():
            self.shape = x.shape
        elif x.dim == 1 and y.dim == 1:
            if x.shape[0] is not None and y.shape[0] is not None and x.shape[0] != y.shape[0]:
                raise ValueError('The dimensions of inputs should be equal, found %s and %s'
                                 % (str(x.shape), str(y.shape)))
            self.shape = ()
        elif x.dim == 2 and y.dim == 2:
            if x.shape[1] is not None and y.shape[0] is not None and x.shape[1] != y.shape[0]:
                raise ValueError('The last dimension of the first input and the first dimension of the second input '
                                 'should be equal, found %s and %s' % (str(x.shape), str(y.shape)))
            self.shape = (x.shape[0], y.shape[1])
        elif y.dim == 1:
            if x.shape[-1] is not None and y.shape[0] is not None and x.shape[-1] != y.shape[0]:
                raise ValueError('The last dimension of the first input and dimension of the second input '
                                 'should be equal, found %s and %s' % (str(x.shape), str(y.shape)))
            self.shape = x.shape[:-1]
        else:
            if x.shape[-1] is not None and y.shape[-2] is not None and x.shape[-1] != y.shape[-2]:
                raise ValueError('The last dimension of the first input and second-to-last dimension of the second '
                                 'input should be equal, found %s and %s' % (str(x.shape), str(y.shape)))
            self.shape = x.shape[:-1] + y.shape[:-2] + (y.shape[-1],)
        super(OpDot, self).__init__(**kwargs)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.dot(self.inputs[0].forward(feed_dict), self.inputs[1].forward(feed_dict))

    def _backward(self, gradient: Operation) -> None:
        x, y = self.inputs
        if x.isscalar():
            self.gradients = [
                (gradient * y).sum(),
                gradient * x,
            ]
        elif y.isscalar():
            self.gradients = [
                gradient * y,
                (gradient * x).sum(),
            ]
        elif x.dim == 1 and y.dim == 1:
            self.gradients = [
                gradient * y,
                gradient * x,
            ]
        elif x.dim == 2 and y.dim == 2:
            self.gradients = [
                gradient.dot(y.transpose()),
                x.transpose().dot(gradient),
            ]
        elif y.dim == 1:
            self.gradients = [
                gradient.expand_dims(axis=-1).dot(y.expand_dims(axis=0)),
                (x * gradient.expand_dims(axis=-1)).sum(axis=tuple(range(x.dim - 1))),
            ]
        else:
            x_pre_shape, y_pre_shape = x.shape[:-1], y.shape[:-2]
            x_pre_dims = np.prod(x_pre_shape, dtype=np.int)
            x_reshaped = x.reshape((-1, x.shape[-1]))
            y_reshaped = y.reshape((-1, y.shape[-2], y.shape[-1])).transpose((1, 0, 2)).reshape((y.shape[-2], -1))
            g_reshaped = gradient.reshape((x_pre_dims, -1))
            self.gradients = [
                OpDot(
                    g_reshaped,
                    y_reshaped.transpose()
                ).reshape(x.shape),
                OpDot(
                    x_reshaped.transpose(),
                    g_reshaped,
                ).reshape((y.shape[-2], -1, y.shape[-1])).transpose((1, 0, 2)).reshape(y.shape),
            ]
