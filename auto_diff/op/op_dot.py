from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpDot(Operation):
    """Dot product of two tensors.

    **If** either `x` or `y` is a scalar, then it is equivalent to :class:`OpMultiply`.

    **If** both `x` and `y` are 1-D arrays, it is the inner product of vectors, the result is a scalar:

    .. math::
       z = \sum_k x_{k} \cdot y_{k}

    Partial derivatives of a single element:

    .. math::
       \\frac{\\partial L}{\\partial x_{i}} =
       \\frac{\partial L}{\\partial z} \\cdot \\frac{\\partial z}{\\partial x_i} =
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
       z_{ij} = \sum_{k} x_{ik} \cdot y_{kj}

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

    **If** `x` is an N-D tensor and `y` is an M-D tensor (M >= 2), it is a sum product over the last axis of `x` and
    second-to-last axis of `y`.

    **If** `x` is an N-D tensor and `y` is a 1-D array, it is a sum product over the last axis of `x` and `y`.
    It is a special case of the previous condition if `y` is considered as a K x 1 matrix and result is squeezed.
    """

    def __init__(self, x: Operation, y: Operation, **kwargs):
        self.inputs = [x, y]
        if x.isscalar():
            self.shape = y.shape
        elif y.isscalar():
            self.shape = x.shape
        elif x.dim == 1 and y.dim == 1:
            if x.shape[0] != y.shape[0]:
                raise ValueError('The dimensions of inputs should be equal, found %s and %s'
                                 % (str(x.shape), str(y.shape)))
            self.shape = ()
        elif x.dim == 2 and y.dim == 2:
            if x.shape[1] != y.shape[0]:
                raise ValueError('The last dimension of the first input and the first dimension of the second input '
                                 'should be equal, found %s and %s' % (str(x.shape), str(y.shape)))
            self.shape = (x.shape[0], y.shape[1])
        elif y.dim == 1:
            if x.shape[-1] != y.shape[0]:
                raise ValueError('The last dimension of the first input and dimension of the second input '
                                 'should be equal, found %s and %s' % (str(x.shape), str(y.shape)))
            self.shape = x.shape[:-1]
        else:
            if x.shape[-1] != y.shape[-2]:
                raise ValueError('The last dimension of the first input and second-to-last dimension of the second '
                                 'input should be equal, found %s and %s' % (str(x.shape), str(y.shape)))
            self.shape = x.shape[:-1] + y.shape[:-2] + (y.shape[-1],)
        super(OpDot, self).__init__(**kwargs)

    def _get_name(self) -> str:
        return 'dot(%s, %s)' % (self.inputs[0].name, self.inputs[1].name)

    def _get_op_name(self) -> str:
        return 'dot(%s, %s)' % (self.inputs[0]._op_name, self.inputs[1]._op_name)

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        return np.dot(self.inputs[0].forward(feed_dict), self.inputs[1].forward(feed_dict))

    def _backward(self, gradient: Operation) -> None:
        x, y = self.inputs
        if x.isscalar():
            self.gradient = [
                (gradient * y).sum(),
                gradient * x,
            ]
        elif y.isscalar():
            self.gradient = [
                gradient * y,
                (gradient * x).sum(),
            ]
        elif x.dim == 1 and y.dim == 1:
            self.gradient = [
                gradient * y,
                gradient * x,
            ]
        elif x.dim == 2 and y.dim == 2:
            self.gradient = [
                gradient.dot(y.transpose()),
                x.transpose().dot(gradient),
            ]
        elif y.dim == 1:
            self.gradient = [
                gradient.expand_dims(axis=-1).dot(y.expand_dims(axis=0)),
                (x * gradient.expand_dims(axis=-1)).sum(axis=tuple(range(x.dim - 1))),
            ]
        else:
            zx_dim, zy_dim = x.dim - 1, y.dim - 2
            self.gradient = [
                gradient.sum(axis=tuple(range(zx_dim, zx_dim + zy_dim))).dot(
                    y.sum(axis=tuple(range(zy_dim))).transpose()
                ) * (1.0 / np.prod(y.shape[:zy_dim])),
                gradient.sum(axis=tuple(range(zx_dim))).expand_dims(axis=-1).dot(
                    x.sum(axis=tuple(range(zx_dim))).expand_dims(axis=0)
                ).transpose(axes=tuple(range(zy_dim)) + (-1, -2)) * (1.0 / np.prod(x.shape[:zx_dim])),
            ]
        x.backward(self.gradient[0])
        y.backward(self.gradient[1])
