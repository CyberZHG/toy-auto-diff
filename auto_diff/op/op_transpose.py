from typing import Mapping, Union, Optional, Sequence
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpTranspose(Operation):
    """Transpose the tensor.

    Basic operation without `axes`:

    .. math::
       Y = X^T

    Partial derivative of a single element:

    .. math::
       \\begin{array}{rcl}
       \\displaystyle \\frac{\\partial L}{\\partial x_{ij}}
       &=& \\displaystyle \\sum_{i,j}
       \\frac{\\partial L}{\\partial y_{ij}} \\cdot \\frac{\\partial y_{ij}}{\\partial x_{ij}} \\\\
       &=& \\displaystyle \\frac{\\partial L}{\\partial y_{ji}} \\cdot \\frac{\\partial y_{ji}}{\\partial x_{ij}} \\\\
       &=& \\displaystyle \\frac{\\partial L}{\\partial y_{ji}} \\cdot \\frac{\\partial x_{ij}}{\\partial x_{ij}} \\\\
       &=& \\displaystyle \\frac{\\partial L}{\\partial y_{ji}} \\\\
       \\end{array}

    Matrix derivative:

    .. math::
       \\frac{\\partial L}{\\partial X} = \\left ( \\frac{\\partial L}{\\partial Y} \\right )^T

    Generally, `axes` should be a permutation of the dimensions, suppose there is a function :math:`f` that maps from
    :math:`(0, 1, \dots, k)` to the new permutation, then this transpose operation would be:

    .. math::
       y_{i_1, i_2, \\dots, i_k} = x_{f(i_1), f(i_2), \\dots, f(i_k)}

    The partial derivative of :math:`x_{i_1, i_2, \dots, i_k}` is 1 only with
    :math:`y_{f(i_1)^{-1}, f(i_2)^{-1}, \\dots, f(i_k)^{-1}}`. Therefore the derivative should be another transpose
    operation with inverse mapping function :math:`f^{-1}`.
    """

    def __init__(self, x: Operation, axes: Optional[Sequence[int]] = None, **kwargs):
        """
        :param x: Input operation.
        :param axes: A permutation of dimensions. The dimensions will be reversed if it is None.
        :param kwargs: Arguments for parent.
        """
        self.inputs = [x]
        self.axes = axes
        if axes is None:
            self.reverse_axes = None
            self.shape = tuple(reversed(x.shape))
        else:
            self.reverse_axes = [0] * len(axes)
            for i, axis in enumerate(axes):
                self.reverse_axes[axis] = i
            self.shape = tuple(x.shape[axis] for axis in axes)
        super(OpTranspose, self).__init__(**kwargs)

    def _get_name(self) -> str:
        if self.axes is None:
            return 'transpose(%s)' % self.inputs[0].name
        return 'transpose(%s, axes=%s)' % (self.inputs[0].name, str(self.axes))

    def _get_op_name(self) -> str:
        if self.axes is None:
            return 'transpose(%s)' % self.inputs[0]._op_name
        return 'transpose(%s, axes=%s)' % (self.inputs[0]._op_name, str(self.axes))

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Transpose the tensor."""
        return np.transpose(self.inputs[0].forward(feed_dict), axes=self.axes)

    def _backward(self, gradient: Operation) -> None:
        """Transpose the gradients to its old shape."""
        self.gradient = gradient.transpose(axes=self.reverse_axes)
        self.inputs[0].backward(self.gradient)
