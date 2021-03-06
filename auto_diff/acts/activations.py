import auto_diff as ad


__all__ = ['relu', 'leaky_relu', 'softmax', 'sigmoid']


def relu(x: ad.Operation) -> ad.Operation:
    """ReLU"""
    return ad.maximum(x, ad.constant(0.0))


def leaky_relu(x: ad.Operation, alpha=0.01) -> ad.Operation:
    """Leaky ReLU"""
    return ad.maximum(x, ad.constant(0.0)) + ad.minimum(x, ad.constant(0.0)) * ad.constant(alpha)


def softmax(x: ad.Operation) -> ad.Operation:
    """Softmax over the last axis.

    .. math::
       \\text{softmax}(x)_i = \\frac{e^{x_i}}{\\sum_j  e^{x_j}}

    The result and gradient of `exp` may be very large. For numerical stability, the maximum value is subtracted:

    .. math::
       \\text{softmax}(x)_i
       = \\frac{e^{x_i}}{\\sum_j  e^{x_j}}
       = \\frac{e^{x_i} \\cdot e^{-\\max(x)}}{\\sum_j  e^{x_j} \\cdot e^{-\\max(x)}}
       = \\frac{e^{x_i - \\max(x)}}{\\sum_j  e^{x_j - \\max(x)}}

    :param x: Input operation.
    :return: The result operation.
    """
    m = ad.max(x, axis=-1, keepdims=True)
    e = ad.exp(x - m)
    s = ad.sum(e, axis=-1, keepdims=True)
    y = e / (s + 1e-8)
    y.name = 'softmax(%s)' % x.name
    return y


def sigmoid(x: ad.Operation) -> ad.Operation:
    """Sigmoid"""
    return 1.0 / (1.0 + ad.exp(-x))
