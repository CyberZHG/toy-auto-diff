import auto_diff as ad


__all__ = ['cross_entropy', 'mean_square_error']


def cross_entropy(y_true: ad.Operation, y_pred: ad.Operation) -> ad.Operation:
    """Cross entropy over the last axis.

    .. math::
       H(y, \\hat{y}) = - \\sum_i y_i \\log P \\left ( \\hat{y}_i \\right )

    :param y_true: Real label.
    :param y_pred: Probabilities for each label.
    :return: The result operation.
    """
    return ad.sum(-y_true * ad.log(y_pred), axis=-1)


def mean_square_error(y_true: ad.Operation, y_pred: ad.Operation) -> ad.Operation:
    return ad.sum(ad.square(y_true - y_pred), axis=-1)
