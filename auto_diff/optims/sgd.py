from typing import List
import auto_diff as ad
from .optim import Optimizer


class SGD(Optimizer):

    def __init__(self, lr=1e-3, momentum=0.0, decay=0.0, nesterov=False, **kwargs):
        self.lr = lr
        self.momentum = momentum
        self.moments = None
        self.decay = decay
        self.nesterov = nesterov
        self.step_num = 0.0
        super(SGD, self).__init__(**kwargs)

    def update(self, weights: List[ad.OpVariable], session: ad.Session):
        """

        Nesterov: instead of moving towards the momentum and calculating a new gradient, the current gradient is
                  regarded as the location after moving in the last step.

        :param weights:
        :param session:
        :return:
        """
        self.step_num += 1.0
        lr = self.lr
        if self.decay > 0.0:
            lr /= (1.0 + self.decay * self.step_num)
        if self.moments is None:
            self.moments = [0.0] * len(weights)
        for index, weight in enumerate(weights):
            self.moments[index] = self.momentum * self.moments[index] - lr * weight.gradient
            if self.nesterov:
                weight.update_add(self.momentum * self.moments[index] - lr * weight.gradient)
            else:
                weight.update_add(self.moments[index])
