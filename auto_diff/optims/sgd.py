from typing import List
import auto_diff as ad
from .optim import Optimizer


class SGD(Optimizer):

    def __init__(self, momentum=0.0, decay=0.0, lr=1e-3, **kwargs):
        self.momentum = momentum
        self.moments = None
        self.decay = decay
        self.lr = lr
        self.step_num = 0.0
        super(SGD, self).__init__(**kwargs)

    def update(self, weights: List[ad.OpVariable], session: ad.Session):
        self.step_num += 1.0
        if self.decay == 0.0:
            lr = self.lr
        else:
            lr = self.lr / (1.0 + self.decay * self.step_num)
        if self.moments is None:
            self.moments = [0.0] * len(weights)
        for index, weight in enumerate(weights):
            self.moments[index] = self.momentum * self.moments[index] - lr * weight.gradient
            weight.update_add(self.moments[index])
