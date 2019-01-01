from typing import List
import auto_diff as ad
from .optim import Optimizer


class SGD(Optimizer):

    def __init__(self, decay=0.0, lr=1e-3, **kwargs):
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
        for weight in weights:
            weight.update_add(- lr * weight.gradient)
