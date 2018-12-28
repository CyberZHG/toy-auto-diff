from typing import List
import auto_diff as ad
from .optim import Optimizer


class SGD(Optimizer):

    def __init__(self, lr=1e-3, **kwargs):
        self.lr = lr
        super(SGD, self).__init__(**kwargs)

    def update(self, weights: List[ad.OpVariable], session: ad.Session):
        for weight in weights:
            weight.update_add(- self.lr * weight.gradient)
