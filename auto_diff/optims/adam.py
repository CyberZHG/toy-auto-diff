from typing import List
import numpy as np
import auto_diff as ad
from .optim import Optimizer


class Adam(Optimizer):

    def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0.0, amsgrad=False, **kwargs):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        self.amsgrad = amsgrad
        self.step_num = 0.0
        self.ms, self.vs, self.vhs = None, None, None
        super(Adam, self).__init__(**kwargs)

    def update(self, weights: List[ad.OpVariable], session: ad.Session):
        self.step_num += 1.0
        lr = self.lr
        if self.decay > 0.0:
            lr /= (1.0 + self.decay * self.step_num)
        lr_t = lr * np.sqrt(1.0 - self.beta_2 ** self.step_num) / (1.0 - self.beta_1 ** self.step_num)
        if self.ms is None:
            self.ms = [0.0] * len(weights)
            self.vs = [0.0] * len(weights)
            self.vhs = [0.0] * len(weights)
        for index, weight in enumerate(weights):
            self.ms[index] = (self.beta_1 * self.ms[index]) + (1.0 - self.beta_1) * weight.gradient
            self.vs[index] = (self.beta_2 * self.vs[index]) + (1.0 - self.beta_2) * np.square(weight.gradient)
            if self.amsgrad:
                self.vhs[index] = np.maximum(self.vhs[index], self.vs[index])
                weight.update_add(-lr_t * self.ms[index] / (np.sqrt(self.vhs[index]) + self.epsilon))
            else:
                weight.update_add(-lr_t * self.ms[index] / (np.sqrt(self.vs[index] + self.epsilon)))
