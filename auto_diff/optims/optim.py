from typing import List
import auto_diff as ad


class Optimizer(object):

    def __init__(self, **kwargs):
        pass

    def update(self, weights: List[ad.OpVariable], session: ad.Session):
        raise NotImplementedError('Optimizer not implemented.')
