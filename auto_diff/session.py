from typing import Union, Mapping, List
from .operation import Operation

__all__ = ['Session']


class Session(object):

    def __init__(self):
        self._step: int = 0

    def prepare(self):
        self._step += 1

    def run(self, fetches: Union[Operation, List[Operation], Mapping[str, Operation]], feed_dict=None):
        if feed_dict is None:
            feed_dict = {}
        feed_dict[Operation.STEP_KEY] = self._step
        if isinstance(fetches, Operation):
            return fetches.forward(feed_dict)
        if isinstance(fetches, list):
            return [fetch.forward(feed_dict) for fetch in fetches]
        if isinstance(fetches, dict):
            return {key: fetch.forward(feed_dict) for key, fetch in fetches.items()}
        raise NotImplementedError('Unknown type of fetches: %s' % type(fetches))
