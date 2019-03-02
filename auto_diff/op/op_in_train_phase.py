from typing import Mapping, Union
import numpy as np
from .operation import Operation
from .op_placeholder import OpPlaceholder


class OpInTrainPhase(Operation):
    """Whether it is in training phase."""

    def __init__(self, **kwargs):
        self.inputs = []
        self.params = {}
        self.shape = ()
        super(OpInTrainPhase, self).__init__(**kwargs)
        self._enable_cache = False

    def _forward(self, feed_dict: Mapping[Union[str, OpPlaceholder], np.ndarray]) -> np.ndarray:
        """Find training information from feed dictionary."""
        return np.array(Operation.KEY_TRAINING in feed_dict and feed_dict[Operation.KEY_TRAINING], dtype=np.float64)

    def _backward(self, gradient: np.ndarray) -> None:
        """No backward operation needed."""
        pass
