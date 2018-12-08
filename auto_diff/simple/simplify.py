from typing import Optional, Sequence, Callable
from ..op import Operation
from .sp_const_sub_tree import sp_const_sub_tree


def simplify(op: Operation, simplifies: Optional[Sequence[Callable]] = None):
    if simplifies is None:
        simplifies = [
            sp_const_sub_tree,
        ]
    for sim in simplifies:
        op = sim(op)
    return op
