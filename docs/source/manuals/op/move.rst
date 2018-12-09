Movements
=========

Operations like `transpose`, `reshape` and `squeeze` only move the elements to other locations. The number of elements would not change after these operations. The back propagation of a movement is another movement, and more specifically, its inverse operation.

transpose
---------

The inverse operation of `transpose` is `transpose`.

.. autoclass:: auto_diff.OpTranspose
    :special-members:
    :show-inheritance:
