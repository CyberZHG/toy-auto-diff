Element-wise Operations
=======================

By default, numpy uses element-wise operations when using operators like `+`, `-` and `*`. The back-propagation is easy to implement when the two elements have the same shape. However, `broadcasting <https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html>`_ must be considered for all these operations. Suppose there is an element :math:`x` whose shape is `(1, 3, 1)` before broadcasting, and the expanded element :math:`x'` has shape `(1, 2, 3, 4)` after broadcasting. Then:

.. math::
   \frac{\partial L}{\partial x_{1,j,1}} = \sum_{l,i,k} \frac{\partial L}{\partial x_{l,i,j,k}'}

The derivative is the sum over the broadcasted dimensions, and the expanded dimensions in the head should be removed.
