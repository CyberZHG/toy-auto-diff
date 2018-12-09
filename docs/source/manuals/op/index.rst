Operations
==========

.. toctree::
   :maxdepth: 4

   special
   move
   element
   unary
   binary


:class:`auto_diff.Operation` is used for representing calculation graphs. All the operations (e.g. sum, dot) are subclasses of :class:`auto_diff.Operation`. The names and arguments of these operations follow the definitions of `numpy <https://www.numpy.org/>`_, therefore you can use the operations similar to numpy:

.. code-block:: python

   import auto_diff as np

   x = np.array([[1, 2], [3, 4]])
   y = np.array([[5, 6], [7, 8]])
   z = np.dot(x, y)


However, the result of operations is another operation, no real calculation is performed in this stage. You need to call ``forward`` explicitly to get the final result:

.. code-block:: python

   print(z)
   print(z.forward())


.. autoclass:: auto_diff.Operation
    :members:
    :undoc-members:
    :show-inheritance:
