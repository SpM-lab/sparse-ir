Other bases
===========

Aside from the intermediate representation (IR), `sparse-ir` allows to work
with any bases derived from :class:`sparse_ir.abstract.AbstractBasis`.

In particular, we offer a variant of the discrete Lehmann representation
(DLR), with poles at the roots of the first discarded IR basis function
V_L: :class:`sparse_ir.dlr.DiscreteLehmannRepresentation`.

.. autoclass:: sparse_ir.dlr.DiscreteLehmannRepresentation
    :members:

Base classes
------------

.. autoclass:: sparse_ir.abstract.AbstractBasis
    :members:
