Sparse sampling
===============

Sparse sampling is a way to transition between the truncated IR representation
of a propagator (useful for convergence analysis) and a sparse set of
sampling points in imaginary time or Matsubara frequency.  This is mediated
by two classes:

 - :class:`sparse_ir.TauSampling`:
   sparse sampling in imaginary time, useful for, e.g., constructing Feynman
   diagrams with a spontaneous interation term.

 - :class:`sparse_ir.MatsubaraSampling`:
   sparse sampling in Matsubara frequencies, useful for, e.g., solving
   diagrammatic equations such as the Dyson equation.

All sampling classes contain ``sampling_points``, which are the corresponding
sampling points in time or frequency, and a method ``evaluate()``, which allows
one to go from coefficients to sampling points, and a method ``fit()`` to go
back::

         ________________                   ___________________
        |                |    evaluate()   |                   |
        |     Basis      |---------------->|     Value on      |
        |  coefficients  |<----------------|  sampling_points  |
        |________________|      fit()      |___________________|


.. warning::
   When storing data in sparse time/frequency, *always* store the sampling
   points together with the data.  The exact location of the sampling points
   may be different from between platforms and/or between releases.


Sparse sampling transformers
----------------------------

.. autoclass:: sparse_ir.TauSampling
    :members: tau, evaluate, fit

.. autoclass:: sparse_ir.MatsubaraSampling
    :members: wn, evaluate, fit


Base classes
-------------

.. autoclass:: sparse_ir.sampling.AbstractSampling
    :members:

.. autoclass:: sparse_ir.sampling.DecomposedMatrix
    :members:

.. autoclass:: sparse_ir.sampling.SplitDecomposedMatrix
    :members:

.. autoclass:: sparse_ir.sampling.ConditioningWarning
    :members:
