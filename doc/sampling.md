Sparse sampling
===============

Sparse sampling is a way to transition between the truncated IR representation
of a propagator (useful for convergence analysis) and a sparse set of
sampling points in imaginary time or Matsubara frequency.  This is mediated
by two classes:

 - `TauSampling`:  sparse sampling in imaginary time, useful for, e.g.,
   constructing Feynman diagrams with a spontaneous interation term.

 - `MatsubaraSampling`: sparse sampling in Matsubara frequencies, useful for,
   e.g., solving diagrammatic equations such as the Dyson equation.

**Warning**:
When storing data in sparse time/frequency, *always* store the sampling points
together with the data.  The exact location of the sampling points may be
different from between platforms and/or between releases.


Sparse sampling transformers
----------------------------

```{eval-rst}
.. autoclass:: irbasis3.sampling.TauSampling
    :members: tau, evaluate, fit

.. autoclass:: irbasis3.sampling.MatsubaraSampling
    :members: wn, evaluate, fit
```

Base classes
-------------

```{eval-rst}
.. autoclass:: irbasis3.sampling.SamplingBase
    :members:

.. autoclass:: irbasis3.sampling.DecomposedMatrix
    :members:
```
