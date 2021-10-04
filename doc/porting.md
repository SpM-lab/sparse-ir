Porting
=======
For porting codes from irbasis, version 2, to irbasis3, you have two options:

  * using the `irbasis3.adapter` module (easier)
  * porting to the new `irbasis3` API (cleaner)

In both cases, please note that `irbasis3` now computes the basis on-the-fly
rather than loading it from a file.  This has some important consequences:

  * Singular values and basis functions may slightly differ from the old
    code, and may also slightly differ from platform to platform, but should
    be consistent within a few units in the last place (`2e-16`).

  * The sampling times and frequencies are chosen in an improved fashion
    and thus differ from the old code and may also differ slightly between
    versions and platforms.

In transferring persistent data, please therefore either directly transfer the
basis coefficients or store the sampling points together with the data.

Adapter module
--------------
For ease of porting, we provide the `irbasis3.adapter` module.  This module
is API-compatible with the `irbasis` package, version 2.  This means you
simply have to replace::

    import irbasis

with the following::

    import irbasis3.adapter as irbasis

and everything should work as expected.

```{eval-rst}
.. autoclass:: irbasis3.adapter.Basis
    :members:
```
