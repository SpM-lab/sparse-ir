# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT

class AbstractBasis:
    """Abstract base class for bases on the imaginary-time axis.

    Let `basis` be an abstract basis.  Then we can expand a two-point
    propagator  `G(τ)`, where `τ` is imaginary time, into a set of basis
    functions::

        G(τ) == sum(basis.u[l](τ) * g[l] for l in range(basis.size)) + ϵ(τ),

    where `basis.u[l]` is the `l`-th basis function, `g[l]` is the associated
    expansion coefficient and `ϵ(τ)` is an error term.  Similarly, the Fourier
    transform `Ĝ(n)`, where `n` is now a Matsubara frequency, can be expanded
    as follows:

        Ĝ(n) == sum(basis.uhat[l](n) * g[l] for l in range(basis.size)) + ϵ(n),

    where `basis.uhat[l]` is now the Fourier transform of the basis function.
    """
    @property
    def u(self):
        """Basis functions on the imaginary time axis.

        Set of IR basis functions on the imaginary time (`tau`) axis.
        To obtain the value of all basis functions at a point or a array of
        points `tau`, you can call the function ``utau)``.  To obtain a single
        basis function, a slice or a subset `l`, you can use ``u[l]``.
        """
        raise NotImplementedError()

    @property
    def uhat(self):
        """Basis functions on the reduced Matsubara frequency (`wn`) axis.

        To obtain the value of all basis functions at a Matsubara frequency
        or a array of points `wn`, you can call the function ``uhat(wn)``.
        Note that we expect reduced frequencies, which are simply even/odd
        numbers for bosonic/fermionic objects. To obtain a single basis
        function, a slice or a subset `l`, you can use ``uhat[l]``.
        """
        raise NotImplementedError()

    @property
    def statistics(self):
        """Quantum statistic (`"F"` for fermionic, `"B"` for bosonic)"""
        raise NotImplementedError()

    def __getitem__(self, index):
        """Return basis functions/singular values for given index/indices.

        This can be used to truncate the basis to the n most significant
        singular values: `basis[:3]`.
        """
        raise NotImplementedError()

    @property
    def shape(self):
        """Shape of the basis function set"""
        raise NotImplementedError()

    @property
    def size(self):
        """Number of basis functions / singular values"""
        raise NotImplementedError()

    @property
    def significance(self):
        """Significances of the basis functions

        Vector of significance values, one for each basis function.  Each
        value is a number between 0 and 1 which is a a-priori bound on the
        (relative) error made by discarding the associated coefficient.
        """
        return NotImplementedError()

    @property
    def accuracy(self):
        """Accuracy of the basis.

        Upper bound to the relative error of reprensenting a propagator with
        the given number of basis functions (number between 0 and 1).
        """
        return self.significance[-1]

    @property
    def lambda_(self):
        """Basis cutoff parameter, `Λ == β * wmax`, or None if not present"""
        raise NotImplementedError()

    @property
    def beta(self):
        """Inverse temperature"""
        raise NotImplementedError()

    @property
    def wmax(self):
        """Real frequency cutoff or `None` if not present"""
        raise NotImplementedError()

    def default_tau_sampling_points(self):
        """Default sampling points on the imaginary time axis"""
        raise NotImplementedError()

    def default_matsubara_sampling_points(self, *, positive_only=False):
        """Default sampling points on the imaginary frequency axis

        Arguments:
            positive_only (bool):
                Only return non-negative frequencies.  This is useful if the
                object to be fitted is symmetric in Matsubura frequency,
                ``ghat(w) == ghat(-w).conj()``, or, equivalently, real in
                imaginary time.
        """
        raise NotImplementedError()

    @property
    def is_well_conditioned(self):
        """Returns True if the sampling is expected to be well-conditioned"""
        return True
