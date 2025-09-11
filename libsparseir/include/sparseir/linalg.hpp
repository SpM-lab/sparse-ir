#pragma once

#include <iostream>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "xprec/ddouble-header-only.hpp"

namespace sparseir {

template <typename Derived>
int argmax(const Eigen::MatrixBase<Derived> &vec)
{
    // Ensure this function is only used for 1D Eigen column vectors
    static_assert(Derived::ColsAtCompileTime == 1,
                  "argmax function is only for Eigen column vectors.");

    int maxIndex = 0;
    auto maxValue = vec(0);

    for (int i = 1; i < vec.size(); ++i) {
        if (vec(i) > maxValue) {
            maxValue = vec(i);
            maxIndex = i;
        }
    }

    return maxIndex;
}

template <typename T>
struct QRPivoted
{
    Eigen::MatrixX<T> factors;
    Eigen::VectorX<T> taus;
    Eigen::VectorX<int> jpvt;
};

template <typename T>
struct QRPackedQ
{
    Eigen::MatrixX<T> factors;
    Eigen::VectorX<T> taus;
};

template <typename T>
void lmul(const QRPackedQ<T> Q, Eigen::MatrixX<T> &B);

template <typename T>
void mul(Eigen::MatrixX<T> &C, const QRPackedQ<T> &Q,
         const Eigen::MatrixX<T> &B);

// TODO: FIX THIS
template <typename T>
Eigen::MatrixX<T> getPropertyP(const QRPivoted<T> &F)
{
    int n = F.factors.cols();
    Eigen::MatrixX<T> P = Eigen::MatrixX<T>::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        P(F.jpvt[i], i) = 1;
    }
    return P;
}

template <typename T>
QRPackedQ<T> getPropertyQ(const QRPivoted<T> &F)
{
    return QRPackedQ<T>{F.factors, F.taus};
}

template <typename T>
Eigen::MatrixX<T> getPropertyR(const QRPivoted<T> &F)
{
    int m = F.factors.rows();
    int n = F.factors.cols();

    Eigen::MatrixX<T> upper = Eigen::MatrixX<T>::Zero(std::min(m, n), n);

    for (int i = 0; i < upper.rows(); ++i) {
        for (int j = i; j < upper.cols(); ++j) {
            upper(i, j) = F.factors(i, j);
        }
    }
    return upper;
}

// General template for _copysign, handles standard floating-point types like
// double and float
inline double _copysign(double x, double y) { return std::copysign(x, y); }

// Specialization for xprec::DDouble type, assuming xprec::copysign is defined
inline xprec::DDouble _copysign(xprec::DDouble x, xprec::DDouble y)
{
    return xprec::copysign(x, y);
}

/*
This implementation is based on Julia's LinearAlgebra.refrector! function.

Elementary reflection similar to LAPACK. The reflector is not Hermitian but
ensures that tridiagonalization of Hermitian matrices become real. See lawn72
*/
template <typename Derived>
typename Derived::Scalar reflector(Eigen::MatrixBase<Derived> &x);

/*
This implementation is based on Julia's LinearAlgebra.reflectorApply! function.

    reflectorApply!(x, τ, A)

Multiplies `A` in-place by a Householder reflection on the left. It is
equivalent to `A .= (I - conj(τ)*[1; x[2:end]]*[1; x[2:end]]')*A`.
*/
template <typename T>
void reflectorApply(
    Eigen::VectorBlock<Eigen::Block<Eigen::MatrixX<T>, -1, 1, true>, -1> &x,
    T tau, Eigen::Block<Eigen::MatrixX<T>> &A);

/*
A will be modified in-place.
*/
template <typename T>
std::pair<QRPivoted<T>, int> rrqr(Eigen::MatrixX<T> &A,
                                  T rtol = std::numeric_limits<T>::epsilon());

template <typename T>
std::pair<Eigen::MatrixX<T>, Eigen::MatrixX<T>> truncate_qr_result(QRPivoted<T> &qr, int k);

// Swap columns of a matrix A
template <typename T>
void swapCols(Eigen::MatrixX<T> &A, int i, int j);

// Truncate RRQR result to low rank
template <typename T>
std::pair<Eigen::MatrixX<T>, Eigen::MatrixX<T>>
truncateQRResult(const Eigen::MatrixX<T> &Q, const Eigen::MatrixX<T> &R, int k);

// Truncated SVD (TSVD)
template <typename T>
std::tuple<Eigen::MatrixX<T>, Eigen::VectorX<T>, Eigen::MatrixX<T>>
tsvd(const Eigen::MatrixX<T> &A, T rtol = std::numeric_limits<T>::epsilon());

Eigen::MatrixXd pinv(const Eigen::MatrixXd &A, double tolerance = 1e-6);

} // namespace sparseir
