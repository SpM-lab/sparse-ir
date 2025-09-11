#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "xprec/ddouble-header-only.hpp"

namespace sparseir {

template <typename T>
void lmul(const QRPackedQ<T> Q, Eigen::MatrixX<T> &B)
{
    Eigen::MatrixX<T> A_factors = Q.factors;
    Eigen::VectorX<T> A_tau = Q.taus;
    int mA = A_factors.rows();
    int nA = A_factors.cols();
    int mB = B.rows();
    int nB = B.cols();

    if (mA != mB) {
        throw std::invalid_argument("DimensionMismatch: matrix A has different "
                                    "dimensions than matrix B");
    }

    for (int k = std::min(mA, nA) - 1; k >= 0; --k) {
        for (int j = 0; j < nB; ++j) {
            T vBj = B(k, j);
            for (int i = k + 1; i < mB; ++i) {
                vBj += A_factors(i, k) * B(i, j);
            }
            vBj = A_tau(k) * vBj;
            B(k, j) -= vBj;
            for (int i = k + 1; i < mB; ++i) {
                B(i, j) -= A_factors(i, k) * vBj;
            }
        }
    }
}

template <typename T>
void mul(Eigen::MatrixX<T> &C, const QRPackedQ<T> &Q,
         const Eigen::MatrixX<T> &B)
{

    int mB = B.rows();
    int nB = B.cols();
    int mC = C.rows();
    int nC = C.cols();

    if (nB != nC) {
        throw std::invalid_argument(
            "DimensionMismatch: number of columns in B and C do not match");
    }

    if (mB < mC) {
        C.topRows(mB) = B;
        C.bottomRows(mC - mB).setZero();
        lmul<T>(Q, C);
    } else {
        C = B;
        lmul<T>(Q, C);
    }
}

/*
This implementation is based on Julia's LinearAlgebra.refrector! function.

Elementary reflection similar to LAPACK. The reflector is not Hermitian but
ensures that tridiagonalization of Hermitian matrices become real. See lawn72
*/
template <typename Derived>
typename Derived::Scalar reflector(Eigen::MatrixBase<Derived> &x)
{
    using T = typename Derived::Scalar;

    int n = x.size();
    if (n == 0)
        return T(0);

    T xi1 = x(0);       // First element of x
    T normu = x.norm(); // Norm of vector x
    if (normu == T(0))
        return T(0);

    // Calculate ν using copysign, which gives normu with the sign of xi1
    T nu = _copysign(normu, xi1);
    xi1 += nu;  // Update xi1
    x(0) = -nu; // Set first element to -ν

    // Divide remaining elements by the new xi1 value
    for (int i = 1; i < n; ++i) {
        x(i) /= xi1;
    }

    return xi1 / nu;
}

/*
This implementation is based on Julia's LinearAlgebra.reflectorApply! function.

    reflectorApply!(x, τ, A)

Multiplies `A` in-place by a Householder reflection on the left. It is
equivalent to `A .= (I - conj(τ)*[1; x[2:end]]*[1; x[2:end]]')*A`.
*/
template <typename T>
void reflectorApply(
    Eigen::VectorBlock<Eigen::Block<Eigen::MatrixX<T>, -1, 1, true>, -1> &x,
    T tau, Eigen::Block<Eigen::MatrixX<T>> &A)
{
    int m = A.rows();
    int n = A.cols();

    if (x.size() != m) {
        throw std::invalid_argument(
            "Reflector length must match the first dimension of matrix A");
    }

    if (m == 0)
        return;

    // Loop over each column of A
    for (int j = 0; j < n; ++j) {
        // Equivalent to `Aj = view(A, 2:m, j)` and `xj = view(x, 2:m)`
        auto Aj = A.block(1, j, m - 1, 1).reshaped();
        auto xj = x.tail(m - 1);

        // We expect tau to be real, so we use conj(tau) = tau
        T conj_tau = tau;
        // Compute vAj = conj(tau) * (A(0, j) + xj.dot(Aj));
        T vAj = conj_tau * (A(0, j) + xj.dot(Aj));

        // Update A(0, j)
        A(0, j) -= vAj;

        // Apply axpy operation: Aj -= vAj * xj
        Aj.noalias() -= vAj * xj;
    }
}

/*
A will be modified in-place.
*/
template <typename T>
std::pair<QRPivoted<T>, int> rrqr(Eigen::MatrixX<T> &A, T rtol)
{
    using std::abs;
    using std::sqrt;

    int m = A.rows();
    int n = A.cols();
    int k = std::min(m, n);
    int rk = k;
    Eigen::VectorX<int> jpvt = Eigen::VectorX<int>::LinSpaced(n, 0, n - 1);
    Eigen::VectorX<T> taus(k);

    Eigen::VectorX<T> xnorms = A.colwise().norm();
    Eigen::VectorX<T> pnorms = xnorms;
    T sqrteps = sqrt(std::numeric_limits<T>::epsilon());

    for (int i = 0; i < k; ++i) {

        int pvt = argmax(pnorms.tail(n - i)) + i;
        if (i != pvt) {
            std::swap(jpvt[i], jpvt[pvt]);
            std::swap(xnorms[pvt], xnorms[i]);
            std::swap(pnorms[pvt], pnorms[i]);
            A.col(i).swap(A.col(pvt)); // swapcols!
        }

        auto Ainp = A.col(i).tail(m - i);
        T tau_i = reflector(Ainp);

        taus[i] = tau_i;

        auto block = A.bottomRightCorner(m - i, n - (i + 1));
        reflectorApply<T>(Ainp, tau_i, block);

        for (int j = i + 1; j < n; ++j) {
            T temp = abs((A(i, j))) / pnorms[j];
            temp = std::max<T>(T(0), (T(1) + temp) * (T(1) - temp));
            // abs2
            T temp2 = temp * (pnorms(j) / xnorms(j)) * (pnorms(j) / xnorms(j));
            if (temp2 < sqrteps) {
                auto recomputed = A.col(j).tail(m - i - 1).norm();
                pnorms(j) = recomputed;
                xnorms(j) = recomputed;
            } else {
                pnorms(j) = pnorms(j) * sqrt(temp);
            }
        }

        if (abs(A(i, i)) < rtol * abs((A(0, 0)))) {
            A.bottomRightCorner(m - i, n - i).setZero();
            taus.tail(k - i).setZero();
            rk = i;
            break;
        }
    }

    return {QRPivoted<T>{A, taus, jpvt}, rk};
}

template <typename T>
std::pair<Eigen::MatrixX<T>, Eigen::MatrixX<T>> truncate_qr_result(QRPivoted<T> &qr, int k)
{
    int m = qr.factors.rows();
    int n = qr.factors.cols();
    if (k < 0 || k > std::min(m, n)) {
        throw std::domain_error("Invalid rank, must be in [0, " +
                                std::to_string(std::min(m, n)) + "]");
    }

    // Extract Q matrix

    Eigen::MatrixX<T> k_factors = qr.factors.topLeftCorner(qr.factors.rows(), k);
    Eigen::VectorX<T> k_taus = qr.taus.head(k);
    auto Qfull = QRPackedQ<T>{k_factors, k_taus};

    Eigen::MatrixX<T> Q = Eigen::MatrixX<T>::Identity(m, k);
    lmul<T>(Qfull, Q);
    // Extract R matrix
    auto R = qr.factors.topRows(k);
    for (int j = 0; j < R.rows(); ++j) {
        for (int i = j + 1; i < R.rows(); ++i) {
            R(i, j) = T(0);
        }
    }
    return std::make_pair(Q, R);
}

// Swap columns of a matrix A
template <typename T>
void swapCols(Eigen::MatrixX<T> &A, int i, int j)
{
    if (i != j) {
        A.col(i).swap(A.col(j));
    }
}

// Truncate RRQR result to low rank
template <typename T>
std::pair<Eigen::MatrixX<T>, Eigen::MatrixX<T>>
truncateQRResult(const Eigen::MatrixX<T> &Q, const Eigen::MatrixX<T> &R, int k)
{
    int m = Q.rows();
    int n = R.cols();

    if (k < 0 || k > std::min(m, n)) {
        throw std::domain_error("Invalid rank, must be in [0, min(m, n)]");
    }

    Eigen::MatrixX<T> Q_trunc = Q.leftCols(k);
    Eigen::MatrixX<T> R_trunc = R.topLeftCorner(k, n);
    return std::make_pair(Q_trunc, R_trunc);
}

// Truncated SVD (TSVD)
template <typename T>
std::tuple<Eigen::MatrixX<T>, Eigen::VectorX<T>, Eigen::MatrixX<T>>
tsvd(const Eigen::MatrixX<T> &A, T rtol)
{
    // Step 1: Apply RRQR to A
    QRPivoted<T> A_qr;
    int k;
    Eigen::MatrixX<T> A_ = A; // create a copy of A
    std::tie(A_qr, k) = rrqr<T>(A_, rtol);
    // Step 2: Truncate QR Result to rank k
    auto tqr = truncate_qr_result<T>(A_qr, k);
    auto p = A_qr.jpvt;
    auto Q_trunc = tqr.first;
    auto R_trunc = tqr.second;
    // TODO

    // Step 3: Compute SVD of R_trunc

    // Eigen::JacobiSVD<Eigen::MatrixX<T>> svd(R_trunc.transpose(),
    // Eigen::ComputeThinU | Eigen::ComputeThinV);
    //  There seems to be a bug in the latest version of Eigen3.
    //  Please first construct a Jacobi SVD and then compare the results.
    //  Do not use the svd_jacobi function directly.
    //  Better to write a wrrapper function for the SVD.
    Eigen::JacobiSVD<decltype(R_trunc)> svd;

    // The following comment is taken from Julia's implementation
    // # RRQR is an excellent preconditioner for Jacobi. One should then perform
    // # Jacobi on RT
    svd.compute(R_trunc.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Reconstruct A from QR factorization
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(p.size());
    perm.indices() = p;

    Eigen::MatrixX<T> U = Q_trunc * svd.matrixV();
    // implement invperm
    Eigen::MatrixX<T> V = (perm * svd.matrixU());

    Eigen::VectorX<T> s = svd.singularValues();
    // TODO: Create a return type for truncated SVD
    return std::make_tuple(U, s, V);
}

} // namespace sparseir