#include <Eigen/Dense>

#include "sparseir/linalg.hpp"
#include "sparseir/impl/linalg_impl.hpp"

namespace sparseir {

using xprec::DDouble;

Eigen::MatrixXd pinv(const Eigen::MatrixXd &A, double tolerance)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU |
                                                 Eigen::ComputeThinV);
    const auto &singularValues = svd.singularValues();
    Eigen::MatrixXd singularValuesInv =
        Eigen::MatrixXd::Zero(A.cols(), A.rows());
    for (int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) > tolerance)
            singularValuesInv(i, i) = 1.0 / singularValues(i);
    }
    return svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
}

// Explicit template instantiations

// lmul
template void lmul<double>(const QRPackedQ<double>, Eigen::MatrixX<double> &);
template void lmul<DDouble>(const QRPackedQ<DDouble>,
                            Eigen::MatrixX<DDouble> &);

// mul
template void mul<double>(Eigen::MatrixX<double> &, const QRPackedQ<double> &,
                          const Eigen::MatrixX<double> &);
template void mul<DDouble>(Eigen::MatrixX<DDouble> &,
                           const QRPackedQ<DDouble> &,
                           const Eigen::MatrixX<DDouble> &);

// reflector
template double reflector<Eigen::Matrix<double, -1, 1>>(
    Eigen::MatrixBase<Eigen::Matrix<double, -1, 1>> &);
template DDouble reflector<Eigen::Matrix<DDouble, -1, 1>>(
    Eigen::MatrixBase<Eigen::Matrix<DDouble, -1, 1>> &);
template double reflector<Eigen::Block<
    Eigen::Block<Eigen::Matrix<double, -1, -1, 0, -1, -1>, -1, 1, true>, -1, 1,
    false>>(
    Eigen::MatrixBase<Eigen::Block<
        Eigen::Block<Eigen::Matrix<double, -1, -1, 0, -1, -1>, -1, 1, true>, -1,
        1, false>> &);

// reflectorApply
template void reflectorApply<double>(
    Eigen::VectorBlock<Eigen::Block<Eigen::MatrixX<double>, -1, 1, true>, -1>
        &x,
    double tau, Eigen::Block<Eigen::MatrixX<double>> &A);
template void reflectorApply<DDouble>(
    Eigen::VectorBlock<Eigen::Block<Eigen::MatrixX<DDouble>, -1, 1, true>, -1>
        &x,
    DDouble tau, Eigen::Block<Eigen::MatrixX<DDouble>> &A);

// rrqr
template std::pair<QRPivoted<double>, int> rrqr(Eigen::MatrixX<double> &A,
                                                double rtol);
template std::pair<QRPivoted<DDouble>, int> rrqr(Eigen::MatrixX<DDouble> &A,
                                                 DDouble rtol);

// triu
// template <typename Derived>
// Eigen::MatrixBase<Derived> triu(Eigen::MatrixBase<Derived> &M);

// truncate_qr_result
template std::pair<Eigen::MatrixX<double>, Eigen::MatrixX<double>>
truncate_qr_result<double>(QRPivoted<double> &qr, int k);
template std::pair<Eigen::MatrixX<DDouble>, Eigen::MatrixX<DDouble>>
truncate_qr_result<DDouble>(QRPivoted<DDouble> &qr, int k);

// swapCols
template void swapCols<double>(Eigen::MatrixX<double> &A, int i, int j);
template void swapCols<DDouble>(Eigen::MatrixX<DDouble> &A, int i, int j);

// truncateQRResult
template std::pair<Eigen::MatrixX<double>, Eigen::MatrixX<double>>
truncateQRResult<double>(const Eigen::MatrixX<double> &Q,
                         const Eigen::MatrixX<double> &R, int k);
template std::pair<Eigen::MatrixX<DDouble>, Eigen::MatrixX<DDouble>>
truncateQRResult<DDouble>(const Eigen::MatrixX<DDouble> &Q,
                          const Eigen::MatrixX<DDouble> &R, int k);

// tsvd
template std::tuple<Eigen::MatrixX<double>, Eigen::VectorX<double>,
                    Eigen::MatrixX<double>>
tsvd<double>(const Eigen::MatrixX<double> &A, double rtol);
template std::tuple<Eigen::MatrixX<DDouble>, Eigen::VectorX<DDouble>,
                    Eigen::MatrixX<DDouble>>
tsvd<DDouble>(const Eigen::MatrixX<DDouble> &A, DDouble rtol);

} // namespace sparseir