#include "sparseir/svd.hpp"
#include "sparseir/impl/svd_impl.hpp"

namespace sparseir {

using xprec::DDouble;

template std::tuple<MatrixXd, VectorXd, MatrixXd>
compute_svd(const MatrixXd &A, int n_sv_hint, std::string strategy);

template std::tuple<MatrixX<DDouble>, VectorX<DDouble>, MatrixX<DDouble>>
compute_svd(const MatrixX<DDouble> &A, int n_sv_hint, std::string strategy);

} // namespace sparseir