#include "sparseir/sve.hpp"
#include "sparseir/impl/sve_impl.hpp"

namespace sparseir {

using xprec::DDouble;

SVEResult::SVEResult() { }

SVEResult::SVEResult(const PiecewiseLegendrePolyVector &u_,
                     const Eigen::VectorXd &s_,
                     const PiecewiseLegendrePolyVector &v_, double epsilon_)
    : u(std::make_shared<PiecewiseLegendrePolyVector>(u_)),
      s(s_),
      v(std::make_shared<PiecewiseLegendrePolyVector>(v_)),
      epsilon(epsilon_)
{
}

std::tuple<PiecewiseLegendrePolyVector, Eigen::VectorXd,
           PiecewiseLegendrePolyVector>
SVEResult::part(double eps, int max_size) const
{
    eps = std::isnan(eps) ? this->epsilon : eps;
    double threshold = eps * s(0);

    int cut =
        std::count_if(s.data(), s.data() + s.size(),
                      [threshold](double val) { return val >= threshold; });

    if (max_size > 0) {
        cut = std::min(cut, max_size);
    }
    std::vector<PiecewiseLegendrePoly> u_part_(u->begin(), u->begin() + cut);
    PiecewiseLegendrePolyVector u_part(u_part_);
    Eigen::VectorXd s_part(s.head(cut));
    std::vector<PiecewiseLegendrePoly> v_part_(v->begin(), v->begin() + cut);
    PiecewiseLegendrePolyVector v_part(v_part_);
    return std::make_tuple(u_part, s_part, v_part);
}

std::tuple<double, std::string, std::string>
choose_accuracy(double epsilon, const std::string &Twork)
{
    using std::sqrt;

    if (Twork != "Float64" && Twork != "Float64x2") {
        throw std::invalid_argument(
            "Twork must be either 'Float64' or 'Float64x2'");
    }
    if (Twork == "Float64") {
        if (epsilon >= sqrt(std::numeric_limits<double>::epsilon())) {
            return std::make_tuple(epsilon, Twork, "default");
        } else {
            std::cerr << "Warning: Basis cutoff is " << epsilon
                      << ", which is below sqrt(ε) with ε = "
                      << std::numeric_limits<double>::epsilon() << ".\n"
                      << "Expect singular values and basis functions for large "
                         "l to have lower precision than the cutoff.\n";
            return std::make_tuple(epsilon, Twork, "accurate");
        }
    } else {
        if (epsilon >= sqrt(std::numeric_limits<xprec::DDouble>::epsilon())) {
            return std::make_tuple(epsilon, Twork, "default");
        } else {
            std::cerr << "Warning: Basis cutoff is " << epsilon
                      << ", which is below sqrt(ε) with ε = "
                      << std::numeric_limits<xprec::DDouble>::epsilon() << ".\n"
                      << "Expect singular values and basis functions for large "
                         "l to have lower precision than the cutoff.\n";
            return std::make_tuple(epsilon, Twork, "accurate");
        }
    }
}

std::tuple<double, std::string, std::string> choose_accuracy(double epsilon,
                                                             std::nullptr_t)
{
    if (epsilon >= std::sqrt(std::numeric_limits<double>::epsilon())) {
        return std::make_tuple(epsilon, "Float64", "default");
    } else {
        if (epsilon < std::sqrt(std::numeric_limits<double>::epsilon())) {
            std::cerr
                << "Warning: Basis cutoff is " << epsilon << ", which is"
                << " below sqrt(ε) with ε = "
                << std::numeric_limits<double>::epsilon() << ".\n"
                << "Expect singular values and basis functions for large l"
                << " to have lower precision than the cutoff.\n";
        }
        return std::make_tuple(epsilon, "Float64x2", "default");
    }
}

std::tuple<double, std::string, std::string> choose_accuracy(std::nullptr_t,
                                                             std::string Twork)
{
    if (Twork == "Float64x2") {
        const double epsilon = 2.220446049250313e-16;
        return std::make_tuple(epsilon, Twork, "default");
    } else {
        return std::make_tuple(
            std::sqrt(std::numeric_limits<double>::epsilon()), Twork,
            "default");
    }
}

std::tuple<double, std::string, std::string>
choose_accuracy_epsilon_nan(std::string Twork)
{
    if (Twork == "Float64x2") {
        const double epsilon = 2.220446049250313e-16;
        return std::make_tuple(epsilon, Twork, "default");
    } else {
        return std::make_tuple(
            std::sqrt(std::numeric_limits<double>::epsilon()), Twork,
            "default");
    }
}

std::tuple<double, std::string, std::string> choose_accuracy(std::nullptr_t,
                                                             std::nullptr_t)
{
    const double epsilon = 2.220446049250313e-16;
    return std::make_tuple(epsilon, "Float64x2", "default");
}

std::tuple<double, std::string, std::string>
auto_choose_accuracy(double epsilon, std::string Twork, std::string svd_strat)
{
    std::string auto_svd_strat;
    if (std::isnan(epsilon)) {
        std::tie(epsilon, Twork, auto_svd_strat) =
            choose_accuracy_epsilon_nan(Twork);
    } else {
        std::tie(epsilon, Twork, auto_svd_strat) =
            choose_accuracy(epsilon, Twork);
    }
    std::string final_svd_strat =
        (svd_strat == "auto") ? auto_svd_strat : svd_strat;
    return std::make_tuple(epsilon, Twork, final_svd_strat);
}

void canonicalize(PiecewiseLegendrePolyVector &u,
                  PiecewiseLegendrePolyVector &v)
{
    for (size_t i = 0; i < u.size(); ++i) {
        double gauge = std::copysign(1.0, u.polyvec[i](1.0));
        u.polyvec[i].data *= gauge;
        v.polyvec[i].data *= gauge;
    }
}

/*
SVEResult compute_sve(const std::shared_ptr<AbstractKernel> &kernel,
                      double epsilon, double cutoff, int lmax, int n_gauss,
                      std::string Twork)
{
    double safe_epsilon;
    std::string Twork_actual;
    std::string svd_strategy_actual;
    std::tie(safe_epsilon, Twork_actual, svd_strategy_actual) =
        sparseir::auto_choose_accuracy(epsilon, Twork);

    if (Twork_actual == "Float64") {
        return std::get<0>(pre_postprocess<double>(kernel, safe_epsilon,
                                                   n_gauss, cutoff, lmax));
    } else if (Twork_actual == "Float64x2") {
        return std::get<0>(pre_postprocess<xprec::DDouble>(
            kernel, safe_epsilon, n_gauss, cutoff, lmax));
    } else {
        throw std::invalid_argument(
            "Twork must be either 'Float64' or 'Float64x2'");
    }
}
*/

// Explicit template instantiations
template SVEResult compute_sve(const LogisticKernel &, double, double, int, int,
                               std::string);
template SVEResult compute_sve(const RegularizedBoseKernel &, double, double,
                               int, int, std::string);

template class SamplingSVE<LogisticKernel, double>;
template class SamplingSVE<LogisticKernel, DDouble>;
template class SamplingSVE<RegularizedBoseKernel, double>;
template class SamplingSVE<RegularizedBoseKernel, DDouble>;
template class SamplingSVE<ReducedKernel<LogisticKernel>, double>;
template class SamplingSVE<ReducedKernel<LogisticKernel>, DDouble>;
template class SamplingSVE<LogisticKernelOdd, double>;
template class SamplingSVE<LogisticKernelOdd, DDouble>;

// Explicit template instantiations for CentrosymmSVE
template class CentrosymmSVE<LogisticKernel, double>;
template class CentrosymmSVE<LogisticKernel, DDouble>;
template class CentrosymmSVE<RegularizedBoseKernel, double>;
template class CentrosymmSVE<RegularizedBoseKernel, DDouble>;

template std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>,
                    std::vector<Eigen::MatrixXd>>
truncate(const std::vector<Eigen::MatrixXd> &u,
         const std::vector<Eigen::VectorXd> &s,
         const std::vector<Eigen::MatrixXd> &v, double rtol, int lmax);

template std::tuple<
    std::vector<Eigen::Matrix<DDouble, Eigen::Dynamic, Eigen::Dynamic>>,
    std::vector<Eigen::Vector<DDouble, Eigen::Dynamic>>,
    std::vector<Eigen::Matrix<DDouble, Eigen::Dynamic, Eigen::Dynamic>>>
truncate(
    const std::vector<Eigen::Matrix<DDouble, Eigen::Dynamic, Eigen::Dynamic>>
        &u,
    const std::vector<Eigen::Vector<DDouble, Eigen::Dynamic>> &s,
    const std::vector<Eigen::Matrix<DDouble, Eigen::Dynamic, Eigen::Dynamic>>
        &v,
    DDouble rtol, int lmax);

} // namespace sparseir