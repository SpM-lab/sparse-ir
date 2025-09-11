// Template class for FiniteTempBasis
#pragma once

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <memory> // for std::shared_ptr
#include <vector>

namespace sparseir {

Eigen::VectorXd default_sampling_points(const PiecewiseLegendrePolyVector &u,
                                        int L);

// Abstract class with S = Fermionic or Bosonic
template <typename S>
class AbstractBasis {
public:
    virtual ~AbstractBasis() { }
    double beta;
    virtual double get_beta() const = 0;
    /*
     * @brief Significances of the basis functions.
     *
     * Vector of significance values, one for each basis function. Each value
     * is a number between 0 and 1 which provides an a-priori bound on the
     * relative error made by discarding the associated coefficient.
     *
     * @return Vector of significance values.
     */
    virtual const Eigen::VectorXd significance() const = 0;

    /**
     * @brief Accuracy of the basis.
     *
     * Upper bound to the relative error of representing a propagator with
     * the given number of basis functions (number between 0 and 1).
     *
     * @return Accuracy value.
     */
    virtual double get_accuracy() const = 0;
    virtual double get_wmax() const = 0;
    virtual size_t size() const = 0;
    virtual Eigen::VectorXd default_tau_sampling_points() const = 0;
    virtual std::vector<MatsubaraFreq<S>>
    default_matsubara_sampling_points(int L, bool fence = false,
                                      bool positive_only = false) const = 0;
};

template <typename S>
using IRTauFuncsType = PeriodicFunctions<S, PiecewiseLegendrePolyVector>;

template <typename S>
class FiniteTempBasis : public AbstractBasis<S> {
public:
    double lambda;
    std::shared_ptr<SVEResult> sve_result;
    double accuracy;
    double beta;
    std::shared_ptr<IRTauFuncsType<S>> u;
    std::shared_ptr<PiecewiseLegendrePolyVector> v;
    Eigen::VectorXd s;
    std::shared_ptr<PiecewiseLegendreFTVector<S>> uhat;
    std::shared_ptr<PiecewiseLegendreFTVector<S>> uhat_full;

    std::function<double(double, double)> weight_func; // weight function especially for bosonic basis

    template <typename K, typename = typename std::enable_if<is_concrete_kernel<K>::value>::type>
    FiniteTempBasis(double beta, double omega_max, double epsilon,
                    const K& kernel, int max_size = -1)
        : FiniteTempBasis(beta, omega_max, epsilon, kernel,
                          compute_sve(kernel, epsilon), max_size)
    {
    }

    template <typename K, typename = typename std::enable_if<is_concrete_kernel<K>::value>::type>
    FiniteTempBasis(double beta, double omega_max, double epsilon,
                    const K &kernel,
                    SVEResult sve_result, int max_size = -1)
    {
        if (sve_result.s.size() == 0) {
            throw std::runtime_error("SVE result sve_result.s is empty");
        }
        if (beta <= 0.0) {
            throw std::domain_error(
                "Inverse temperature beta must be positive");
        }
        if (omega_max < 0.0) {
            throw std::domain_error(
                "Frequency cutoff omega_max must be non-negative");
        }
        if (std::fabs(beta * omega_max - kernel.lambda_) > 1e-10) {
            throw std::runtime_error("Product of beta and omega_max must be "
                                     "equal to lambda in kernel");
        }
        this->beta = beta;
        this->lambda = kernel.lambda_;
        this->sve_result = std::make_shared<SVEResult>(sve_result);
        this->weight_func = kernel.template weight_func<double>(S());

        double wmax = this->lambda / beta;

        auto part_result = sve_result.part(epsilon, max_size);
        PiecewiseLegendrePolyVector u_ = std::get<0>(part_result);
        Eigen::VectorXd s_ = std::get<1>(part_result);
        PiecewiseLegendrePolyVector v_ = std::get<2>(part_result);
        double sve_result_s0 = sve_result.s(0);

        if (sve_result.s.size() > s_.size()) {
            this->accuracy = sve_result.s(s_.size()) / sve_result_s0;
        } else {
            this->accuracy = sve_result.s(s_.size() - 1) / sve_result_s0;
        }

        /*
        Port the following Julia code to C++:
        # The polynomials are scaled to the new variables by transforming
        the # knots according to: tau = β/2 * (x + 1), w = ωmax * y. Scaling #
        the data is not necessary as the normalization is inferred. ωmax =
        Λ(kernel) / β u_knots = (β / 2) .* (knots(u_) .+ 1) v_knots = ωmax .*
        knots(v_) u = PiecewiseLegendrePolyVector(u_, u_knots; Δx=(β / 2) .*
        Δx(u_), symm=symm(u_)) v = PiecewiseLegendrePolyVector(v_, v_knots;
        Δx=ωmax .* Δx(v_), symm=symm(v_))
        */

        auto u_knots_ = u_.polyvec[0].knots;
        auto v_knots_ = v_.polyvec[0].knots;

        Eigen::VectorXd u_knots = (beta / 2) * (u_knots_.array() + 1);
        Eigen::VectorXd v_knots = wmax * v_knots_;

        Eigen::VectorXd deltax4u = (beta / 2) * u_.get_delta_x();
        Eigen::VectorXd deltax4v = wmax * v_.get_delta_x();
        std::vector<int> u_symm_vec;
        for (std::size_t i = 0; i < u_.size(); ++i) {
            u_symm_vec.push_back(u_.polyvec[i].get_symm());
        }
        std::vector<int> v_symm_vec;
        for (std::size_t i = 0; i < v_.size(); ++i) {
            v_symm_vec.push_back(v_.polyvec[i].get_symm());
        }

        Eigen::VectorXi u_symm =
            Eigen::Map<Eigen::VectorXi>(u_symm_vec.data(), u_symm_vec.size());
        Eigen::VectorXi v_symm =
            Eigen::Map<Eigen::VectorXi>(v_symm_vec.data(), v_symm_vec.size());

        this->u = std::make_shared<PeriodicFunctions<S, PiecewiseLegendrePolyVector>>(
            std::make_shared<PiecewiseLegendrePolyVector>(u_, u_knots, deltax4u, u_symm), beta);
        this->v = std::make_shared<PiecewiseLegendrePolyVector>(
            v_, v_knots, deltax4v, v_symm);
        this->s =
            (std::sqrt(beta * wmax / 2) * std::pow(wmax, (kernel.ypower()))) *
            s_;

        Eigen::Tensor<double, 3> udata3d = sve_result.u->get_data();
        PiecewiseLegendrePolyVector uhat_base_full =
            PiecewiseLegendrePolyVector(sqrt(beta) * udata3d, *sve_result.u);
        S statistics = S();

        this->uhat_full = std::make_shared<PiecewiseLegendreFTVector<S>>(
            uhat_base_full, statistics, kernel.conv_radius());

        std::vector<PiecewiseLegendreFT<S>> uhat_polyvec;
        for (int i = 0; i < this->s.size(); ++i) {
            uhat_polyvec.push_back(this->uhat_full->operator[](i));
        }
        this->uhat =
            std::make_shared<PiecewiseLegendreFTVector<S>>(uhat_polyvec);
    }

    // Delegating constructor 1
    FiniteTempBasis(double beta, double omega_max, double epsilon,
                    int max_size = -1)
    {
        LogisticKernel kernel(beta * omega_max);
        SVEResult sve_result = compute_sve(kernel, epsilon);
        *this = FiniteTempBasis<S>(beta, omega_max, epsilon, kernel, sve_result, max_size);
    }

    // Delegating constructor 3
    template <typename K, typename = typename std::enable_if<is_concrete_kernel<K>::value>::type>
    FiniteTempBasis(double beta, double omega_max,
                    const K& kernel,
                    SVEResult sve_result, int max_size = -1)
        : FiniteTempBasis(beta, omega_max, sve_result.epsilon, kernel,
                          sve_result, max_size)
    {
    }

    // Overload operator[] for indexing (get a subset of the basis)
    FiniteTempBasis<S> operator[](const std::pair<int, int> &range) const
    {
        int new_size = range.second - range.first + 1;
        return FiniteTempBasis<S>(statistics(), this->get_beta(), this->get_wmax(),
                                  0.0, new_size, lambda, sve_result);
    }

    // Calculate significance
    const Eigen::VectorXd significance() const override { return s / s[0]; }

    // Getter for accuracy
    double get_accuracy() const override { return accuracy; }

    // Getter for ωmax
    double get_wmax() const override { return lambda / this->get_beta(); }
    double get_beta() const override { return beta; }
    size_t size() const override { return s.size(); }
    // Getter for SVEResult
    std::shared_ptr<const SVEResult> getSVEResult() const { return sve_result; }

    // Getter for kernel
    // std::shared_ptr<const K> getKernel() const { return kernel; }

    // Getter for Λ
    // double Lambda() const { return lambda; }

    // Default ω sampling points
    Eigen::VectorXd default_omega_sampling_points() const
    {
        Eigen::VectorXd y = default_sampling_points(*(sve_result->v),
                                                    static_cast<int>(s.size()));
        return this->get_wmax() * y.array();
    }

    // Rescale function
    FiniteTempBasis<S> rescale(double new_beta) const
    {
        double new_omega_max = lambda / new_beta;
        auto kernel = LogisticKernel(lambda);
        return FiniteTempBasis<S>(
            new_beta, new_omega_max, std::numeric_limits<double>::quiet_NaN(),
            kernel, *sve_result, static_cast<int>(s.size()));
    }

    Eigen::VectorXd default_tau_sampling_points() const override
    {
        int sz = size();
        auto x = default_sampling_points(*(this->sve_result->u), sz);
        Eigen::VectorXd smpl_taus = (this->beta / 2.0) * (x.array() + 1.0);
        std::sort(smpl_taus.data(), smpl_taus.data() + smpl_taus.size());

        // Check if the distribution is symmetric
        Eigen::VectorXd smpl_taus_reverse = Eigen::VectorXd::Constant(smpl_taus.size(), beta) - smpl_taus;
        std::sort(smpl_taus_reverse.data(), smpl_taus_reverse.data() + smpl_taus_reverse.size());
        Eigen::VectorXd diff = (smpl_taus - smpl_taus_reverse).array().abs();
        auto max_diff = diff.maxCoeff();
        if (max_diff > beta * 1e-5) {
            std::cerr << "Warning: The distribution of tau sampling points is not symmetric with respect to the center of the interval [0, beta]." << std::endl;
            std::cerr << "smpl_taus: " << smpl_taus << std::endl;
            std::cerr << "max_diff: " << max_diff << std::endl;
        }

        auto smpl_taus_symm = smpl_taus;
        if (smpl_taus.size() % 2 == 0) {
            auto halfN = smpl_taus.size() / 2;
            smpl_taus_symm.segment(halfN, halfN) = -smpl_taus.segment(0, halfN);
        } else {
            auto halfN = (smpl_taus.size() - 1) / 2;
            smpl_taus_symm.segment(halfN + 1, halfN) = -smpl_taus.segment(0, halfN);
        }

        std::sort(smpl_taus_symm.data(), smpl_taus_symm.data() + smpl_taus_symm.size());
        return smpl_taus_symm;
    }

    std::vector<MatsubaraFreq<S>>
    default_matsubara_sampling_points(int L, bool fence = false,
                                      bool positive_only = false) const override
    {
        return default_matsubara_sampling_points_impl(*this->uhat_full, L,
                                                      fence, positive_only);
    }

private:
    // Placeholder statistics function
    std::shared_ptr<S> statistics() const { return std::make_shared<S>(); }
};

// Default sampling points function
inline Eigen::VectorXd
default_sampling_points(const PiecewiseLegendrePolyVector &u, int L)
{
    if (u.xmin() != -1.0 || u.xmax() != 1.0)
        throw std::runtime_error("Expecting unscaled functions here.");

    if (static_cast<std::size_t>(L) < u.size()) {
        return u.polyvec[L].roots();
    } else {
        // Approximate roots by extrema
        PiecewiseLegendrePoly poly = u.polyvec.back();
        Eigen::VectorXd maxima = poly.deriv().roots();

        double left = (maxima[0] + poly.xmin) / 2.0;
        double right = (maxima[maxima.size() - 1] + poly.xmax) / 2.0;

        Eigen::VectorXd x0(maxima.size() + 2);
        x0[0] = left;
        x0.segment(1, maxima.size()) = maxima;
        x0[x0.size() - 1] = right;

        if (x0.size() != L) {
            std::cerr << "Warning: Expected " << L << " sampling points, got "
                      << x0.size() << ".\n";
        }

        return x0;
    }
}

template <typename S>
inline void fence_matsubara_sampling(std::vector<MatsubaraFreq<S>> &wn,
                                     bool positive_only)
{
    // Original implementation remains unchanged
    if (wn.empty()) {
        return;
    }

    std::vector<MatsubaraFreq<S>> outer_frequencies;
    if (positive_only) {
        outer_frequencies.push_back(wn.back());
    } else {
        outer_frequencies.push_back(wn.front());
        outer_frequencies.push_back(wn.back());
    }

    for (const auto &wn_outer : outer_frequencies) {
        int outer_val = wn_outer.n;
        int diff_val = 2 * static_cast<int>(std::round(0.025 * outer_val));
        int wn_diff = MatsubaraFreq<S>(diff_val).n;

        if (wn.size() >= 20) {
            wn.push_back(
                MatsubaraFreq<S>(wn_outer.n - sign(wn_outer) * wn_diff));
        }
        if (wn.size() >= 42) {
            wn.push_back(
                MatsubaraFreq<S>(wn_outer.n + sign(wn_outer) * wn_diff));
        }
    }

    std::sort(wn.begin(), wn.end());
    wn.erase(std::unique(wn.begin(), wn.end()), wn.end());
}

template <typename S>
std::vector<MatsubaraFreq<S>> default_matsubara_sampling_points_impl(
    const PiecewiseLegendreFTVector<S> &u_hat, int L, bool fence = false,
    bool positive_only = false)
{
    std::size_t l_requested = L;

    // Adjust l_requested based on statistics
    if (std::is_same<S, Fermionic>::value && l_requested % 2 != 0)
        l_requested += 1;
    else if (std::is_same<S, Bosonic>::value && l_requested % 2 == 0)
        l_requested += 1;

    std::vector<MatsubaraFreq<S>> omega_n;

    if (l_requested < u_hat.size()) {
        omega_n = sign_changes(u_hat[l_requested], positive_only);
    } else {
        omega_n = find_extrema(u_hat[u_hat.size() - 1], positive_only);
    }

    std::size_t expected_size = l_requested;
    if (positive_only)
        expected_size = (expected_size + 1) / 2;

    if (omega_n.size() != expected_size) {
        std::cerr << "Warning: Requested " << expected_size
                  << " sampling frequencies for basis size L = " << L
                  << ", but got " << omega_n.size() << ".\n";
    }

    if (fence) {
        fence_matsubara_sampling(omega_n, positive_only);
    }
    return omega_n;
}

inline std::pair<std::shared_ptr<FiniteTempBasis<Fermionic>>, std::shared_ptr<FiniteTempBasis<Bosonic>>>
finite_temp_bases(double beta, double omega_max, double epsilon,
                  const SVEResult& sve_result)
{
    LogisticKernel kernel(beta * omega_max);
    auto basis_f = std::make_shared<FiniteTempBasis<Fermionic>>(beta, omega_max, epsilon, kernel,
                                              sve_result);
    auto basis_b = std::make_shared<FiniteTempBasis<Bosonic>>(beta, omega_max, epsilon, kernel,
                                              sve_result);
    return std::make_pair(basis_f, basis_b);
}

inline std::pair<std::shared_ptr<FiniteTempBasis<Fermionic>>, std::shared_ptr<FiniteTempBasis<Bosonic>>>
finite_temp_bases(double beta, double omega_max,
                  double epsilon = std::numeric_limits<double>::quiet_NaN())
{
    LogisticKernel kernel(beta * omega_max);
    SVEResult sve_result = compute_sve(kernel, epsilon);
    auto basis_f = std::make_shared<FiniteTempBasis<Fermionic>>(beta, omega_max, epsilon, kernel,
                                              sve_result);
    auto basis_b = std::make_shared<FiniteTempBasis<Bosonic>>(beta, omega_max, epsilon, kernel,
                                              sve_result);
    return std::make_pair(basis_f, basis_b);
}




} // namespace sparseir
