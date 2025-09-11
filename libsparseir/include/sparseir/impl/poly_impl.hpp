#pragma once

#include "sparseir/poly.hpp"

namespace sparseir {

// Template implementations for PiecewiseLegendreFT
template <typename S>
std::complex<double>
PiecewiseLegendreFT<S>::compute_unl_inner(const PiecewiseLegendrePoly &poly,
                                          int wn) const
{
    double wred = M_PI / 4.0 * wn;
    Eigen::VectorXcd phase_wi = phase_stable(poly, wn);
    std::complex<double> res = 0.0;

    int order_max = poly.data.rows();
    int segment_count = poly.data.cols();
    for (int order = 0; order < order_max; ++order) {
        for (int j = 0; j < segment_count; ++j) {
            double data_oj = poly.data(order, j);
            std::complex<double> tnl = get_tnl(order, wred * poly.delta_x(j));
            res += data_oj * tnl * phase_wi(j) / poly.norms(j);
        }
    }
    return res / std::sqrt(2.0);
}

template <typename S>
std::complex<double> PiecewiseLegendreFT<S>::giw(int wn) const
{
    std::complex<double> iw(0.0, M_PI / 2.0 * wn);
    if (wn == 0)
        return std::complex<double>(0.0, 0.0);
    std::complex<double> inv_iw = 1.0 / iw;
    std::vector<double> moments_vec(
        model.moments.data(), model.moments.data() + model.moments.size());
    std::complex<double> result = inv_iw * evalpoly(inv_iw, moments_vec);
    return result;
}

template <typename S>
std::complex<double>
PiecewiseLegendreFT<S>::evalpoly(const std::complex<double> &x,
                                 const std::vector<double> &coeffs) const
{
    std::complex<double> result(0, 0);
    for (int i = coeffs.size() - 1; i >= 0; --i) {
        result = result * x + coeffs[i];
    }
    return result;
}

template <typename S>
std::function<double(int)>
func_for_part(const PiecewiseLegendreFT<S> &polyFT,
              std::function<double(std::complex<double>)> part)
{
    if (part == nullptr) {
        int parity = polyFT.poly.get_symm();
        if (parity == 1) {
            part = std::is_same<S, Bosonic>::value
                       ? [](std::complex<double> x) { return x.real(); }
                       : [](std::complex<double> x) { return x.imag(); };
        } else if (parity == -1) {
            part = std::is_same<S, Bosonic>::value
                       ? [](std::complex<double> x) { return x.imag(); }
                       : [](std::complex<double> x) { return x.real(); };
        } else {
            throw std::runtime_error("Cannot detect parity");
        }
    }

    return [polyFT, part](int n) -> double {
        auto omega = MatsubaraFreq<S>(2 * n + polyFT.zeta());
        return part(polyFT(omega));
    };
}

template <typename S>
std::vector<MatsubaraFreq<S>> sign_changes(const PiecewiseLegendreFT<S> &u_hat,
                                           bool positive_only)
{
    auto grid = DEFAULT_GRID;
    auto f = func_for_part(u_hat);
    auto x0 = find_all(f, grid);
    for (std::size_t i = 0; i < x0.size(); i++) {
        x0[i] = 2 * x0[i] + u_hat.zeta();
    }

    if (!positive_only) {
        symmetrize_matsubara_inplace(x0);
    }

    std::vector<MatsubaraFreq<S>> result;
    for (auto x : x0) {
        result.push_back(MatsubaraFreq<S>(x));
    }
    return result;
}

template <typename S>
std::vector<MatsubaraFreq<S>> find_extrema(const PiecewiseLegendreFT<S> &u_hat,
                                           bool positive_only)
{
    auto f = func_for_part(u_hat);
    auto x0 = discrete_extrema(f, DEFAULT_GRID);
    for (auto &x : x0) {
        x = 2 * x + u_hat.zeta();
    }
    if (!positive_only) {
        symmetrize_matsubara_inplace(x0);
    }
    std::vector<MatsubaraFreq<S>> results;
    for (auto x : x0) {
        results.push_back(MatsubaraFreq<S>(x));
    }
    return results;
}

} // namespace sparseir