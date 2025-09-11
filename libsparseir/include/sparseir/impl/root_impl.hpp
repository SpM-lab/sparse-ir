#pragma once

#include "sparseir/root.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

namespace sparseir {

// Midpoint function for floating-point types
template <typename T1, typename T2>
typename std::enable_if<
    std::is_floating_point<T1>::value && std::is_floating_point<T2>::value,
    typename std::common_type<T1, T2>::type>::type inline midpoint(T1 a, T2 b)
{
    typedef typename std::common_type<T1, T2>::type CommonType;
    return static_cast<CommonType>(a) +
           (static_cast<CommonType>(b) - static_cast<CommonType>(a)) *
               static_cast<CommonType>(0.5);
}

// Midpoint function for integral types
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type inline midpoint(
    T a, T b)
{
    return a + ((b - a) / 2);
}

// For floating point types, returns true if absolute difference is within
// epsilon. For integer types, performs exact equality comparison
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
closeenough(T a, T b, T eps)
{
    return std::abs(a - b) <= eps;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
closeenough(T a, T b, T /*eps*/)
{
    return a == b;
}

// Signbit function (handles both floating-point and integral types)
template <typename T>
inline bool signbit(T x)
{
    return x < static_cast<T>(0);
}

// Bisection method to find root in interval [a,b] where f(a) and f(b) have
// opposite signs f: Function (T -> double) a, b: Interval endpoints fa: Value
// of f(a) eps: Error tolerance (for floating point)
template <typename T, typename F>
T bisect(const F &f, T a, T b, T fa, T eps)
{
    while (true) {
        T mid = midpoint(a, b);
        if (closeenough(a, mid, eps)) {
            return mid;
        }
        double fmid = f(mid);
        // Check for sign change (std::signbit only works for floating point,
        // but comparison works for integers)
        if (std::signbit(fa) != std::signbit(fmid)) {
            b = mid;
        } else {
            a = mid;
            fa = fmid;
        }
    }
}

template <typename F, typename T>
std::vector<T> find_all(F f, const std::vector<T> &xgrid)
{
    if (xgrid.empty()) {
        return {};
    }
    std::vector<double> fx;
    std::transform(xgrid.begin(), xgrid.end(), std::back_inserter(fx),
                   [&](T x) { return static_cast<double>(f(x)); });

    std::vector<bool> hit(fx.size());
    std::transform(fx.begin(), fx.end(), hit.begin(),
                   [](double val) { return val == 0.0; });

    std::vector<T> x_hit;
    for (size_t i = 0; i < hit.size(); ++i) {
        if (hit[i])
            x_hit.push_back(xgrid[i]);
    }

    std::vector<bool> sign_change(fx.size() - 1);
    for (size_t i = 0; i < fx.size() - 1; ++i) {
        sign_change[i] = std::signbit(fx[i]) != std::signbit(fx[i + 1]);
    }

    for (size_t i = 0; i < sign_change.size(); ++i) {
        sign_change[i] = sign_change[i] && !hit[i] && !hit[i + 1];
    }

    if (std::none_of(sign_change.begin(), sign_change.end(),
                     [](bool v) { return v; })) {
        return x_hit;
    }

    std::vector<bool> where_a(sign_change.size() + 1, false);
    std::vector<bool> where_b(sign_change.size() + 1, false);
    for (size_t i = 0; i < sign_change.size(); ++i) {
        where_a[i] = sign_change[i];
        where_b[i + 1] = sign_change[i];
    }

    std::vector<T> a, b;
    std::vector<double> fa;
    for (size_t i = 0; i < where_a.size(); ++i) {
        if (where_a[i]) {
            a.push_back(xgrid[i]);
            fa.push_back(fx[i]);
        }
        if (where_b[i]) {
            b.push_back(xgrid[i]);
        }
    }

    T max_elm = std::abs(xgrid[0]);
    for (size_t i = 1; i < xgrid.size(); ++i) {
        max_elm = std::max(max_elm, std::abs(xgrid[i]));
    }
    T epsilon_x = std::numeric_limits<T>::epsilon() * max_elm;

    std::vector<T> x_bisect;
    for (size_t i = 0; i < a.size(); ++i) {
        double root = bisect(f, a[i], b[i], static_cast<T>(fa[i]), epsilon_x);
        x_bisect.push_back(static_cast<T>(root));
    }

    x_hit.insert(x_hit.end(), x_bisect.begin(), x_bisect.end());
    std::sort(x_hit.begin(), x_hit.end());
    return x_hit;
}

template <typename T>
std::vector<T> refine_grid(const std::vector<T> &grid, int alpha)
{
    size_t n = grid.size();
    size_t newn = alpha * (n - 1) + 1;
    std::vector<T> newgrid(newn);

    for (size_t i = 0; i < n - 1; ++i) {
        T xb = grid[i];
        T xe = grid[i + 1];
        T delta_x = (xe - xb) / alpha;
        size_t newi = alpha * i;
        for (int j = 0; j < alpha; ++j) {
            newgrid[newi + j] = xb + delta_x * j;
        }
    }
    newgrid.back() = grid.back();
    return newgrid;
}

template <typename F>
double bisect_discr_extremum(F absf, double a, double b, double absf_a,
                             double absf_b)
{
    double d = b - a;

    if (d <= 1)
        return absf_a > absf_b ? a : b;
    if (d == 2)
        return a + 1;

    double m = midpoint(a, b);
    double n = m + 1;
    double absf_m = absf(m);
    double absf_n = absf(n);

    if (absf_m > absf_n) {
        return bisect_discr_extremum(absf, a, n, absf_a, absf_n);
    } else {
        return bisect_discr_extremum(absf, m, b, absf_m, absf_b);
    }
}

template <typename F, typename T>
std::vector<T> discrete_extrema(F f, const std::vector<T> &xgrid)
{
    std::vector<double> fx(xgrid.size());
    for (size_t i = 0; i < xgrid.size(); ++i) {
        fx[i] = f(xgrid[i]);
    }

    std::vector<double> absfx(fx.size());
    for (size_t i = 0; i < fx.size(); ++i) {
        absfx[i] = std::abs(fx[i]);
    }

    std::vector<bool> signdfdx(fx.size() - 1);
    for (size_t i = 0; i < fx.size() - 1; ++i) {
        signdfdx[i] = std::signbit(fx[i + 1] - fx[i]);
    }

    std::vector<bool> derivativesignchange(signdfdx.size() - 1);
    for (size_t i = 0; i < signdfdx.size() - 1; ++i) {
        derivativesignchange[i] = signdfdx[i] != signdfdx[i + 1];
    }

    // create copy of derivativesignchange and add two false at the end
    std::vector<bool> derivativesignchange_a(derivativesignchange);
    derivativesignchange_a.push_back(false);
    derivativesignchange_a.push_back(false);

    std::vector<bool> derivativesignchange_b;
    derivativesignchange_b.reserve(derivativesignchange.size() + 2);
    derivativesignchange_b.push_back(false);
    derivativesignchange_b.push_back(false);
    derivativesignchange_b.insert(derivativesignchange_b.end(),
                                  derivativesignchange.begin(),
                                  derivativesignchange.end());

    std::vector<double> a, b;
    std::vector<double> absf_a, absf_b;
    for (size_t i = 0; i < derivativesignchange_a.size(); ++i) {
        if (derivativesignchange_a[i]) {
            a.push_back(xgrid[i]);
            absf_a.push_back(absfx[i]);
        }
        if (derivativesignchange_b[i]) {
            b.push_back(xgrid[i]);
            absf_b.push_back(absfx[i]);
        }
    }

    std::vector<T> res;
    for (size_t i = 0; i < a.size(); ++i) {
        // abs ∘ f
        auto abf = [f](double x) { return std::fabs(f(x)); };
        res.push_back(
            bisect_discr_extremum(abf, a[i], b[i], absf_a[i], absf_b[i]));
    }

    // We consider the outer points to be extrema if there is a decrease
    // in magnitude or a sign change inwards

    std::vector<bool> sfx(fx.size());
    for (size_t i = 0; i < fx.size(); ++i) {
        sfx[i] = std::signbit(fx[i]);
    }

    if (absfx.front() > absfx[1] || sfx.front() != sfx[1]) {
        res.insert(res.begin(), xgrid.front());
    }
    if (absfx.back() > absfx[absfx.size() - 2] ||
        sfx.back() != sfx[sfx.size() - 2]) {
        res.push_back(xgrid.back());
    }

    return res;
}

} // namespace sparseir