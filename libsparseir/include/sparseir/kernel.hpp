#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include <type_traits>
#include <iostream>

#include <Eigen/Dense>

#include "xprec/ddouble-header-only.hpp"
#include "sparseir/sparseir-fwd.hpp"

#include "sparseir/gauss.hpp"
#include "sparseir/freq.hpp"

namespace sparseir {

/**
 * @brief Abstract base class for an integral kernel K(x, y).
 *
 * AbstractKernel represents a real binary function K(x, y) used in a Fredholm
 * integral equation of the first kind:
 *
 *     u(x) = ∫ K(x, y) v(y) dy
 *
 * where x ∈ [xmin, xmax] and y ∈ [ymin, ymax].
 *
 * In general, the kernel is applied to a scaled spectral function ρ'(y) as:
 *
 *     ∫ K(x, y) ρ'(y) dy,
 *
 * where ρ'(y) = w(y) ρ(y).
 */
class AbstractKernel {
public:
    double lambda_;
    // Constructor
    AbstractKernel() { }
    AbstractKernel(double lambda) : lambda_(lambda) { }

    virtual std::vector<std::shared_ptr<AbstractKernel>>
    get_derived_kernels() const
    {
        return std::vector<std::shared_ptr<AbstractKernel>>();
    }

    /**
     * @brief Return the weight function for given statistics.
     *
     * @param statistics 'F' for fermions or 'B' for bosons.
     */
    template <typename T>
    std::function<T(T, T)> weight_func(Fermionic) const {
        return [](T beta, T omega) { (void)beta; (void)omega; return 1.0; };
    }

    template <typename T>
    std::function<T(T, T)> weight_func(Bosonic) const {
        return [](T beta, T omega) { (void)beta; (void)omega; return 1.0; };
    }

    virtual double compute(
        double x, double y,
        double x_plus = std::numeric_limits<double>::quiet_NaN(),
        double x_minus = std::numeric_limits<double>::quiet_NaN()) const = 0;

    virtual xprec::DDouble
    compute(xprec::DDouble x, xprec::DDouble y,
            xprec::DDouble x_plus = std::numeric_limits<double>::quiet_NaN(),
            xprec::DDouble x_minus =
                std::numeric_limits<double>::quiet_NaN()) const = 0;

    /**
     * @brief Return tuple (xmin, xmax) delimiting the range of allowed x
     * values.
     *
     * @return A pair containing xmin and xmax.
     */
    virtual std::pair<double, double> xrange() const
    {
        return std::make_pair(-1.0, 1.0);
    }

    /**
     * @brief Return tuple (ymin, ymax) delimiting the range of allowed y
     * values.
     *
     * @return A pair containing ymin and ymax.
     */
    virtual std::pair<double, double> yrange() const
    {
        return std::make_pair(-1.0, 1.0);
    }

    /**
     * @brief Check if the kernel is centrosymmetric.
     *
     * Returns true if and only if K(x, y) == K(-x, -y) for all values of x and
     * y. This allows the kernel to be block-diagonalized, speeding up the
     * singular value expansion by a factor of 4. Defaults to false.
     *
     * @return True if the kernel is centrosymmetric, false otherwise.
     */
    virtual bool is_centrosymmetric() const = 0;

    /**
     * @brief Power with which the y coordinate scales.
     *
     * @return The power with which y scales.
     */
    virtual int ypower() const { return 0; }

    /**
     * @brief Convergence radius of the Matsubara basis asymptotic model.
     *
     * For improved relative numerical accuracy, the IR basis functions on the
     * Matsubara axis can be evaluated from an asymptotic expression for
     * abs(n) > conv_radius. If conv_radius is infinity, then the asymptotics
     * are unused (the default).
     *
     * @return The convergence radius.
     */
    virtual double conv_radius() const
    {
        return std::numeric_limits<double>::infinity();
    }

    virtual ~AbstractKernel() = default;
};


// Type trait to check if a type is a concrete kernel
template <typename T>
struct is_concrete_kernel {
    static constexpr bool value =
        !std::is_abstract<T>::value &&
        std::is_base_of<AbstractKernel, T>::value;
};


/**
 * @brief Non-templated base class for all reduced kernels.
 * This allows us to use dynamic_pointer_cast with all reduced kernel types.
 */
class AbstractReducedKernelBase : public AbstractKernel {
public:
    AbstractReducedKernelBase(double lambda) : AbstractKernel(lambda) { }
    virtual ~AbstractReducedKernelBase() { }

    /**
     * @brief Get the inner kernel.
     *
     * @return A shared pointer to the inner kernel.
     */
    virtual std::shared_ptr<const AbstractKernel> get_inner_kernel() const = 0;
};

template <typename InnerKernel>
class AbstractReducedKernel : public AbstractReducedKernelBase {
public:
    int sign;
    const InnerKernel inner;

    // Constructor
    AbstractReducedKernel(const InnerKernel &inner_kernel, int sign)
        : AbstractReducedKernelBase(inner_kernel.lambda_),
          sign(sign),
          inner(inner_kernel)
    {
        // Validate inputs
        if (!inner.is_centrosymmetric()) {
            throw std::invalid_argument("Inner kernel must be centrosymmetric");
        }
        if (sign != 1 && sign != -1) {
            throw std::invalid_argument("sign must be -1 or 1");
        }
    }

    virtual double
    compute(double x, double y,
            double x_plus = std::numeric_limits<double>::quiet_NaN(),
            double x_minus = std::numeric_limits<double>::quiet_NaN()) const
    {
        return callreduced(*this, x, y, x_plus, x_minus);
    }

    virtual xprec::DDouble compute(
        xprec::DDouble x, xprec::DDouble y,
        xprec::DDouble x_plus = std::numeric_limits<double>::quiet_NaN(),
        xprec::DDouble x_minus = std::numeric_limits<double>::quiet_NaN()) const
    {
        return callreduced(*this, x, y, x_plus, x_minus);
    }
};

template <typename T, typename K>
T callreduced(const K &kernel, T x, T y, T x_plus, T x_minus)
{
    x_plus += 1;
    auto K_plus = kernel.inner.compute(x, +y, x_plus, x_minus);
    auto K_minus = kernel.inner.compute(x, -y, x_plus, x_minus);
    return K_plus + kernel.sign * K_minus;
}

/**
 * @brief Fermionic/bosonic analytical continuation kernel.
 *
 * In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is a
 * function on [-1, 1] × [-1, 1]:
 *
 *     K(x, y) = exp(-Λ y (x + 1) / 2) / (1 + exp(-Λ y))
 *
 * LogisticKernel is a fermionic analytic continuation kernel.
 * Nevertheless, one can model the τ dependence of a bosonic correlation
 * function as follows:
 *
 *     ∫ [exp(-Λ y (x + 1) / 2) / (1 - exp(-Λ y))] ρ(y) dy = ∫ K(x, y) ρ'(y) dy,
 *
 * with ρ'(y) = w(y) ρ(y), where the weight function is given by w(y) = 1 /
 * tanh(Λ y / 2).
 */
class LogisticKernel : public AbstractKernel {
public:
    // Default constructor
    LogisticKernel() : AbstractKernel() { }

    /**
     * @brief Constructor for LogisticKernel.
     *
     * @param lambda The kernel cutoff Λ.
     */
    LogisticKernel(double lambda) : AbstractKernel(lambda)
    {
        if (lambda < 0) {
            throw std::domain_error("Kernel cutoff Λ must be non-negative");
        }
    }

    std::vector<std::shared_ptr<AbstractKernel>>
    get_derived_kernels() const override
    {
        std::vector<std::shared_ptr<AbstractKernel>> kernels;
        kernels.push_back(std::make_shared<LogisticKernel>(*this));
        return kernels;
    }

    /**
     * @brief Evaluate the kernel at point (x, y).
     *
     * @param x The x coordinate.
     * @param y The y coordinate.
     * @return The kernel value at (x, y).
     */
    double compute(double x, double y,
                   double x_plus = std::numeric_limits<double>::quiet_NaN(),
                   double x_minus =
                       std::numeric_limits<double>::quiet_NaN()) const override
    {
        return _compute_impl(x, y, x_plus, x_minus);
    }

    xprec::DDouble
    compute(xprec::DDouble x, xprec::DDouble y,
            xprec::DDouble x_plus = std::numeric_limits<double>::quiet_NaN(),
            xprec::DDouble x_minus =
                std::numeric_limits<double>::quiet_NaN()) const override
    {
        return _compute_impl(x, y, x_plus, x_minus);
    }

    template <typename T>
    T _compute_impl(T x, T y,
                    T x_plus = std::numeric_limits<double>::quiet_NaN(),
                    T x_minus = std::numeric_limits<double>::quiet_NaN()) const
    {
        // Check that x and y are within the valid ranges
        std::pair<double, double> x_range = this->xrange();
        double xmin = x_range.first;
        double xmax = x_range.second;
        if (x < xmin || x > xmax) {
            throw std::out_of_range("x value not in range [-1, 1]");
        }
        std::pair<double, double> y_range = this->yrange();
        double ymin = y_range.first;
        double ymax = y_range.second;
        if (y < ymin || y > ymax) {
            throw std::out_of_range("y value not in range [-1, 1]");
        }

        std::tuple<T, T, T> uv_values = compute_uv(x, y, x_plus, x_minus);
        T u_plus = std::get<0>(uv_values);
        T u_minus = std::get<1>(uv_values);
        T v = std::get<2>(uv_values);

        return compute_from_uv(u_plus, u_minus, v);
    }

    // Inside class LogisticKernel definition
    template <typename T>
    std::shared_ptr<SVEHintsLogistic<T>> sve_hints(double epsilon) const
    {
        return std::make_shared<SVEHintsLogistic<T>>(*this, epsilon);
    }

    /**
     * @brief Check if the kernel is centrosymmetric.
     *
     * @return True, since LogisticKernel is centrosymmetric.
     */
    bool is_centrosymmetric() const override { return true; }

    /**
     * @brief Convergence radius of the Matsubara basis asymptotic model.
     *
     * For LogisticKernel, conv_radius = 40 * Λ.
     *
     * @return The convergence radius.
     */
    double conv_radius() const override { return 40.0 * this->lambda_; }

    template <typename T>
    std::function<T(T, T)> weight_func(Fermionic) const
    {
        return [](T beta , T omega) { (void)beta; (void)omega; return 1.0; };
    }

    template <typename T>
    std::function<T(T, T)> weight_func(Bosonic) const
    {
        using std::tanh;
        return [](T beta, T omega) { return 1.0 / tanh(0.5 * beta * omega); };
    }

private:
    /**
     * @brief Compute the variables u_plus, u_minus, v.
     *
     * @param x The x value.
     * @param y The y value.
     * @param x_plus x - xmin.
     * @param x_minus xmax - x.
     * @return A tuple containing u_plus, u_minus, and v.
     */
    template <typename T>
    std::tuple<T, T, T> compute_uv(T x, T y, T x_plus, T x_minus) const
    {
        using std::isnan;
        // Compute u_plus, u_minus, v
        if (isnan(x_plus)) {
            x_plus = 1.0 + x;
        }
        if (isnan(x_minus)) {
            x_minus = 1.0 - x;
        }
        T u_plus = 0.5 * x_plus;
        T u_minus = 0.5 * x_minus;
        T v = this->lambda_ * y;
        return std::make_tuple(u_plus, u_minus, v);
    }

    /**
     * @brief Compute the kernel value using u_plus, u_minus, and v.
     *
     * @param u_plus Computed u_plus.
     * @param u_minus Computed u_minus.
     * @param v Computed v.
     * @return The value of K(x, y).
     */
    template <typename T>
    T compute_from_uv(T u_plus, T u_minus, T v) const
    {
        using std::abs;

        T mabs_v = -abs(v);

        T numerator;
        T denominator;

        if (v >= 0) {
            numerator = exp_impl(u_plus * mabs_v);
        } else {
            numerator = exp_impl(u_minus * mabs_v);
        }

        denominator = 1.0 + exp_impl(mabs_v);

        return numerator / denominator;
    }
};

/**
 * @brief Regularized bosonic analytical continuation kernel.
 *
 * In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is a
 * function on [-1, 1] × [-1, 1]:
 *
 *     K(x, y) = y * exp(-Λ y (x + 1) / 2) / (1 - exp(-Λ y))
 *
 * This non-dimensionalized kernel is connected to the dimensionalized kernel as
 *     K(τ, ω) = ωmax * K(2τ/β - 1, ω/ωmax),
 * where ωmax = Λ/β.
 *
 * Care has to be taken in evaluating this expression around y = 0.
 */
class RegularizedBoseKernel : public AbstractKernel {
public:
    // Default constructor
    RegularizedBoseKernel() : AbstractKernel() { }

    /**
     * @brief Constructor for RegularizedBoseKernel.
     *
     * @param lambda The kernel cutoff Λ.
     */
    RegularizedBoseKernel(double lambda) : AbstractKernel(lambda)
    {
        if (lambda < 0) {
            throw std::domain_error("Kernel cutoff Λ must be non-negative");
        }
    }

    std::vector<std::shared_ptr<AbstractKernel>>
    get_derived_kernels() const override
    {
        std::vector<std::shared_ptr<AbstractKernel>> kernels;
        kernels.push_back(std::make_shared<RegularizedBoseKernel>(*this));
        return kernels;
    }

    /**
     * @brief Evaluate the kernel at point (x, y).
     *
     * @param x The x value.
     * @param y The y value.
     * @param x_plus Optional. x - xmin.
     * @param x_minus Optional. xmax - x.
     * @return The value of K(x, y).
     */
    double compute(double x, double y,
                   double x_plus = std::numeric_limits<double>::quiet_NaN(),
                   double x_minus =
                       std::numeric_limits<double>::quiet_NaN()) const override
    {
        return _compute_impl(x, y, x_plus, x_minus);
    }

    xprec::DDouble
    compute(xprec::DDouble x, xprec::DDouble y,
            xprec::DDouble x_plus = std::numeric_limits<double>::quiet_NaN(),
            xprec::DDouble x_minus =
                std::numeric_limits<double>::quiet_NaN()) const override
    {
        return _compute_impl(x, y, x_plus, x_minus);
    }

    template <typename T>
    T _compute_impl(T x, T y,
                    T x_plus = std::numeric_limits<double>::quiet_NaN(),
                    T x_minus = std::numeric_limits<double>::quiet_NaN()) const
    {
        // Check that x and y are within the valid ranges
        std::pair<double, double> xrange_values = this->xrange();
        double xmin = xrange_values.first;
        double xmax = xrange_values.second;
        if (x < xmin || x > xmax) {
            throw std::out_of_range("x value not in range [-1, 1]");
        }
        std::pair<double, double> yrange_values = this->yrange();
        double ymin = yrange_values.first;
        double ymax = yrange_values.second;
        if (y < ymin || y > ymax) {
            throw std::out_of_range("y value not in range [-1, 1]");
        }

        std::tuple<T, T, T> uv_values = compute_uv(x, y, x_plus, x_minus);
        T u_plus = std::get<0>(uv_values);
        T u_minus = std::get<1>(uv_values);
        T v = std::get<2>(uv_values);

        return compute_from_uv(u_plus, u_minus, v);
    }

    /**
     * @brief Check if the kernel is centrosymmetric.
     *
     * @return True, since RegularizedBoseKernel is centrosymmetric.
     */
    bool is_centrosymmetric() const override { return true; }

    /**
     * @brief Returns the power with which y scales.
     *
     * For RegularizedBoseKernel, ypower = 1.
     *
     * @return The power with which y scales.
     */
    int ypower() const override { return 1; }

    /**
     * @brief Convergence radius of the Matsubara basis asymptotic model.
     *
     * For RegularizedBoseKernel, conv_radius = 40 * Λ.
     *
     * @return The convergence radius.
     */
    double conv_radius() const override { return 40.0 * this->lambda_; }

    template <typename T>
    std::function<T(T, T)> weight_func(Fermionic) const
    {
        std::cerr
            << "RegularizedBoseKernel does not support fermionic functions"
            << std::endl;
        throw std::invalid_argument(
            "RegularizedBoseKernel does not support fermionic functions");
    }

    template <typename T>
    std::function<T(T, T)> weight_func(Bosonic) const
    {
        using std::tanh;
        return [](T beta, T omega) {
            (void)beta;  // Silence unused parameter warning
            return static_cast<T>(1.0) / omega;
        };
    }

    template <typename T>
    std::shared_ptr<SVEHintsRegularizedBose<T>> sve_hints(double epsilon) const
    {
        return std::make_shared<SVEHintsRegularizedBose<T>>(*this, epsilon);
    }

private:
    /**
     * @brief Compute the variables u_plus, u_minus, v.
     *
     * @param x The x value.
     * @param y The y value.
     * @param x_plus x - xmin.
     * @param x_minus xmax - x.
     * @return A tuple containing u_plus, u_minus, and v.
     */
    template <typename T>
    std::tuple<T, T, T> compute_uv(T x, T y, T x_plus, T x_minus) const
    {
        using std::isnan;
        // Compute u_plus, u_minus, v
        if (isnan(x_plus)) {
            x_plus = 1.0 + x;
        }
        if (isnan(x_minus)) {
            x_minus = 1.0 - x;
        }
        T u_plus = 0.5 * x_plus;
        T u_minus = 0.5 * x_minus;
        T v = this->lambda_ * y;
        return std::make_tuple(u_plus, u_minus, v);
    }

    /**
     * @brief Compute the kernel value using u_plus, u_minus, and v.
     *
     * @param u_plus Computed u_plus.
     * @param u_minus Computed u_minus.
     * @param v Computed v.
     * @return The value of K(x, y).
     */

    template <typename T>
    T compute_from_uv(T u_plus, T u_minus, T v) const
    {
        using std::abs;
        using std::exp;
        using std::expm1;

        T absv = abs(v);
        T enum_val = exp(-absv * (v >= 0 ? u_plus : u_minus));

        // Handle the tricky expression v / (exp(v) - 1)
        T denom;
        if (absv >= 1e-200) {
            denom = absv / expm1(-absv);
        } else {
            denom = -1; // Assuming T is a floating-point type
        }

        return -1 / static_cast<T>(this->lambda_) * enum_val * denom;
    }
};

/**
 * @brief Restriction of centrosymmetric kernel to positive interval.
 *
 * For a kernel K on [-1, 1] × [-1, 1] that is centrosymmetric, i.e.,
 * K(x, y) = K(-x, -y), it is straightforward to show that the left/right
 * singular vectors can be chosen as either odd or even functions.
 *
 * Consequently, they are singular functions of a reduced kernel K_red on
 * [0, 1] × [0, 1] that is given as either:
 *
 *     K_red(x, y) = K(x, y) ± K(x, -y)
 *
 * This kernel is what this class represents. The full singular functions can be
 * reconstructed by (anti-)symmetrically continuing them to the negative axis.
 */
template <typename K>
class ReducedKernel : public AbstractReducedKernel<K> {
public:
    K inner_kernel_; ///< The inner kernel K.
    int sign_;       ///< The sign (+1 or -1).

    /**
     * @brief Constructor for ReducedKernel.
     *
     * @param inner_kernel The inner kernel K.
     * @param sign The sign (+1 or -1). Must satisfy abs(sign) == 1.
     */
    ReducedKernel(const K &inner_kernel, int sign)
        : AbstractReducedKernel<K>(inner_kernel, sign), // Initialize base class
          inner_kernel_(inner_kernel),
          sign_(sign)
    {
        if (!inner_kernel_.is_centrosymmetric()) {
            throw std::invalid_argument("Inner kernel must be centrosymmetric");
        }
        if (sign != 1 && sign != -1) {
            throw std::invalid_argument("sign must be -1 or 1");
        }
    }

    /**
     * @brief Get the inner kernel.
     *
     * @return A shared pointer to the inner kernel.
     */
    std::shared_ptr<const AbstractKernel> get_inner_kernel() const override
    {
        return std::make_shared<K>(inner_kernel_);
    }

    /**
     * @brief Evaluate the reduced kernel at point (x, y).
     *
     * @param x The x value.
     * @param y The y value.
     * @param x_plus Optional. x - xmin.
     * @param x_minus Optional. xmax - x.
     * @return The value of K_red(x, y).
     */
    template <typename T>
    T compute(T x, T y, T x_plus = std::numeric_limits<double>::quiet_NaN(),
              T x_minus = std::numeric_limits<double>::quiet_NaN()) const
    {
        return callreduced(*this, x, y, x_plus, x_minus);
    }

    /**
     * @brief Return tuple (xmin, xmax) delimiting the range of allowed x
     * values.
     *
     * For ReducedKernel, xrange is modified to [0, xmax_inner].
     *
     * @return A pair containing xmin and xmax.
     */
    std::pair<double, double> xrange() const override
    {
        auto range = inner_kernel_.xrange();
        return std::make_pair(0.0, range.second);
    }

    /**
     * @brief Return tuple (ymin, ymax) delimiting the range of allowed y
     * values.
     *
     * For ReducedKernel, yrange is modified to [0, ymax_inner].
     *
     * @return A pair containing ymin and ymax.
     */
    std::pair<double, double> yrange() const override
    {
        auto range = inner_kernel_.yrange();
        return std::make_pair(0.0, range.second);
    }

    /**
     * @brief Check if the kernel is centrosymmetric.
     *
     * @return False, since ReducedKernel cannot be symmetrized further.
     */
    bool is_centrosymmetric() const override { return false; }

    /**
     * @brief Returns the power with which y scales.
     *
     * @return The ypower of the inner kernel.
     */
    int ypower() const override { return inner_kernel_.ypower(); }

    /**
     * @brief Convergence radius of the Matsubara basis asymptotic model.
     *
     * @return The convergence radius of the inner kernel.
     */
    double conv_radius() const override { return inner_kernel_.conv_radius(); }

    template <typename T>
    std::shared_ptr<SVEHintsReduced<T>> sve_hints(double epsilon) const
    {
        return std::make_shared<SVEHintsReduced<T>>(
            inner_kernel_.template sve_hints<T>(epsilon));
    }

private:
};

/*
    AbstractSVEHints

Discretization hints for singular value expansion of a given kernel.
*/

template <typename T>
class AbstractSVEHints {
public:
    virtual ~AbstractSVEHints() = default;

    virtual std::vector<T> segments_x() const = 0;
    virtual std::vector<T> segments_y() const = 0;

    // Additional methods if needed
    virtual int nsvals() const = 0;
    virtual int ngauss() const = 0;
};

template <typename T>
class SVEHintsLogistic final : public AbstractSVEHints<T> {
public:
    SVEHintsLogistic(const LogisticKernel &kernel, double epsilon)
        : kernel_(kernel), epsilon_(epsilon)
    {
    }

    std::vector<T> segments_x() const override
    {
        int nzeros = std::max(
            static_cast<int>(std::round(15 * std::log10(kernel_.lambda_))), 1);

        // Create a range of values
        std::vector<T> temp(nzeros);
        for (int i = 0; i < nzeros; ++i) {
            temp[i] = (T)0.143 * i;
        }

        // Calculate diffs using the inverse hyperbolic cosine
        std::vector<T> diffs(nzeros);
        for (int i = 0; i < nzeros; ++i) {
            diffs[i] = 1.0 / cosh_impl(temp[i]);
        }

        // Calculate cumulative sum of diffs
        std::vector<T> zeros(nzeros);
        zeros[0] = diffs[0];
        for (int i = 1; i < nzeros; ++i) {
            zeros[i] = zeros[i - 1] + diffs[i];
        }

        // Normalize zeros
        T last_zero = zeros.back();
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] /= last_zero;
        }

        // Create the final segments vector
        std::vector<T> segments(2 * nzeros + 1, 0);
        for (int i = 0; i < nzeros; ++i) {
            segments[i] = -zeros[nzeros - i - 1];
            segments[nzeros + i + 1] = zeros[i];
        }
        return segments;
    };

    std::vector<T> segments_y() const override
    {
        // Calculate the number of zeros
        int nzeros = std::max(
            static_cast<int>(std::round(20 * std::log10(kernel_.lambda_))), 2);

        // Initial differences
        std::vector<T> diffs = {0.01523, 0.03314, 0.04848, 0.05987, 0.06703,
                                0.07028, 0.07030, 0.06791, 0.06391, 0.05896,
                                0.05358, 0.04814, 0.04288, 0.03795, 0.03342,
                                0.02932, 0.02565, 0.02239, 0.01951, 0.01699};

        // Truncate diffs if necessary
        if (nzeros < static_cast<int>(diffs.size())) {
            diffs.resize(nzeros);
        }

        // Calculate trailing differences
        for (int i = 20; i < nzeros; ++i) {
            T x = (T)0.141 * i;
            diffs.push_back(0.25 * exp_impl(-x));
        }

        // Calculate cumulative sum of diffs
        std::vector<T> zeros(nzeros);
        zeros[0] = diffs[0];
        for (int i = 1; i < nzeros; ++i) {
            zeros[i] = zeros[i - 1] + diffs[i];
        }

        // Normalize zeros
        T last_zero = zeros.back();
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] /= last_zero;
        }
        zeros.pop_back();

        // updated nzeros
        nzeros = zeros.size();

        // Adjust zeros
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] -= 1.0;
        }

        // Create the final segments vector
        std::vector<T> segments(2 * nzeros + 3, 0);
        for (int i = 0; i < nzeros; ++i) {
            segments[1 + i] = zeros[i];
            segments[1 + nzeros + 1 + i] = -zeros[nzeros - i - 1];
        }
        segments[0] = -T(1.0);
        segments[1 + nzeros] = T(0.0);
        segments[2 * nzeros + 2] = T(1.0);
        return segments;
    }

    int nsvals() const override
    {
        double log10_Lambda = std::max(1.0, std::log10(kernel_.lambda_));
        return static_cast<int>(std::round((25 + log10_Lambda) * log10_Lambda));
    }
    int ngauss() const override
    {
        return epsilon_ >= std::sqrt(std::numeric_limits<double>::epsilon())
                   ? 10
                   : 16;
    };

private:
    LogisticKernel kernel_;
    double epsilon_;
};

template <typename T>
class SVEHintsRegularizedBose : public AbstractSVEHints<T> {
public:
    SVEHintsRegularizedBose(const RegularizedBoseKernel &kernel, double epsilon)
        : kernel_(kernel), epsilon_(epsilon)
    {
    }

    std::vector<T> segments_x() const override
    {
        using std::cosh;

        int nzeros = std::max(
            static_cast<int>(std::round(15 * std::log10(kernel_.lambda_))), 15);
        std::vector<T> temp(nzeros);
        std::vector<T> diffs(nzeros);
        std::vector<T> zeros(nzeros);

        for (int i = 0; i < nzeros; ++i) {
            temp[i] = T(0.18) * i;
            diffs[i] = 1.0 / cosh(temp[i]);
        }

        std::partial_sum(diffs.begin(), diffs.end(), zeros.begin());
        T last_zero = zeros.back();
        std::transform(zeros.begin(), zeros.end(), zeros.begin(),
                       [last_zero](T z) { return z / last_zero; });

        std::vector<T> result(2 * nzeros + 1, T(0));

        for (int i = 0; i < nzeros; ++i) {
            result[i] = -zeros[nzeros - i - 1];
            result[nzeros + i + 1] = zeros[i];
        }

        return result;
    }

    std::vector<T> segments_y() const override
    {
        int nzeros = std::max(
            static_cast<int>(std::round(20 * std::log10(kernel_.lambda_))), 20);
        std::vector<T> diffs(nzeros);

        for (int j = 0; j < nzeros; ++j) {
            diffs[j] = T(0.12) / std::exp(0.0337 * j * std::log(j + 1));
        }

        // Calculate cumulative sum of diffs
        std::vector<T> zeros(nzeros);
        zeros[0] = diffs[0];
        for (int i = 1; i < nzeros; ++i) {
            zeros[i] = zeros[i - 1] + diffs[i];
        }

        // Normalize zeros
        T last_zero = zeros.back();
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] /= last_zero;
        }
        zeros.pop_back();

        // updated nzeros
        nzeros = zeros.size();

        // Adjust zeros
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] -= 1.0;
        }

        std::vector<T> result(2 * nzeros + 3, T(0));

        for (int i = 0; i < nzeros; ++i) {
            result[1 + i] = zeros[i];
            result[1 + nzeros + 1 + i] = -zeros[nzeros - i - 1];
        }
        result[0] = T(-1);
        result[1 + nzeros] = T(0);
        result[2 * nzeros + 2] = T(1);
        return result;
    }

    int nsvals() const override
    {
        double log10_Lambda = std::max(1.0, std::log10(kernel_.lambda_));
        return static_cast<int>(std::round(28 * log10_Lambda));
    }
    int ngauss() const override
    {
        return epsilon_ >= std::sqrt(std::numeric_limits<double>::epsilon())
                   ? 10
                   : 16;
    };

private:
    RegularizedBoseKernel kernel_;
    double epsilon_;
};

class RegularizedBoseKernelOdd
    : public AbstractReducedKernel<RegularizedBoseKernel> {
public:
    RegularizedBoseKernelOdd(const RegularizedBoseKernel &inner, int sign)
        : AbstractReducedKernel<RegularizedBoseKernel>(inner, sign),
          inner_kernel_(inner)
    {
        using std::abs;
        if (!inner.is_centrosymmetric()) {
            throw std::runtime_error("inner kernel must be centrosymmetric");
        }
        if (abs(sign) != 1) {
            throw std::domain_error("sign must be -1 or 1");
        }
    }

    virtual bool is_centrosymmetric() const override {
        return false;
    }

    /**
     * @brief Get the inner kernel.
     *
     * @return A shared pointer to the inner kernel.
     */
    std::shared_ptr<const AbstractKernel> get_inner_kernel() const override
    {
        return std::make_shared<RegularizedBoseKernel>(inner_kernel_);
    }

    template <typename T>
    T compute(T x, T y, T x_plus = std::numeric_limits<double>::quiet_NaN(),
              T x_minus = std::numeric_limits<double>::quiet_NaN()) const
    {
        T v_half = inner_kernel_.lambda_ * 0.5 * y;
        T xv_half = x * v_half;
        bool xy_small = xv_half < 1;
        bool sinh_range = 1e-200 < v_half && v_half < 85;
        if (xy_small && sinh_range) {
            return y * sinh_impl(xv_half) / sinh_impl(v_half);
        } else {
            return callreduced(*this, x, y, x_plus, x_minus);
        }
    }

    template <typename T>
    std::shared_ptr<SVEHintsReduced<T>> sve_hints(double epsilon) const
    {
        return std::make_shared<SVEHintsReduced<T>>(
            inner_kernel_.template sve_hints<T>(epsilon));
    }

private:
    RegularizedBoseKernel inner_kernel_;
};

class LogisticKernelOdd : public AbstractReducedKernel<LogisticKernel> {
public:
    LogisticKernelOdd(const LogisticKernel &inner, int sign)
        : AbstractReducedKernel<LogisticKernel>(inner, sign),
          inner_kernel_(inner)
    {
        if (sign != -1) {
            throw std::invalid_argument("sign must be -1");
        }
    }

    virtual bool is_centrosymmetric() const override {
        return false;
    }

    /**
     * @brief Get the inner kernel.
     *
     * @return A shared pointer to the inner kernel.
     */
    std::shared_ptr<const AbstractKernel> get_inner_kernel() const override
    {
        return std::make_shared<LogisticKernel>(inner_kernel_);
    }

    // Implement the compute method
    template <typename T>
    T compute(T x, T y, T x_plus = std::numeric_limits<double>::quiet_NaN(),
              T x_minus = std::numeric_limits<double>::quiet_NaN()) const
    {
        using std::cosh;
        using std::sinh;
        T v_half = this->inner.lambda_ * 0.5 * y;
        bool xy_small = x * v_half < 1;
        bool cosh_finite = v_half < 85;
        if (xy_small && cosh_finite) {
            return -sinh(v_half * x) / cosh(v_half);
        } else {
            return callreduced(*this, x, y, x_plus, x_minus);
        }
    }

    // Implement the pure virtual function from the parent class
    template <typename T>
    std::shared_ptr<SVEHintsReduced<T>> sve_hints(double epsilon) const
    {
        return std::make_shared<SVEHintsReduced<T>>(
            inner_kernel_.sve_hints<T>(epsilon));
    }

private:
    LogisticKernel inner_kernel_;
};

// Traits class
template <typename Kernel, typename Sign>
struct SymmKernelTraits;

template <>
struct SymmKernelTraits<LogisticKernel, std::integral_constant<int, -1>>
{
    using type = LogisticKernelOdd;
};

template <>
struct SymmKernelTraits<LogisticKernel, std::integral_constant<int, 1>>
{
    using type = ReducedKernel<LogisticKernel>;
};

// Add the missing specializations for RegularizedBoseKernel
template <>
struct SymmKernelTraits<RegularizedBoseKernel, std::integral_constant<int, -1>>
{
    using type = RegularizedBoseKernelOdd;
};

template <>
struct SymmKernelTraits<RegularizedBoseKernel, std::integral_constant<int, 1>>
{
    using type = ReducedKernel<RegularizedBoseKernel>;
};

template <typename K, typename Sign>
typename SymmKernelTraits<K, Sign>::type get_symmetrized(const K &kernel, Sign);

template <>
inline ReducedKernel<LogisticKernel>
get_symmetrized(const LogisticKernel &kernel, std::integral_constant<int, +1>)
{
    return ReducedKernel<LogisticKernel>(kernel, 1);
}

template <>
inline LogisticKernelOdd get_symmetrized(const LogisticKernel &kernel,
                                         std::integral_constant<int, -1>)
{
    return LogisticKernelOdd(kernel, -1);
}

template <>
inline RegularizedBoseKernelOdd
get_symmetrized(const RegularizedBoseKernel &kernel,
                std::integral_constant<int, -1>)
{
    return RegularizedBoseKernelOdd(kernel, -1);
}

template <>
inline ReducedKernel<RegularizedBoseKernel>
get_symmetrized(const RegularizedBoseKernel &kernel,
                std::integral_constant<int, +1>)
{
    return ReducedKernel<RegularizedBoseKernel>(kernel, 1);
}

template <typename T>
Eigen::MatrixX<T> matrix_from_gauss(const AbstractKernel &kernel,
                                    const Rule<T> &gauss_x,
                                    const Rule<T> &gauss_y);

// Function to validate symmetry and extract the right-hand side of the segments
template <typename T>
std::vector<T> symm_segments(const std::vector<T> &x);

template <typename T>
class SVEHintsReduced : public AbstractSVEHints<T> {
public:
    SVEHintsReduced(std::shared_ptr<AbstractSVEHints<T>> inner_hints)
        : inner(inner_hints)
    {
    }

    // Implement required methods
    int nsvals() const override
    {
        // Implement this function
        // For example, you can delegate the call to the inner object
        return (inner->nsvals() + 1) / 2;
    }

    int ngauss() const override
    {
        // Implement this function
        // For example, you can delegate the call to the inner object
        return inner->ngauss();
    }

    std::vector<T> segments_x() const override
    {
        return symm_segments(inner->segments_x());
    }

    std::vector<T> segments_y() const override
    {
        return symm_segments(inner->segments_y());
    }

private:
    std::shared_ptr<AbstractSVEHints<T>> inner;
};

// Function to provide SVE hints
template <typename T>
std::shared_ptr<AbstractSVEHints<T>>
sve_hints(const std::shared_ptr<const AbstractKernel> &kernel, double epsilon);

} // namespace sparseir