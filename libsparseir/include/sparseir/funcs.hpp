// Template class for FiniteTempBasis
#pragma once

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <memory> // for std::shared_ptr
#include <vector>
#include <set>

namespace sparseir {

// Helper function to check indices for duplicates and out of range
template<typename T>
void check_indices(const std::vector<T>& indices, size_t max_size) {
    std::set<T> unique_indices(indices.begin(), indices.end());
    if (unique_indices.size() != indices.size()) {
        throw std::runtime_error("Duplicate indices are not allowed");
    }
    for (T idx : indices) {
        if (static_cast<size_t>(idx) >= max_size) {
            throw std::runtime_error("Index out of range");
        }
    }
}


// Abstract class for periodic functions with fermionic/bosonic statistics
template <typename S>
class AbstractTauFunctions {
public:
    virtual ~AbstractTauFunctions() = default;
    virtual Eigen::MatrixXd operator()(const Eigen::VectorXd &xs) const = 0;
    virtual Eigen::VectorXd operator()(double x) const = 0;
    virtual double get_beta() const = 0;
    virtual size_t size() const = 0;
    virtual int nroots() const = 0;
    virtual Eigen::VectorXd roots() const = 0;
};


// Accept a tau in [-beta, beta] and return a tau in [0, beta] with a sign
// change for the fermionic case
//
// Interpreted as:
// -beta -> -beta^+
//  beta -> beta^-
//  +0.0 -> 0^+
//  -0.0 -> 0^-
inline std::pair<double, double> regularize_tau(double tau, double beta, int fermionic_sign)
{
    // Check if tau is in the valid range
    if (tau < -beta) {
        throw std::invalid_argument("tau is less than -beta");
    }
    if (tau > beta) {
        std::cout << "tau " << tau << " is greater than beta " << beta << std::endl;
        throw std::invalid_argument("tau is greater than beta");
    }

    if (0 < tau && tau <= beta) {
        return std::make_pair(tau, 1.0);
    } else if (-beta <= tau && tau < 0) {
        return std::make_pair(tau + beta, fermionic_sign);
    } else if (tau == 0) {
        // Check if tau is -0.0
        if (std::signbit(tau)) {
            return std::make_pair(beta, fermionic_sign);
        } else {
            return std::make_pair(0.0, 1.0);
        }
    }

    throw std::invalid_argument("Something is wrong with the input tau");
}


/**
 * @brief A class for handling periodic functions with fermionic/bosonic statistics.
 *
 * This class manages periodic functions that can be either fermionic or bosonic,
 * handling the periodicity and sign changes appropriately. It wraps an implementation
 * class and provides methods to evaluate the function at single or multiple points
 * in the range [-beta, beta] while respecting the periodic boundary conditions.
 * The treatment of tau = -beta, 0^+, 0^-, beta can be found in the regularize_tau function.
 * The underlying implementation class must support evaluations in the range [0, beta].
 *
 * @tparam S Statistics type (Fermionic or Bosonic)
 * @tparam ImplType Implementation type for the underlying functions
 */
template <typename S, typename ImplType>
class PeriodicFunctions : public AbstractTauFunctions<S> {
private:
    std::shared_ptr<const ImplType> impl;
    double beta;

public:
    PeriodicFunctions(std::shared_ptr<const ImplType> impl, double beta)
        : impl(impl), beta(beta)
    {
    }

    PeriodicFunctions(const ImplType &data, double beta)
        : impl(std::make_shared<const ImplType>(data)), beta(beta)
    {
    }

    Eigen::MatrixXd operator()(const Eigen::VectorXd &xs) const override {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        auto _sign = fermionic_sign<S>();

        Eigen::VectorXd xs_reg(xs.size());
        Eigen::VectorXd signs(xs.size());
        for (int i = 0; i < xs.size(); ++i) {
            std::pair<double, int> regularized_tau = regularize_tau(xs[i], beta, _sign);
            xs_reg[i] = regularized_tau.first;
            signs[i] = regularized_tau.second;
        }

        Eigen::MatrixXd result = impl->operator()(xs_reg);
        for (int i = 0; i < xs.size(); ++i) {
            result.col(i) *= signs[i];
        }

        return result;
    }

    Eigen::VectorXd operator()(double x) const override {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        auto _sign = fermionic_sign<S>();
        std::pair<double, int> regularized_tau = regularize_tau(x, beta, _sign);
        return impl->operator()(regularized_tau.first) * regularized_tau.second;
    }

    std::pair<double, double> get_domain() const
    {
        return std::make_pair(-beta, beta);
    }

    ImplType get_obj() const {
        return *impl;
    }

    virtual size_t size() const override {
        return impl->size();
    }

    virtual double get_beta() const override {
        return beta;
    }

    virtual int nroots() const override {
        return roots().size();
    }

    virtual Eigen::VectorXd roots() const override {
        if (!impl || impl->size() == 0) {
            return Eigen::VectorXd();
        }

        // Collect all roots from all functions
        auto roots_ = impl->roots();
        if (roots_.size() == 0) {
            return Eigen::VectorXd();
        }

        std::set<double> all_roots;
        for (int i = 0; i < roots_.size(); ++i) {
            all_roots.insert(roots_[i]);
            if (roots_[i] >= 0 && roots_[i] <= beta) {
                all_roots.insert(roots_[i] - beta);
            }
        }

        // Convert to vector and sort in non-ascending order
        Eigen::VectorXd result(all_roots.size());
        int i = 0;
        for (auto it = all_roots.rbegin(); it != all_roots.rend(); ++it) {
            result[i++] = *it;
        }
        return result;
    }

    PeriodicFunctions<S, ImplType> operator[](size_t i) const {
        return PeriodicFunctions<S, ImplType>(impl->slice(i), beta);
    }

    // Slice method to extract multiple functions by indices
    std::shared_ptr<PeriodicFunctions<S, ImplType>> slice(const std::vector<size_t>& indices) const {
        // Check for duplicate indices and out of range indices
        check_indices(indices, impl->size());
        
        // Create new PeriodicFunctions with selected functions
        auto new_impl = impl->slice(indices);
        return std::make_shared<PeriodicFunctions<S, ImplType>>(new_impl, beta);
    }
};


} // namespace sparseir
