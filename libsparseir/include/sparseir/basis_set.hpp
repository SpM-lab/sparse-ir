#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "basis.hpp"
#include "freq.hpp"

namespace sparseir {

class FiniteTempBasisSet {
public:
    // Member variables
    std::shared_ptr<FiniteTempBasis<Fermionic>> basis_f;
    std::shared_ptr<FiniteTempBasis<Bosonic>> basis_b;
    Eigen::VectorXd tau;
    std::vector<int> wn_f;
    std::vector<int> wn_b;

    // Constructor
    FiniteTempBasisSet(std::shared_ptr<FiniteTempBasis<Fermionic>> basis_f_,
                       std::shared_ptr<FiniteTempBasis<Bosonic>> basis_b_,
                       const Eigen::VectorXd &tau_,
                       const std::vector<int> &wn_f_,
                       const std::vector<int> &wn_b_)
        : basis_f(basis_f_),
          basis_b(basis_b_),
          tau(tau_),
          wn_f(wn_f_),
          wn_b(wn_b_)
    {
        // Validate that bases have same parameters
        if (std::abs(basis_f->get_beta() - basis_b->get_beta()) > 1e-10) {
            throw std::runtime_error(
                "Fermionic and bosonic bases must have same beta");
        }
        if (std::abs(basis_f->get_wmax() - basis_b->get_wmax()) > 1e-10) {
            throw std::runtime_error(
                "Fermionic and bosonic bases must have same wmax");
        }
    }

    // Getters
    double beta() const { return basis_f->get_beta(); }
    double wmax() const { return basis_f->get_wmax(); }
    const SVEResult &sve_result() const { return *basis_f->sve_result; }

    // Factory method
    static std::shared_ptr<FiniteTempBasisSet>
    create(double beta, double wmax, double epsilon)
    {
        auto bases = finite_temp_bases(beta, wmax, epsilon);
        auto basis_f = bases.first;
        auto basis_b = bases.second;

        // Get default sampling points
        Eigen::VectorXd tau = basis_f->default_tau_sampling_points();

        // Get Matsubara frequencies using uhat_full
        std::vector<int> wn_f;
        bool fence = false;
        bool positive_only = false;
        for (const auto &freq : basis_f->default_matsubara_sampling_points(
                 basis_f->size(), fence, positive_only)) {
            wn_f.push_back(freq.get_n());
        }

        std::vector<int> wn_b;
        for (const auto &freq : basis_b->default_matsubara_sampling_points(
                 basis_b->size(), fence, positive_only)) {
            wn_b.push_back(freq.get_n());
        }

        return std::make_shared<FiniteTempBasisSet>(basis_f, basis_b, tau, wn_f, wn_b);
    }

    // String representation
    friend std::ostream &operator<<(std::ostream &os,
                                    const FiniteTempBasisSet &b)
    {
        os << "FiniteTempBasisSet with β = " << b.beta()
           << ", ωmax = " << b.wmax();
        return os;
    }
};

} // namespace sparseir