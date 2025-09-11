#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp> // for Approx

#include "sve_cache.hpp"
#include <catch2/catch_test_macros.hpp>
#include <sparseir/sparseir.hpp>
#include <xprec/ddouble-header-only.hpp>

TEST_CASE("AbstractAugmentation", "[augment]")
{
    using Catch::Approx;

    SECTION("TauConst")
    {
        REQUIRE_THROWS_AS(sparseir::TauConst(-34), std::domain_error);

        sparseir::TauConst tc(123);
        REQUIRE(tc.beta == 123.0);

        REQUIRE_THROWS_AS(tc(-3), std::domain_error);
        REQUIRE_THROWS_AS(tc(321), std::domain_error);

        REQUIRE(tc(100) == 1 / std::sqrt(123));
        sparseir::MatsubaraFreq<sparseir::Bosonic> freq0(0);
        sparseir::MatsubaraFreq<sparseir::Bosonic> freq92(92);
        REQUIRE(tc(freq0) == std::sqrt(123));
        REQUIRE(tc(freq92) == 0.0);
        sparseir::MatsubaraFreq<sparseir::Fermionic> freq93(93);
        REQUIRE_THROWS_AS(tc(freq93), std::invalid_argument);
        auto tdc = tc.deriv();
        REQUIRE(tdc(4.2) == 0.0);

        tdc = tc.deriv(0);
        double x = 4.2;
        // Expected tdc == tc
        REQUIRE(tdc(x) == tc(x));
    }

    SECTION("TauLinear")
    {
        REQUIRE_THROWS_AS(sparseir::TauLinear(-34), std::domain_error);

        sparseir::TauLinear tl(123);
        REQUIRE(tl.beta == Approx(123.0));

        REQUIRE_THROWS_AS(tl(-3), std::domain_error);
        REQUIRE_THROWS_AS(tl(321), std::domain_error);
        REQUIRE(tl.norm == Approx(std::sqrt(3.0 / 123.0)));
        double tau = 100;
        REQUIRE(tl(tau) == std::sqrt(3.0 / 123.0) * (2.0 / 123. * tau - 1.));

        sparseir::MatsubaraFreq<sparseir::Bosonic> freq0(0);
        REQUIRE(tl(freq0) == 0.0);
        sparseir::MatsubaraFreq<sparseir::Bosonic> freq92(92);
        // Calculate the expected complex value
        std::complex<double> expected_value = std::sqrt(3. / 123.) * 2. /
                                              std::complex<double>(0, 1) *
                                              123. / (92. * M_PI);
        // Get the actual value from the function
        std::complex<double> actual_value = tl(freq92);

        REQUIRE(actual_value.real() == Approx(expected_value.real()));
        REQUIRE(actual_value.imag() == Approx(expected_value.imag()));

        sparseir::MatsubaraFreq<sparseir::Fermionic> freq93(93);
        REQUIRE_THROWS_AS(tl(freq93), std::invalid_argument);

        double x = 4.2;
        auto d0tl = tl.deriv(0);
        REQUIRE(d0tl(x) == tl(x));
        auto dtl = tl.deriv();
        REQUIRE(dtl(4.2) == Approx(std::sqrt(3. / 123.) * 2. / 123.));
        auto ddtl = tl.deriv(2);
        REQUIRE(ddtl(4.2) == 0.0);
    }

    SECTION("MatsubaraConst")
    {
        REQUIRE_THROWS_AS(sparseir::MatsubaraConst(-34), std::domain_error);

        sparseir::MatsubaraConst mc(123);
        REQUIRE(mc.beta == Approx(123.0));

        REQUIRE_THROWS_AS(mc(-3), std::domain_error);
        REQUIRE_THROWS_AS(mc(321), std::domain_error);

        REQUIRE(std::isnan(mc(100)));
        sparseir::MatsubaraFreq<sparseir::Bosonic> freq0(0);
        REQUIRE(mc(freq0) == 1.0);
        sparseir::MatsubaraFreq<sparseir::Bosonic> freq92(0);
        REQUIRE(mc(freq92) == 1.0);
        sparseir::MatsubaraFreq<sparseir::Fermionic> freq93(93);
        REQUIRE(mc(freq93) == 1.0);

        auto d0mc = mc.deriv(0);
        auto dmc = mc.deriv();
        double x = 4.2;
        // Expected dmc == mc
        REQUIRE(std::isnan(d0mc(x)));
        REQUIRE(std::isnan(mc(x)));
        REQUIRE(std::isnan(dmc(x)));
        REQUIRE(std::isnan(mc(x)));
    }
}

TEST_CASE("Augmented bosonic basis", "[augment]")
{
    double beta = 1000.0;
    double wmax = 2.0;
    double Lambda = beta * wmax;
    // Create bosonic basis
    auto kernel = sparseir::LogisticKernel(Lambda);
    auto sve_result = SVECache::get_sve_result(kernel, 1e-6);
    auto basis = std::make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
        beta, wmax, 1e-6, kernel, sve_result);
    // Create augmented basis with TauConst and TauLinear
    std::vector<std::shared_ptr<sparseir::AbstractAugmentation>> augmentations;
    augmentations.push_back(std::make_shared<sparseir::TauConst>(beta));
    augmentations.push_back(std::make_shared<sparseir::TauLinear>(beta));
    auto u = *basis->u;
    auto uhat = *basis->uhat;

    // Define G(τ) = c - exp(-τ * pole) / (1 - exp(-β * pole))
    double pole = 1.0;
    double c = 1e-2;
    // Create tau sampling points
    auto basis_aug =
        std::make_shared<sparseir::AugmentedBasis<sparseir::Bosonic>>(
            basis, augmentations);

    auto tau_sampling = sparseir::TauSampling<sparseir::Bosonic>(basis_aug);
    Eigen::VectorXd tau = tau_sampling.sampling_points();
    Eigen::VectorXd gtau(tau.size());
    for (size_t i = 0; i < tau.size(); ++i) {
        gtau(i) = c - std::exp(-tau(i) * pole) / (1 - std::exp(-beta * pole));
    }
    double magn = gtau.array().abs().maxCoeff();

    // This illustrates that "naive" fitting is a problem if the fitting matrix
    // is not well-conditioned.
    Eigen::MatrixXd tau_matrix = tau_sampling.get_matrix();

    Eigen::VectorXd gl_fit_bad = sparseir::pinv(tau_matrix) * gtau;
    Eigen::VectorXd gtau_reconst_bad = tau_matrix * gl_fit_bad;
    // Check that the naive reconstruction is not exact
    REQUIRE(!gtau_reconst_bad.isApprox(gtau, 1e-13 * magn));
    // TODO: Need to port to C++ the collowing lines:
    // @test isapprox(gτ_reconst_bad, gτ, atol=5e-16 * cond(τ_smpl) * magn)
    // @test cond(τ_smpl) > 1e7
    // @test size(τ_smpl.matrix) == (length(basis_aug), length(τ_smpl.τ))
    REQUIRE(tau_matrix.cols() == tau.size());
    REQUIRE(tau_matrix.rows() == basis_aug->size());

    // Now do the fit properly
    Eigen::Tensor<double, 1> gtau_tensor(gtau.size());
    for (size_t i = 0; i < gtau.size(); ++i) {
        gtau_tensor(i) = gtau(i);
    }
    auto gl_fit = tau_sampling.fit(gtau_tensor);
    Eigen::Tensor<double, 1> gtau_reconst_tensor =
        tau_sampling.evaluate(gl_fit);
    Eigen::VectorXd gtau_reconst(gtau.size());
    for (size_t i = 0; i < gtau.size(); ++i) {
        gtau_reconst(i) = gtau_reconst_tensor(i);
    }
    REQUIRE(gtau_reconst.isApprox(gtau, 1e-14 * magn));
}

TEST_CASE("Vertex basis with stat = Bosonic", "[augment]")
{

    double beta = 1000.0;
    double wmax = 2.0;
    // Create kernel and get SVE result from cache
    auto kernel = sparseir::LogisticKernel(beta * wmax);
    auto sve_result = SVECache::get_sve_result(kernel, 1e-6);
    auto basis = std::make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
        beta, wmax, 1e-6, kernel, sve_result);

    std::vector<std::shared_ptr<sparseir::AbstractAugmentation>> augmentations;
    augmentations.push_back(std::make_shared<sparseir::MatsubaraConst>(beta));

    // G(iν) = c + 1 / (iν - pole)
    double pole = 1.0;
    double c = 1.0;
    // Create a shared_ptr to AugmentedBasis
    auto basis_aug =
        std::make_shared<sparseir::AugmentedBasis<sparseir::Bosonic>>(
            basis, augmentations);

    auto matsu_sampling =
        std::make_shared<sparseir::MatsubaraSampling<sparseir::Bosonic>>(
            basis_aug);
    Eigen::VectorXcd gi_n(matsu_sampling->sampling_points().size());
    for (std::size_t i = 0; i < matsu_sampling->sampling_points().size(); ++i) {
        std::complex<double> iwn(
            0, matsu_sampling->sampling_points()[i].value(beta));
        gi_n(i) = c + 1.0 / (iwn - pole);
    }
    // Convert VectorXcd to Tensor
    Eigen::Tensor<std::complex<double>, 1> gi_n_tensor(gi_n.size());
    for (std::size_t i = 0; i < gi_n.size(); ++i) {
        gi_n_tensor(i) = gi_n(i);
    }

    // Fit the data
    Eigen::Tensor<std::complex<double>, 1> gl =
        matsu_sampling->fit(gi_n_tensor);
    Eigen::Tensor<std::complex<double>, 1> gi_n_reconst =
        matsu_sampling->evaluate(gl);
    // TODO: Check if this is correct
    REQUIRE(sparseir::tensorIsApprox(gi_n_reconst, gi_n_tensor,
                                     1e-7 * gi_n.array().abs().maxCoeff()));
}

TEST_CASE("Vertex basis with stat = Fermionic", "[augment]")
{

    double beta = 1000.0;
    double wmax = 2.0;
    // Create kernel and get SVE result from cache
    auto kernel = sparseir::LogisticKernel(beta * wmax);
    auto sve_result = SVECache::get_sve_result(kernel, 1e-6);
    auto basis =
        std::make_shared<sparseir::FiniteTempBasis<sparseir::Fermionic>>(
            beta, wmax, 1e-6, kernel, sve_result);

    std::vector<std::shared_ptr<sparseir::AbstractAugmentation>> augmentations;
    augmentations.push_back(std::make_shared<sparseir::MatsubaraConst>(beta));

    // G(iν) = c + 1 / (iν - pole)
    double pole = 1.0;
    double c = 1.0;
    // Create a shared_ptr to AugmentedBasis
    auto basis_aug =
        std::make_shared<sparseir::AugmentedBasis<sparseir::Fermionic>>(
            basis, augmentations);

    auto matsu_sampling =
        std::make_shared<sparseir::MatsubaraSampling<sparseir::Fermionic>>(
            basis_aug);
    Eigen::VectorXcd gi_n(matsu_sampling->sampling_points().size());
    for (std::size_t i = 0; i < matsu_sampling->sampling_points().size(); ++i) {
        std::complex<double> iwn(
            0, matsu_sampling->sampling_points()[i].value(beta));
        gi_n(i) = c + 1.0 / (iwn - pole);
    }
    // Convert VectorXcd to Tensor
    Eigen::Tensor<std::complex<double>, 1> gi_n_tensor(gi_n.size());
    for (std::size_t i = 0; i < gi_n.size(); ++i) {
        gi_n_tensor(i) = gi_n(i);
    }

    // Fit the data
    Eigen::Tensor<std::complex<double>, 1> gl =
        matsu_sampling->fit(gi_n_tensor);
    Eigen::Tensor<std::complex<double>, 1> gi_n_reconst =
        matsu_sampling->evaluate(gl);
    // TODO: Check if this is correct
    REQUIRE(sparseir::tensorIsApprox(gi_n_reconst, gi_n_tensor,
                                     1e-7 * gi_n.array().abs().maxCoeff()));
}

TEST_CASE("unit tests", "[augment]")
{
    using T = double;
    T beta = 1000.0;
    T wmax = 2.0;
    using S = sparseir::Bosonic;
    // Create kernel and get SVE result from cache
    auto kernel = sparseir::LogisticKernel(beta * wmax);
    auto sve_result = SVECache::get_sve_result(kernel, 1e-6);
    auto basis = std::make_shared<sparseir::FiniteTempBasis<S>>(
        beta, wmax, 1e-6, kernel, sve_result);
    std::vector<std::shared_ptr<sparseir::AbstractAugmentation>> augmentations;
    augmentations.push_back(std::make_shared<sparseir::TauConst>(beta));
    augmentations.push_back(std::make_shared<sparseir::TauLinear>(beta));
    sparseir::AugmentedBasis<S> basis_aug(basis, augmentations);
    SECTION("size and access")
    {
        REQUIRE(basis_aug.u->size() == basis_aug.size());
        // The AugmentedTauFunction doesn't have operator[] so we can't test
        // these REQUIRE(basis_aug.u[0]->operator()(0.0) == basis_aug[0](0.0));
        // REQUIRE(basis_aug.u[1]->operator()(0.0) == basis_aug[1](0.0));
    }

    size_t len_basis = basis->size();
    size_t len_aug = len_basis + 2;

    REQUIRE(basis_aug.size() == len_aug);
    REQUIRE(basis_aug.get_accuracy() == basis->get_accuracy());
    // AugmentedBasis doesn't have Lambda() method, but we can check the
    // equivalent
    REQUIRE(basis_aug.basis->get_beta() * basis_aug.basis->get_wmax() ==
            beta * wmax);
    REQUIRE(basis_aug.get_wmax() == wmax);

    REQUIRE(basis_aug.u->size() == len_aug);
    REQUIRE((*basis_aug.u)(0.8).size() == len_aug);

    // Test for MatsubaraFreq
    sparseir::MatsubaraFreq<S> freq4(4);
    REQUIRE((*basis_aug.uhat)(freq4).size() == len_aug);

    // AugmentedTauFunction doesn't have minCoeff/maxCoeff methods
    // REQUIRE(basis_aug.u.minCoeff() == 0.0);
    // REQUIRE(basis_aug.u.maxCoeff() == beta);
}

TEST_CASE("AugmentBasis basis_aug->u", "[augment]")
{
    double omega_max = 2.0;
    // double beta = 1000.0;
    double beta = 10.0;
    double epsilon = 1e-6;

    // Create kernel and get SVE result from cache
    auto kernel = sparseir::LogisticKernel(beta * omega_max);
    auto sve_result = SVECache::get_sve_result(kernel, epsilon);
    auto basis = std::make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
        beta, omega_max, epsilon, kernel, sve_result);

    std::vector<std::shared_ptr<sparseir::AbstractAugmentation>> augmentations;
    augmentations.push_back(std::make_shared<sparseir::TauConst>(beta));
    augmentations.push_back(std::make_shared<sparseir::TauLinear>(beta));
    auto basis_aug =
        std::make_shared<sparseir::AugmentedBasis<sparseir::Bosonic>>(
            basis, augmentations);

    Eigen::VectorXd sampling_points = basis_aug->default_tau_sampling_points();
    sparseir::AugmentedTauFunction<sparseir::Bosonic> u = *(basis_aug->u);
    // TODO: Check numerical correctness
    Eigen::VectorXd v = u(1.0);

    Eigen::MatrixXd m = u(sampling_points);
    REQUIRE(m.cols() == sampling_points.size());
}

TEST_CASE("AugmentBasis basis_aug->uha", "[augment]")
{
    double omega_max = 2.0;
    // double beta = 1000.0;
    double beta = 10.0;
    double epsilon = 1e-6;

    // Create kernel and get SVE result from cache
    auto kernel = sparseir::LogisticKernel(beta * omega_max);
    auto sve_result = SVECache::get_sve_result(kernel, epsilon);
    auto basis = std::make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
        beta, omega_max, epsilon, kernel, sve_result);

    std::vector<std::shared_ptr<sparseir::AbstractAugmentation>> augmentations;
    augmentations.push_back(std::make_shared<sparseir::TauConst>(beta));
    augmentations.push_back(std::make_shared<sparseir::TauLinear>(beta));
    auto basis_aug =
        std::make_shared<sparseir::AugmentedBasis<sparseir::Bosonic>>(
            basis, augmentations);
    bool fence = false;
    bool positive_only = true;
    auto sampling_points = basis_aug->default_matsubara_sampling_points(
        basis_aug->size(), fence, positive_only);
    Eigen::VectorXcd v = (*basis_aug->uhat)(sampling_points[0]);
    REQUIRE(v.size() == basis_aug->size());
}
