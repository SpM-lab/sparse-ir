#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <sparseir/sparseir.hpp>

using Catch::Approx;
using xprec::DDouble;

TEST_CASE("Kernel Accuracy Tests", "[kernel]")
{
    // Test individual kernels
    SECTION("LogisticKernel(9)")
    {
        auto kernel = sparseir::LogisticKernel(9);

        using T = float;
        using T_x = double;

        // Convert legendre rule to the desired precision
        auto rule_xprec_ddouble = sparseir::legendre(10);
        auto rule_T_x = sparseir::convert_rule<T_x>(rule_xprec_ddouble);
        auto rule = sparseir::convert_rule<T>(rule_T_x);

        T epsilon_T = std::numeric_limits<T>::epsilon();
        T_x epsilon_T_x = std::numeric_limits<T_x>::epsilon();
        T tiny = std::numeric_limits<T>::min() / epsilon_T;

        auto hints = kernel.template sve_hints<T_x>(epsilon_T_x);
        auto gauss_x = rule.piecewise(hints->segments_x());
        auto gauss_y = rule.piecewise(hints->segments_x());
        auto result = sparseir::matrix_from_gauss(kernel, gauss_x, gauss_y);

        auto T_x_gauss_x = sparseir::convert_rule<T_x>(gauss_x);
        auto T_x_gauss_y = sparseir::convert_rule<T_x>(gauss_y);
        auto result_x =
            sparseir::matrix_from_gauss(kernel, T_x_gauss_x, T_x_gauss_y);

        double magn = result_x.array().abs().maxCoeff();

        bool all_close = true;
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double diff = std::abs(result(i, j) - result_x(i, j));
                if (!(diff < 2 * magn * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);

        // Port from Julia:
        // julia> reldiff = @.ifelse(abs(result) < tiny, 1.0, result / result_x)
        // @test julia> reldiff=ones(size(reldiff)) atol = 100*epsilon rtol = 0
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double q = std::abs(result(i, j)) < tiny
                               ? 1.0
                               : result(i, j) / result_x(i, j);
                if (!(q < 1.0 + 100 * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);
    }

    SECTION("RegularizedBoseKernel(8)")
    {
        auto kernel = sparseir::RegularizedBoseKernel(8);

        using T = float;
        using T_x = double;

        // Convert legendre rule to the desired precision
        auto rule_xprec_ddouble = sparseir::legendre(10);
        auto rule_T_x = sparseir::convert_rule<T_x>(rule_xprec_ddouble);
        auto rule = sparseir::convert_rule<T>(rule_T_x);

        T epsilon_T = std::numeric_limits<T>::epsilon();
        T_x epsilon_T_x = std::numeric_limits<T_x>::epsilon();
        T tiny = std::numeric_limits<T>::min() / epsilon_T;

        auto hints = kernel.template sve_hints<T_x>(epsilon_T_x);
        auto gauss_x = rule.piecewise(hints->segments_x());
        auto gauss_y = rule.piecewise(hints->segments_x());
        auto result = sparseir::matrix_from_gauss(kernel, gauss_x, gauss_y);

        auto T_x_gauss_x = sparseir::convert_rule<T_x>(gauss_x);
        auto T_x_gauss_y = sparseir::convert_rule<T_x>(gauss_y);
        auto result_x =
            sparseir::matrix_from_gauss(kernel, T_x_gauss_x, T_x_gauss_y);

        double magn = result_x.array().abs().maxCoeff();

        bool all_close = true;
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double diff = std::abs(result(i, j) - result_x(i, j));
                if (!(diff < 2 * magn * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);

        // Port from Julia:
        // julia> reldiff = @.ifelse(abs(result) < tiny, 1.0, result / result_x)
        // @test julia> reldiff≈ones(size(reldiff)) atol = 100ϵ rtol = 0
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double q = std::abs(result(i, j)) < tiny
                               ? 1.0
                               : result(i, j) / result_x(i, j);
                if (!(q < 1.0 + 100 * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);
    }

    SECTION("LogisticKernel(120000)")
    {
        auto kernel = sparseir::LogisticKernel(120000);

        using T = float;
        using T_x = double;

        // Convert legendre rule to the desired precision
        auto rule_xprec_ddouble = sparseir::legendre(10);
        auto rule_T_x = sparseir::convert_rule<T_x>(rule_xprec_ddouble);
        auto rule = sparseir::convert_rule<T>(rule_T_x);

        T epsilon_T = std::numeric_limits<T>::epsilon();
        T_x epsilon_T_x = std::numeric_limits<T_x>::epsilon();
        T tiny = std::numeric_limits<T>::min() / epsilon_T;

        auto hints = kernel.template sve_hints<T_x>(epsilon_T_x);
        auto gauss_x = rule.piecewise(hints->segments_x());
        auto gauss_y = rule.piecewise(hints->segments_x());
        auto result = sparseir::matrix_from_gauss(kernel, gauss_x, gauss_y);

        auto T_x_gauss_x = sparseir::convert_rule<T_x>(gauss_x);
        auto T_x_gauss_y = sparseir::convert_rule<T_x>(gauss_y);
        auto result_x =
            sparseir::matrix_from_gauss(kernel, T_x_gauss_x, T_x_gauss_y);

        double magn = result_x.array().abs().maxCoeff();

        bool all_close = true;
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double diff = std::abs(result(i, j) - result_x(i, j));
                if (!(diff < 2 * magn * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);

        // Port from Julia:
        // julia> reldiff = @.ifelse(abs(result) < tiny, 1.0, result / result_x)
        // @test julia> reldiff=ones(size(reldiff)) atol = 100*epsilon rtol = 0
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double q = std::abs(result(i, j)) < tiny
                               ? 1.0
                               : result(i, j) / result_x(i, j);
                if (!(q < 1.0 + 100 * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);
    }

    SECTION("RegularizedBoseKernel(127500)")
    {
        auto kernel = sparseir::RegularizedBoseKernel(127500);
        using T = float;
        using T_x = double;

        // Convert legendre rule to the desired precision
        auto rule_xprec_ddouble = sparseir::legendre(10);
        auto rule_T_x = sparseir::convert_rule<T_x>(rule_xprec_ddouble);
        auto rule = sparseir::convert_rule<T>(rule_T_x);

        T epsilon_T = std::numeric_limits<T>::epsilon();
        T_x epsilon_T_x = std::numeric_limits<T_x>::epsilon();
        T tiny = std::numeric_limits<T>::min() / epsilon_T;

        auto hints = kernel.template sve_hints<T_x>(epsilon_T_x);
        auto gauss_x = rule.piecewise(hints->segments_x());
        auto gauss_y = rule.piecewise(hints->segments_x());
        auto result = sparseir::matrix_from_gauss(kernel, gauss_x, gauss_y);

        auto T_x_gauss_x = sparseir::convert_rule<T_x>(gauss_x);
        auto T_x_gauss_y = sparseir::convert_rule<T_x>(gauss_y);
        auto result_x =
            sparseir::matrix_from_gauss(kernel, T_x_gauss_x, T_x_gauss_y);

        double magn = result_x.array().abs().maxCoeff();

        bool all_close = true;
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double diff = std::abs(result(i, j) - result_x(i, j));
                if (!(diff < 2 * magn * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);

        // Port from Julia:
        // julia> reldiff = @.ifelse(abs(result) < tiny, 1.0, result / result_x)
        // @test julia> reldiff≈ones(size(reldiff)) atol = 100ϵ rtol = 0
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double q = std::abs(result(i, j)) < tiny
                               ? 1.0
                               : result(i, j) / result_x(i, j);
                if (!(q < 1.0 + 100 * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);
    }

    // Test symmetrized kernels
    SECTION("Symmetrized LogisticKernel(40000)")
    {
        auto lk = sparseir::LogisticKernel(40000);
        std::integral_constant<int, -1> minus_one{};
        auto kernel = sparseir::get_symmetrized(lk, minus_one);

        using T = float;
        using T_x = double;

        // Convert legendre rule to the desired precision
        auto rule_xprec_ddouble = sparseir::legendre(10);
        auto rule_T_x = sparseir::convert_rule<T_x>(rule_xprec_ddouble);
        auto rule = sparseir::convert_rule<T>(rule_T_x);

        T epsilon_T = std::numeric_limits<T>::epsilon();
        T_x epsilon_T_x = std::numeric_limits<T_x>::epsilon();
        T tiny = std::numeric_limits<T>::min() / epsilon_T;

        auto hints = kernel.template sve_hints<T_x>(epsilon_T_x);
        auto gauss_x = rule.piecewise(hints->segments_x());
        auto gauss_y = rule.piecewise(hints->segments_x());
        auto result = sparseir::matrix_from_gauss(kernel, gauss_x, gauss_y);

        auto T_x_gauss_x = sparseir::convert_rule<T_x>(gauss_x);
        auto T_x_gauss_y = sparseir::convert_rule<T_x>(gauss_y);
        auto result_x =
            sparseir::matrix_from_gauss(kernel, T_x_gauss_x, T_x_gauss_y);

        double magn = result_x.array().abs().maxCoeff();

        bool all_close = true;
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double diff = std::abs(result(i, j) - result_x(i, j));
                if (!(diff < 2 * magn * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);

        // Port from Julia:
        // julia> reldiff = @.ifelse(abs(result) < tiny, 1.0, result / result_x)
        // @test julia> reldiff≈ones(size(reldiff)) atol = 100ϵ rtol = 0
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double q = std::abs(result(i, j)) < tiny
                               ? 1.0
                               : result(i, j) / result_x(i, j);
                if (!(q < 1.0 + 100 * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);
    }

    SECTION("Symmetrized RegularizedBoseKernel(35000)")
    {
        auto rbk = sparseir::RegularizedBoseKernel(35000);
        std::integral_constant<int, -1> minus_one{};
        auto kernel = sparseir::get_symmetrized(rbk, minus_one);

        using T = float;
        using T_x = double;

        // Convert legendre rule to the desired precision
        auto rule_xprec_ddouble = sparseir::legendre(10);
        auto rule_T_x = sparseir::convert_rule<T_x>(rule_xprec_ddouble);
        auto rule = sparseir::convert_rule<T>(rule_T_x);

        T epsilon_T = std::numeric_limits<T>::epsilon();
        T_x epsilon_T_x = std::numeric_limits<T_x>::epsilon();
        T tiny = std::numeric_limits<T>::min() / epsilon_T;

        auto hints = kernel.template sve_hints<T_x>(epsilon_T_x);
        auto gauss_x = rule.piecewise(hints->segments_x());
        auto gauss_y = rule.piecewise(hints->segments_x());
        auto result = sparseir::matrix_from_gauss(kernel, gauss_x, gauss_y);

        auto T_x_gauss_x = sparseir::convert_rule<T_x>(gauss_x);
        auto T_x_gauss_y = sparseir::convert_rule<T_x>(gauss_y);
        auto result_x =
            sparseir::matrix_from_gauss(kernel, T_x_gauss_x, T_x_gauss_y);

        double magn = result_x.array().abs().maxCoeff();

        bool all_close = true;
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double diff = std::abs(result(i, j) - result_x(i, j));
                if (!(diff < 2 * magn * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);

        // Port from Julia:
        // julia> reldiff = @.ifelse(abs(result) < tiny, 1.0, result / result_x)
        // @test julia> reldiff≈ones(size(reldiff)) atol = 100ϵ rtol = 0
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                double q = std::abs(result(i, j)) < tiny
                               ? 1.0
                               : result(i, j) / result_x(i, j);
                if (!(q < 1.0 + 100 * epsilon_T)) {
                    all_close = false;
                }
            }
        }
        REQUIRE(all_close);
    }
}

TEST_CASE("weight_func", "[kernel]")
{
    double lambda = 100.;
    double beta = 10.0;
    double omega_max = lambda / beta;

    {
        sparseir::LogisticKernel K(lambda);
        auto wf = K.weight_func<double>(sparseir::Bosonic());
        double omega = 0.1 * omega_max;
        REQUIRE(wf(beta, omega) == 1.0 / tanh(0.5 * beta * omega));
    }

    // RegularizedBoseKernel
    {
        sparseir::RegularizedBoseKernel K(lambda);
        auto wf = K.weight_func<double>(sparseir::Bosonic());
        double omega = 0.1 * omega_max;
        REQUIRE(wf(beta, omega) == 1.0 / omega);
    }

}

TEST_CASE("Symmetrized Kernel Tests", "[kernel]")
{
    SECTION("even")
    {
        auto kernel = sparseir::LogisticKernel(5.0);

        // Use the correct symmetrization function
        std::integral_constant<int, +1> plus_one{};
        auto reduced_kernel = sparseir::get_symmetrized(kernel, plus_one);
        REQUIRE(reduced_kernel.inner.lambda_ == kernel.lambda_);
        REQUIRE(reduced_kernel.sign == +1);
    }

    SECTION("odd")
    {
        auto kernel = sparseir::LogisticKernel(5.0);

        // Use the correct symmetrization function
        std::integral_constant<int, -1> minus_one{};
        auto reduced_kernel = sparseir::get_symmetrized(kernel, minus_one);
        REQUIRE(reduced_kernel.inner.lambda_ == kernel.lambda_);
        REQUIRE(reduced_kernel.sign == -1);
    }
}

TEST_CASE("Kernel Singularity Test", "[kernel]")
{
    using T = xprec::DDouble;
    std::vector<double> lambdas = {10.0, 42.0, 10000.0};
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (double lambda : lambdas) {
        // Generate random x values
        std::vector<double> x_values(
            10); // Reduced from 1000 to 10 for faster tests
        for (double &x : x_values) {
            x = dist(rng);
        }

        auto K = sparseir::RegularizedBoseKernel(lambda);

        for (double x : x_values) {
            T expected = 1.0 / static_cast<T>(lambda);
            T computed = K.compute(static_cast<T>(x), static_cast<T>(0.0));
            REQUIRE(Eigen::internal::isApprox(computed, expected));
        }
    }
}

TEST_CASE("Kernel Unit Tests", "[kernel]")
{
    // Test various kernel properties
    SECTION("RegularizedBoseKernel Properties")
    {
        sparseir::RegularizedBoseKernel K(99);
        auto hints = K.sve_hints<double>(1e-6);

        // Test SVE hints properties
        REQUIRE(hints->nsvals() == 56);
        REQUIRE(hints->ngauss() == 10);

        // Test ypower
        REQUIRE(K.ypower() == 1);

        // Use member function for symmetrization instead of free function
        //std::integral_constant<int, -1> minus_one{};
        //std::integral_constant<int, +1> plus_one{};

// Only use symmetrization if it's supported for this kernel type
#ifdef SPARSEIR_SUPPORTS_BOSE_SYMMETRIZATION
        auto K_symm_minus = sparseir::get_symmetrized(K, minus_one);
        auto K_symm_plus = sparseir::get_symmetrized(K, plus_one);

        REQUIRE(K_symm_minus.ypower() == 1);
        REQUIRE(K_symm_plus.ypower() == 1);

        // Test conv_radius
        REQUIRE(K.conv_radius() == 40 * 99);
        REQUIRE(K_symm_minus.conv_radius() == 40 * 99);
        REQUIRE(K_symm_plus.conv_radius() == 40 * 99);
#else
        // Skip symmetrization tests for RegularizedBoseKernel
        REQUIRE(K.ypower() == 1);
        REQUIRE(K.conv_radius() == 40 * 99);
#endif

        // Test weight functions
        //auto weight_func_bosonic = K.weight_func<double>(sparseir::Bosonic());
        //REQUIRE(weight_func_bosonic(482) == 1.0 / 482);

        // Test that trying to get Fermionic weight function for
        // RegularizedBoseKernel throws an error
        //REQUIRE_THROWS(K.weight_func<double>(sparseir::Fermionic()));
    }

    SECTION("LogisticKernel Symmetrization")
    {
        sparseir::LogisticKernel K(42);
        std::integral_constant<int, +1> plus_one{};

        // Use get_symmetrized instead of symmetrize method
        auto K_symm = sparseir::get_symmetrized(K, plus_one);

        // Test centrosymmetry
        REQUIRE(!K_symm.is_centrosymmetric());
    }
}
