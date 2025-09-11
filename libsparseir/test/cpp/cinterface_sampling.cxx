#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <sparseir/sparseir.h>   // C interface
#include <sparseir/sparseir.hpp> // C++ interface
#include "_utils.hpp"

using Catch::Approx;
using xprec::DDouble;

template <typename S>
int get_stat()
{
    if (std::is_same<S, sparseir::Fermionic>::value) {
        return SPIR_STATISTICS_FERMIONIC;
    } else {
        return SPIR_STATISTICS_BOSONIC;
    }
}

template <typename S>
spir_sampling *create_tau_sampling(spir_basis *basis)
{
    int status;
    int n_tau_points;
    status = spir_basis_get_n_default_taus(basis, &n_tau_points);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    double *tau_points_org = (double *)malloc(n_tau_points * sizeof(double));
    status = spir_basis_get_default_taus(basis, tau_points_org);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    return spir_tau_sampling_new(basis, n_tau_points, tau_points_org, &status);
}

template <typename S>
spir_sampling *create_matsubara_sampling(spir_basis *basis, bool positive_only)
{
    int status;
    int n_matsubara_points;
    status = spir_basis_get_n_default_matsus(basis, positive_only, &n_matsubara_points);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    std::vector<int64_t> smpl_points(n_matsubara_points);
    status = spir_basis_get_default_matsus(basis, positive_only, smpl_points.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    spir_sampling *sampling = spir_matsu_sampling_new(basis, positive_only, n_matsubara_points, smpl_points.data(), &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(sampling != nullptr);

    return sampling;
}

template <typename S>
void test_tau_sampling()
{
    double beta = 1.0;
    double wmax = 10.0;

    auto stat = get_stat<S>();
    int status;

    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-15, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    int n_tau_points;
    status = spir_basis_get_n_default_taus(basis, &n_tau_points);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_tau_points > 0);

    double *tau_points_org = (double *)malloc(n_tau_points * sizeof(double));
    status = spir_basis_get_default_taus(basis, tau_points_org);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    int n_tau_points_ext = n_tau_points + 1;
    double *tau_points_ext = (double *)malloc(n_tau_points_ext * sizeof(double));
    int n_tau_points_returned;
    status = spir_basis_get_default_taus_ext(
        basis, n_tau_points_ext, tau_points_ext, &n_tau_points_returned);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_tau_points_returned == n_tau_points_ext);
    for (int i = 0; i < n_tau_points_returned; i++) {
        REQUIRE(tau_points_ext[i] >= -0.5 * beta);
        REQUIRE(tau_points_ext[i] <=  0.5 * beta);
    }

    int sampling_status;
    spir_sampling *sampling = spir_tau_sampling_new(basis, n_tau_points, tau_points_org, &sampling_status);
    REQUIRE(sampling_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(sampling != nullptr);

    // Condition number
    double cond_num;
    int cond_num_status = spir_sampling_get_cond_num(sampling, &cond_num);
    REQUIRE(cond_num_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(cond_num > 1.0);

    // Test getting number of sampling points
    int n_points;
    int points_status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points > 0);

    // Test getting sampling points
    double *tau_points = (double *)malloc(n_points * sizeof(double));
    int tau_status = spir_sampling_get_taus(sampling, tau_points);
    REQUIRE(tau_status == SPIR_COMPUTATION_SUCCESS);

    // compare tau_points and tau_points_org
    for (int i = 0; i < n_points; i++) {
        REQUIRE(tau_points[i] == Approx(tau_points_org[i]));
    }

    std::vector<int64_t> matsubara_points(n_points);
    int matsubara_status = spir_sampling_get_matsus(sampling, matsubara_points.data());
    REQUIRE(matsubara_status == SPIR_NOT_SUPPORTED);

    // Clean up
    spir_sampling_release(sampling);
    spir_basis_release(basis);
}

template <typename S>
void test_tau_sampling_evaluation_1d_column_major()
{
    double beta = 1.0;
    double wmax = 10.0;

    auto stat = get_stat<S>();

    // Create basis
    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-10, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    // Create sampling
    spir_sampling *sampling = create_tau_sampling<S>(basis);
    REQUIRE(sampling != nullptr);

    // Test getting number of sampling points
    int n_points;
    int points_status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points > 0);

    // Create equivalent C++ objects for comparison
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, 1e-10);
    sparseir::TauSampling<S> cpp_sampling(cpp_basis);

    int basis_size = cpp_basis.size();
    Eigen::VectorXd cpp_Gl_vec = Eigen::VectorXd::Random(basis_size);
    Eigen::Tensor<double, 1> cpp_Gl(basis_size);
    for (size_t i = 0; i < basis_size; ++i) {
        cpp_Gl(i) = cpp_Gl_vec(i);
    }
    Eigen::Tensor<double, 1> Gtau_cpp = cpp_sampling.evaluate(cpp_Gl);
    Eigen::Tensor<double, 1> gl_from_tau = cpp_sampling.fit(Gtau_cpp);

    // Set up parameters for evaluation
    int ndim = 1;
    int dims[1] = {basis_size};
    int target_dim = 0;

    // Allocate memory for coefficients
    double *coeffs = (double *)malloc(basis_size * sizeof(double));
    // Create coefficients (simple test values)
    for (int i = 0; i < basis_size; i++) {
        coeffs[i] = cpp_Gl_vec(i);
    }

    // Create output buffer
    double *evaluate_output = (double *)malloc(n_points * sizeof(double));
    double *fit_output = (double *)malloc(basis_size * sizeof(double));

    // Evaluate using C API
    int evaluate_status = spir_sampling_eval_dd(
        sampling,
        SPIR_ORDER_COLUMN_MAJOR, // Assuming this enum is defined in the
                                 // header
        ndim, dims, target_dim, coeffs, evaluate_output);

    REQUIRE(evaluate_status == 0);

    for (int i = 0; i < basis_size; i++) {
        REQUIRE(evaluate_output[i] == Approx(Gtau_cpp(i)));
    }

    int fit_status = spir_sampling_fit_dd(
        sampling,
        SPIR_ORDER_COLUMN_MAJOR, // Assuming this enum is defined in the
                                 // header
        ndim, dims, target_dim, evaluate_output, fit_output);

    REQUIRE(fit_status == 0);

    for (int i = 0; i < basis_size; i++) {
        REQUIRE(fit_output[i] == Approx(gl_from_tau(i)));
    }

    // Clean up
    spir_sampling_release(sampling);
    spir_basis_release(basis);
    // Free allocated memory
    free(coeffs);
    free(evaluate_output);
    free(fit_output);
}

template <typename S>
void test_tau_sampling_evaluation_4d_row_major()
{
    double beta = 1.0;
    double wmax = 10.0;

    auto stat = get_stat<S>();

    // Create basis
    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-10, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    // Create sampling
    spir_sampling *sampling = create_tau_sampling<S>(basis);
    REQUIRE(sampling != nullptr);

    // Test getting number of sampling points
    int n_points;
    int points_status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points > 0);

    // Create equivalent C++ objects for comparison
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, 1e-10);
    sparseir::TauSampling<S> cpp_sampling(cpp_basis);

    int basis_size = cpp_basis.size();

    int d1 = 2;
    int d2 = 3;
    int d3 = 4;
    Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 1);

    for (int i = 0; i < rhol_tensor.size(); ++i) {
        rhol_tensor.data()[i] = dis(gen);
    }

    double *evaluate_output =
        (double *)malloc(n_points * d1 * d2 * d3 * sizeof(double));
    double *fit_output =
        (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));

    int ndim = 4;
    int dims1[4] = {basis_size, d1, d2, d3};
    int dims2[4] = {d1, basis_size, d2, d3};
    int dims3[4] = {d1, d2, basis_size, d3};
    int dims4[4] = {d1, d2, d3, basis_size};

    std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

    // Test evaluate() and fit() along each dimension
    for (int dim = 0; dim < 4; ++dim) {
        // Move the "frequency" dimension around
        // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
        Eigen::Tensor<double, 4> gl_cpp =
            sparseir::movedim(rhol_tensor, 0, dim);

        // Evaluate from real-time/tau to imaginary-time/tau
        Eigen::Tensor<double, 4> gtau_cpp = cpp_sampling.evaluate(gl_cpp, dim);

        Eigen::Tensor<double, 4> gl_cpp_fit = cpp_sampling.fit(gtau_cpp, dim);

        int *dims = dims_list[dim];
        int target_dim = dim;

        // Note that we need to specify Eigen::RowMajor here
        // because Eigen::Tensor<T, 4> is column-major by default
        Eigen::Tensor<double, 4, Eigen::RowMajor> gl_cpp_rowmajor(
            dims[0], dims[1], dims[2], dims[3]);

        // Fill row-major tensor in the correct order
        for (int i = 0; i < gl_cpp.dimension(0); ++i) {
            for (int j = 0; j < gl_cpp.dimension(1); ++j) {
                for (int k = 0; k < gl_cpp.dimension(2); ++k) {
                    for (int l = 0; l < gl_cpp.dimension(3); ++l) {
                        gl_cpp_rowmajor(i, j, k, l) = gl_cpp(i, j, k, l);
                    }
                }
            }
        }

        // Evaluate using C API
        int evaluate_status = spir_sampling_eval_dd(
            sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
            gl_cpp_rowmajor.data(), evaluate_output);

        REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

        int fit_status =
            spir_sampling_fit_dd(sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims,
                                 target_dim, evaluate_output, fit_output);

        REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

        // Compare results
        // Note that we need to specify Eigen::RowMajor here
        // because Eigen::Tensor<T, 4> is column-major by default
        Eigen::Tensor<double, 4, Eigen::RowMajor> output_tensor(
            dims[0], dims[1], dims[2], dims[3]);
        Eigen::Tensor<double, 4, Eigen::RowMajor> fit_tensor(dims[0], dims[1],
                                                             dims[2], dims[3]);
        for (int i = 0; i < output_tensor.size(); ++i) {
            // store output data to output_tensor
            output_tensor.data()[i] = evaluate_output[i];
            fit_tensor.data()[i] = fit_output[i];
        }
        // Compare results
        for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
            for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                    for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                        REQUIRE(gtau_cpp(i, j, k, l) ==
                                output_tensor(i, j, k, l));
                        REQUIRE(gl_cpp_fit(i, j, k, l) ==
                                fit_tensor(i, j, k, l));
                    }
                }
            }
        }
    }

    // Clean up
    spir_sampling_release(sampling);
    spir_basis_release(basis);
    free(evaluate_output);
    free(fit_output);
}

template <typename S>
void test_tau_sampling_evaluation_4d_row_major_complex()
{
    double beta = 1.0;
    double wmax = 10.0;

    auto stat = get_stat<S>();

    // Create basis
    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-10, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    // Create sampling
    spir_sampling *sampling = create_tau_sampling<S>(basis);
    REQUIRE(sampling != nullptr);

    // Test getting number of sampling points
    int n_points;
    int points_status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points > 0);

    // Create equivalent C++ objects for comparison
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, 1e-10);
    sparseir::TauSampling<S> cpp_sampling(cpp_basis);

    int basis_size = cpp_basis.size();

    int d1 = 2;
    int d2 = 3;
    int d3 = 4;
    Eigen::Tensor<std::complex<double>, 4> rhol_tensor(basis_size, d1, d2, d3);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 1);

    for (int i = 0; i < rhol_tensor.size(); ++i) {
        rhol_tensor.data()[i] = std::complex<double>(dis(gen), dis(gen));
    }

    c_complex *evaluate_output =
        (c_complex *)malloc(n_points * d1 * d2 * d3 * sizeof(c_complex));
    c_complex *fit_output =
        (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

    int ndim = 4;
    int dims1[4] = {basis_size, d1, d2, d3};
    int dims2[4] = {d1, basis_size, d2, d3};
    int dims3[4] = {d1, d2, basis_size, d3};
    int dims4[4] = {d1, d2, d3, basis_size};

    std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

    // Test evaluate() and fit() along each dimension
    for (int dim = 0; dim < 4; ++dim) {
        // Move the "frequency" dimension around
        // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
        Eigen::Tensor<std::complex<double>, 4> gl_cpp =
            sparseir::movedim(rhol_tensor, 0, dim);

        // Evaluate from real-time/tau to imaginary-time/tau
        Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
            cpp_sampling.evaluate(gl_cpp, dim);
        Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
            cpp_sampling.fit(gtau_cpp, dim);
        int *dims = dims_list[dim];
        int target_dim = dim;

        // Note that we need to specify Eigen::RowMajor here
        // because Eigen::Tensor<T, 4> is column-major by default
        Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> gl_cpp_rowmajor(
            dims[0], dims[1], dims[2], dims[3]);

        // Fill row-major tensor in the correct order
        for (int i = 0; i < gl_cpp.dimension(0); ++i) {
            for (int j = 0; j < gl_cpp.dimension(1); ++j) {
                for (int k = 0; k < gl_cpp.dimension(2); ++k) {
                    for (int l = 0; l < gl_cpp.dimension(3); ++l) {
                        gl_cpp_rowmajor(i, j, k, l) = gl_cpp(i, j, k, l);
                    }
                }
            }
        }

        c_complex *evaluate_input =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
        for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
            evaluate_input[i] = (c_complex){gl_cpp_rowmajor.data()[i].real(),
                                            gl_cpp_rowmajor.data()[i].imag()};
        }
        // Evaluate using C API
        int evaluate_status = spir_sampling_eval_zz(
            sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
            evaluate_input, evaluate_output);

        REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

        int fit_status =
            spir_sampling_fit_zz(sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims,
                                 target_dim, evaluate_output, fit_output);
        REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

        // Compare results
        // Note that we need to specify Eigen::RowMajor here
        // because Eigen::Tensor<T, 4> is column-major by default
        Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> output_tensor(
            dims[0], dims[1], dims[2], dims[3]);
        Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> fit_tensor(
            dims[0], dims[1], dims[2], dims[3]);
        for (int i = 0; i < output_tensor.size(); ++i) {
            // store output data to output_tensor
            // to std::complex<double>
            output_tensor.data()[i] = (std::complex<double>){
                __real__ evaluate_output[i], __imag__ evaluate_output[i]};
            fit_tensor.data()[i] = (std::complex<double>){
                __real__ fit_output[i], __imag__ fit_output[i]};
        }
        // Compare results
        for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
            for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                    for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                        REQUIRE(gtau_cpp(i, j, k, l) ==
                                output_tensor(i, j, k, l));
                        REQUIRE(gl_cpp_fit(i, j, k, l) ==
                                fit_tensor(i, j, k, l));
                    }
                }
            }
        }
    }

    // Clean up
    spir_sampling_release(sampling);
    spir_basis_release(basis);
    free(evaluate_output);
    free(fit_output);
}

template <typename S>
void test_tau_sampling_evaluation_4d_column_major()
{
    double beta = 1.0;
    double wmax = 10.0;

    auto stat = get_stat<S>();

    // Create basis
    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-10, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    // Create sampling
    spir_sampling *sampling = create_tau_sampling<S>(basis);
    REQUIRE(sampling != nullptr);

    // Test getting number of sampling points
    int n_points;
    int points_status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points > 0);

    // Create equivalent C++ objects for comparison
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, 1e-10);
    sparseir::TauSampling<S> cpp_sampling(cpp_basis);

    int basis_size = cpp_basis.size();

    int d1 = 2;
    int d2 = 3;
    int d3 = 4;
    Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 1);

    // Fill rhol_tensor with random complex values
    for (int i = 0; i < rhol_tensor.size(); ++i) {
        rhol_tensor.data()[i] = dis(gen);
    }

    double *evaluate_output =
        (double *)malloc(n_points * d1 * d2 * d3 * sizeof(double));
    double *fit_output =
        (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));

    int ndim = 4;
    int dims1[4] = {basis_size, d1, d2, d3};
    int dims2[4] = {d1, basis_size, d2, d3};
    int dims3[4] = {d1, d2, basis_size, d3};
    int dims4[4] = {d1, d2, d3, basis_size};

    std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

    // Test evaluate() and fit() along each dimension
    for (int dim = 0; dim < 4; ++dim) {
        // Move the "frequency" dimension around
        // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
        Eigen::Tensor<double, 4> gl_cpp =
            sparseir::movedim(rhol_tensor, 0, dim);

        // Evaluate from real-time/tau to imaginary-time/tau
        Eigen::Tensor<double, 4> gtau_cpp = cpp_sampling.evaluate(gl_cpp, dim);
        Eigen::Tensor<double, 4> gl_cpp_fit = cpp_sampling.fit(gtau_cpp, dim);

        // Set up parameters for evaluation
        int *dims = dims_list[dim];
        int target_dim = dim;

        // Evaluate using C API
        int evaluate_status = spir_sampling_eval_dd(
            sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
            gl_cpp.data(), evaluate_output);

        REQUIRE(evaluate_status == 0);

        int fit_status =
            spir_sampling_fit_dd(sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims,
                                 target_dim, evaluate_output, fit_output);

        REQUIRE(fit_status == 0);
        // Compare with C++ implementation
        for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
            REQUIRE(evaluate_output[i] == gtau_cpp(i));
            // TODO: fix this
            REQUIRE(gl_cpp_fit(i) == Approx(gl_cpp(i)));
            REQUIRE(fit_output[i] == gl_cpp_fit(i));
        }
    }

    // Clean up
    spir_sampling_release(sampling);
    spir_basis_release(basis);
    free(evaluate_output);
    free(fit_output);
}

template <typename S>
void test_tau_sampling_evaluation_4d_column_major_complex()
{
    double beta = 1.0;
    double wmax = 10.0;

    auto stat = get_stat<S>();

    // Create basis
    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-10, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    // Create sampling
    spir_sampling *sampling = create_tau_sampling<S>(basis);
    REQUIRE(sampling != nullptr);

    // Test getting number of sampling points
    int n_points;
    int points_status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
    std::cout << "n_points" << n_points << std::endl;
    REQUIRE(n_points > 0);

    // Create equivalent C++ objects for comparison
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, 1e-10);
    sparseir::TauSampling<S> cpp_sampling(cpp_basis);

    int basis_size = cpp_basis.size();

    int d1 = 2;
    int d2 = 3;
    int d3 = 4;
    Eigen::Tensor<std::complex<double>, 4> rhol_tensor(basis_size, d1, d2, d3);

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 1);

    // Fill rhol_tensor with random complex values
    for (int i = 0; i < rhol_tensor.size(); ++i) {
        rhol_tensor.data()[i] = std::complex<double>(dis(gen), dis(gen));
    }

    c_complex *evaluate_output =
        (c_complex *)malloc(n_points * d1 * d2 * d3 * sizeof(c_complex));
    c_complex *fit_output =
        (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

    int ndim = 4;
    int dims1[4] = {basis_size, d1, d2, d3};
    int dims2[4] = {d1, basis_size, d2, d3};
    int dims3[4] = {d1, d2, basis_size, d3};
    int dims4[4] = {d1, d2, d3, basis_size};

    std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

    // Test evaluate() and fit() along each dimension
    for (int dim = 0; dim < 4; ++dim) {
        // Move the "frequency" dimension around
        Eigen::Tensor<std::complex<double>, 4> gl_cpp =
            sparseir::movedim(rhol_tensor, 0, dim);
        c_complex *evaluate_input =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
        for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
            evaluate_input[i] =
                (c_complex){gl_cpp.data()[i].real(), gl_cpp.data()[i].imag()};
        }
        // Evaluate from real-time/tau to imaginary-time/tau
        Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
            cpp_sampling.evaluate(gl_cpp, dim);
        Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
            cpp_sampling.fit(gtau_cpp, dim);
        // Set up parameters for evaluation
        int *dims = dims_list[dim];
        int target_dim = dim;

        // Evaluate using C API
        int evaluate_status = spir_sampling_eval_zz(
            sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
            evaluate_input, evaluate_output);
        REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

        int fit_status =
            spir_sampling_fit_zz(sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims,
                                 target_dim, evaluate_output, fit_output);
        REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

        // Compare with C++ implementation
        for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
            REQUIRE(__real__ evaluate_output[i] == Approx(gtau_cpp(i).real()));
            REQUIRE(__imag__ evaluate_output[i] == Approx(gtau_cpp(i).imag()));
            REQUIRE(__real__ fit_output[i] == Approx(gl_cpp_fit(i).real()));
            REQUIRE(__imag__ fit_output[i] == Approx(gl_cpp_fit(i).imag()));
        }
        free(evaluate_input);
    }

    // Clean up
    spir_sampling_release(sampling);
    spir_basis_release(basis);
    free(evaluate_output);
    free(fit_output);
}

TEST_CASE("TauSampling", "[cinterface]")
{
    SECTION("TauSampling Constructor (fermionic)")
    {
        test_tau_sampling<sparseir::Fermionic>();
    }

    SECTION("TauSampling Constructor (bosonic)")
    {
        test_tau_sampling<sparseir::Bosonic>();
    }

    SECTION("TauSampling Evaluation 1-dimensional input COLUMN-MAJOR")
    {
        test_tau_sampling_evaluation_1d_column_major<sparseir::Fermionic>();
    }

    SECTION("TauSampling Evaluation 4-dimensional input ROW-MAJOR")
    {
        test_tau_sampling_evaluation_4d_row_major<sparseir::Fermionic>();
    }

    SECTION(
        "TauSampling Evaluation 4-dimensional complex input/output ROW-MAJOR")
    {
        test_tau_sampling_evaluation_4d_row_major_complex<
            sparseir::Fermionic>();
    }

    SECTION("TauSampling Evaluation 4-dimensional input COLUMN-MAJOR")
    {
        test_tau_sampling_evaluation_4d_column_major<sparseir::Fermionic>();
    }

    SECTION("TauSampling Evaluation 4-dimensional complex input/output "
            "COLUMN-MAJOR")
    {
        test_tau_sampling_evaluation_4d_column_major_complex<
            sparseir::Fermionic>();
    }

    SECTION("TauSampling Error Status", "[cinterface]")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        int basis_status;
        spir_basis *basis = _spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, wmax, 1e-10, &basis_status);
        REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = create_tau_sampling<sparseir::Fermionic>(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int points_status = spir_sampling_get_npoints(sampling, &n_points);
        REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        // Fill rhol_tensor with random complex values
        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = dis(gen);
        }

        c_complex *output_complex =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

        double *output_double =
            (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));
        double *fit_output_double =
            (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));
        c_complex *fit_output_complex =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            Eigen::Tensor<double, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);

            // Set up parameters for evaluation
            //int *dims = dims_list[dim];
            int target_dim = dim;

            //int status_not_supported = spir_sampling_eval_dd(
                //sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                //gl_cpp.data(), output_double);
            //REQUIRE(status_not_supported == SPIR_NOT_SUPPORTED);

            //int fit_status_not_supported = spir_sampling_fit_dd(
                //sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                //output_double, fit_output_double);
            //REQUIRE(fit_status_not_supported == SPIR_NOT_SUPPORTED);

            if (dim == 0) {
                continue;
            }

            // Evaluate using C API that has dimension mismatch
            int status_dimension_mismatch = spir_sampling_eval_dd(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims1, target_dim,
                gl_cpp.data(), output_double);
            REQUIRE(status_dimension_mismatch == SPIR_INPUT_DIMENSION_MISMATCH);

            int fit_status_dimension_mismatch = spir_sampling_fit_zz(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims1, target_dim,
                output_complex, fit_output_complex);

            REQUIRE(fit_status_dimension_mismatch ==
                    SPIR_INPUT_DIMENSION_MISMATCH);
        }

        // Clean up
        spir_sampling_release(sampling);
        spir_basis_release(basis);
        free(output_complex);
        free(output_double);
        free(fit_output_double);
        free(fit_output_complex);
    }
}

template <typename S>
void test_matsubara_sampling_constructor()
{
    double beta = 1.0;
    double wmax = 10.0;
    int status;

    auto stat = get_stat<S>();

    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-10, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    int sampling_status;
    int n_points_org;
    status = spir_basis_get_n_default_matsus(basis, false, &n_points_org);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points_org > 0);

    std::vector<int64_t> smpl_points_org(n_points_org);
    status = spir_basis_get_default_matsus(basis, false, smpl_points_org.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    spir_sampling *sampling = spir_matsu_sampling_new(basis, false, n_points_org, smpl_points_org.data(), &sampling_status);
    REQUIRE(sampling_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(sampling != nullptr);

    double cond_num;
    int cond_num_status = spir_sampling_get_cond_num(sampling, &cond_num);
    REQUIRE(cond_num_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(cond_num > 1.0);

    int n_points_positive_only_org;
    status = spir_basis_get_n_default_matsus(basis, true, &n_points_positive_only_org);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points_positive_only_org > 0);

    std::vector<int64_t> smpl_points_positive_only_org(n_points_positive_only_org);
    status = spir_basis_get_default_matsus(basis, true, smpl_points_positive_only_org.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    spir_sampling *sampling_positive_only = spir_matsu_sampling_new(basis, true, n_points_positive_only_org, smpl_points_positive_only_org.data(), &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(sampling_positive_only != nullptr);

    int n_points;
    status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points > 0);

    int n_points_positive_only;
    status = spir_sampling_get_npoints(sampling_positive_only, &n_points_positive_only);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points_positive_only > 0);

    std::vector<int64_t> smpl_points(n_points);
    status = spir_sampling_get_matsus(sampling, smpl_points.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    // compare with smpl_points_org
    for (int i = 0; i < n_points; ++i) {
        REQUIRE(smpl_points[i] == smpl_points_org[i]);
    }

    std::vector<int64_t> smpl_points_positive_only(n_points_positive_only);
    status = spir_sampling_get_matsus(sampling_positive_only, smpl_points_positive_only.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    // compare with smpl_points_positive_only_org
    for (int i = 0; i < n_points_positive_only; ++i) {
        REQUIRE(smpl_points_positive_only[i] == smpl_points_positive_only_org[i]);
    }

    int n_matsubara_points_returned_ext;
    int L = n_points_org;

    status = spir_basis_get_n_default_matsus_ext(basis, false, L, &n_matsubara_points_returned_ext);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    int64_t *smpl_points_ext = (int64_t *)malloc(n_matsubara_points_returned_ext * sizeof(int64_t));
    status = spir_basis_get_default_matsus_ext(basis, false, L, smpl_points_ext, &n_matsubara_points_returned_ext);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    status = spir_basis_get_n_default_matsus_ext(basis, true, (int)(L / 2), &n_matsubara_points_returned_ext);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    int64_t *smpl_points_ext_positive_only = (int64_t *)malloc(n_matsubara_points_returned_ext * sizeof(int64_t));
    status = spir_basis_get_default_matsus_ext(basis, true, (int)(L / 2), smpl_points_ext_positive_only, &n_matsubara_points_returned_ext);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    for (int i = 0; i < n_matsubara_points_returned_ext; ++i) {
        REQUIRE(smpl_points_ext_positive_only[i] >= 0);
    }

    // Clean up
    spir_sampling_release(sampling);
    spir_sampling_release(sampling_positive_only);
    spir_basis_release(basis);
}

template <typename S>
void test_matsubara_sampling_evaluation_4d_column_major(bool positive_only)
{
    double beta = 1.0;
    double wmax = 10.0;

    auto stat = get_stat<S>();

    // Create basis
    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-10, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    // Create sampling
    spir_sampling *sampling = create_matsubara_sampling<S>(basis, positive_only);
    REQUIRE(sampling != nullptr);

    // Test getting number of sampling points
    int n_points;
    int points_status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
    std::cout << "n_points " << n_points << std::endl;
    REQUIRE(n_points > 0);


    // Create equivalent C++ objects for comparison
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, 1e-10);
    sparseir::MatsubaraSampling<S> cpp_sampling(cpp_basis, positive_only);

    int basis_size = cpp_basis.size();

    int d1 = 2;
    int d2 = 3;
    int d3 = 4;
    Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 1);

    // Fill rhol_tensor with random complex values
    for (int i = 0; i < rhol_tensor.size(); ++i) {
        rhol_tensor.data()[i] = dis(gen);
    }

    c_complex *evaluate_output =
        (c_complex *)malloc(n_points * d1 * d2 * d3 * sizeof(c_complex));
    c_complex *fit_output =
        (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

    int ndim = 4;

    int dims1[4] = {basis_size, d1, d2, d3};
    int dims2[4] = {d1, basis_size, d2, d3};
    int dims3[4] = {d1, d2, basis_size, d3};
    int dims4[4] = {d1, d2, d3, basis_size};
    std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

    int dims1_smpl[4] = {n_points, d1, d2, d3};
    int dims2_smpl[4] = {d1, n_points, d2, d3};
    int dims3_smpl[4] = {d1, d2, n_points, d3};
    int dims4_smpl[4] = {d1, d2, d3, n_points};
    std::vector<int *> dims_list_smpl = {dims1_smpl, dims2_smpl, dims3_smpl, dims4_smpl};

    // Test evaluate() and fit() along each dimension
    for (int dim = 0; dim < 4; ++dim) {
        // Move the "frequency" dimension around
        Eigen::Tensor<double, 4> gl_cpp =
            sparseir::movedim(rhol_tensor, 0, dim);

        // Evaluate from real-time/tau to imaginary-time/tau
        Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
            cpp_sampling.evaluate(gl_cpp, dim);
        Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
            cpp_sampling.fit(gtau_cpp, dim);
        // Set up parameters for evaluation
        int *dims = dims_list[dim];
        int *dims_smpl = dims_list_smpl[dim];
        int target_dim = dim;

        // Evaluate using C API
        int evaluate_status = spir_sampling_eval_dz(
            sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
            gl_cpp.data(), evaluate_output);
        REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

        int fit_status =
            spir_sampling_fit_zz(sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims_smpl,
                                 target_dim, evaluate_output, fit_output);
        REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

        // Compare with C++ implementation
        for (int i = 0; i < n_points * d1 * d2 * d3; ++i) {
            REQUIRE(__real__ evaluate_output[i] == Approx(gtau_cpp(i).real()));
            REQUIRE(__imag__ evaluate_output[i] == Approx(gtau_cpp(i).imag()));
        }

        for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
            REQUIRE(__real__ fit_output[i] == Approx(gl_cpp_fit(i).real()));
            REQUIRE(__imag__ fit_output[i] == Approx(gl_cpp_fit(i).imag()));
        }
    }

    // Clean up
    spir_sampling_release(sampling);
    spir_basis_release(basis);
    free(evaluate_output);
    free(fit_output);
}

template <typename S>
void test_matsubara_sampling_evaluation_4d_column_major_complex()
{
    double beta = 1.0;
    double wmax = 10.0;
    bool positive_only = false;
    auto stat = get_stat<S>();

    // Create basis
    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-10, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    // Create sampling
    spir_sampling *sampling = create_matsubara_sampling<S>(basis, positive_only);
    REQUIRE(sampling != nullptr);

    // Test getting number of sampling points
    int n_points;
    int points_status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points > 0);

    // Create equivalent C++ objects for comparison
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, 1e-10);
    sparseir::MatsubaraSampling<S> cpp_sampling(cpp_basis);

    int basis_size = cpp_basis.size();

    int d1 = 2;
    int d2 = 3;
    int d3 = 4;
    Eigen::Tensor<std::complex<double>, 4> rhol_tensor(basis_size, d1, d2, d3);

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 1);

    // Fill rhol_tensor with random complex values
    for (int i = 0; i < rhol_tensor.size(); ++i) {
        rhol_tensor.data()[i] = std::complex<double>(dis(gen), dis(gen));
    }

    c_complex *evaluate_output =
        (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
    c_complex *fit_output =
        (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

    int ndim = 4;
    int dims1[4] = {basis_size, d1, d2, d3};
    int dims2[4] = {d1, basis_size, d2, d3};
    int dims3[4] = {d1, d2, basis_size, d3};
    int dims4[4] = {d1, d2, d3, basis_size};

    std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

    // Test evaluate() and fit() along each dimension
    for (int dim = 0; dim < 4; ++dim) {
        // Move the "frequency" dimension around
        Eigen::Tensor<std::complex<double>, 4> gl_cpp =
            sparseir::movedim(rhol_tensor, 0, dim);
        // Evaluate from real-time/tau to imaginary-time/tau
        Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
            cpp_sampling.evaluate(gl_cpp, dim);
        Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
            cpp_sampling.fit(gtau_cpp, dim);
        // Set up parameters for evaluation
        int *dims = dims_list[dim];
        int target_dim = dim;

        c_complex *evaluate_input =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
        for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
            evaluate_input[i] =
                (c_complex){gl_cpp.data()[i].real(), gl_cpp.data()[i].imag()};
        }
        // Evaluate using C API
        int evaluate_status = spir_sampling_eval_zz(
            sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
            evaluate_input, evaluate_output);
        REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

        int fit_status =
            spir_sampling_fit_zz(sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims,
                                 target_dim, evaluate_output, fit_output);
        REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

        // Compare with C++ implementation
        for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
            REQUIRE(__real__ evaluate_output[i] == Approx(gtau_cpp(i).real()));
            REQUIRE(__imag__ evaluate_output[i] == Approx(gtau_cpp(i).imag()));
            REQUIRE(__real__ fit_output[i] == Approx(gl_cpp_fit(i).real()));
            REQUIRE(__imag__ fit_output[i] == Approx(gl_cpp_fit(i).imag()));
        }
        free(evaluate_input);
    }

    // Clean up
    spir_sampling_release(sampling);
    spir_basis_release(basis);
    free(evaluate_output);
    free(fit_output);
}

template <typename S>
void test_matsubara_sampling_evaluation_4d_row_major()
{
    double beta = 1.0;
    double wmax = 10.0;
    bool positive_only = false;
    auto stat = get_stat<S>();

    // Create basis
    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-10, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    // Create sampling
    spir_sampling *sampling = create_matsubara_sampling<S>(basis, positive_only);
    REQUIRE(sampling != nullptr);

    // Test getting number of sampling points
    int n_points;
    int points_status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points > 0);

    // Create equivalent C++ objects for comparison
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, 1e-10);
    sparseir::MatsubaraSampling<S> cpp_sampling(cpp_basis);

    int basis_size = cpp_basis.size();

    int d1 = 2;
    int d2 = 3;
    int d3 = 4;
    Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 1);

    for (int i = 0; i < rhol_tensor.size(); ++i) {
        rhol_tensor.data()[i] = dis(gen);
    }

    c_complex *evaluate_output =
        (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
    c_complex *fit_output =
        (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

    int ndim = 4;
    int dims1[4] = {basis_size, d1, d2, d3};
    int dims2[4] = {d1, basis_size, d2, d3};
    int dims3[4] = {d1, d2, basis_size, d3};
    int dims4[4] = {d1, d2, d3, basis_size};

    std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

    // Test evaluate() and fit() along each dimension
    for (int dim = 0; dim < 4; ++dim) {
        // Move the "frequency" dimension around
        Eigen::Tensor<double, 4> gl_cpp =
            sparseir::movedim(rhol_tensor, 0, dim);

        // Evaluate from real-time/tau to imaginary-time/tau
        Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
            cpp_sampling.evaluate(gl_cpp, dim);
        Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
            cpp_sampling.fit(gtau_cpp, dim);
        int *dims = dims_list[dim];
        int target_dim = dim;

        // Note that we need to specify Eigen::RowMajor here
        // because Eigen::Tensor<T, 4> is column-major by default
        Eigen::Tensor<double, 4, Eigen::RowMajor> gl_cpp_rowmajor(
            dims[0], dims[1], dims[2], dims[3]);

        // Fill row-major tensor in the correct order
        for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
            for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                    for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                        gl_cpp_rowmajor(i, j, k, l) = gl_cpp(i, j, k, l);
                    }
                }
            }
        }
        // Evaluate using C API
        int evaluate_status = spir_sampling_eval_dz(
            sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
            gl_cpp_rowmajor.data(), evaluate_output);
        REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

        int fit_status =
            spir_sampling_fit_zz(sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims,
                                 target_dim, evaluate_output, fit_output);
        REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

        // Compare results
        // Note that we need to specify Eigen::RowMajor here
        // because Eigen::Tensor<T, 4> is column-major by default
        Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> output_tensor(
            dims[0], dims[1], dims[2], dims[3]);
        Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> fit_tensor(
            dims[0], dims[1], dims[2], dims[3]);
        for (int i = 0; i < output_tensor.size(); ++i) {
            // store output data to output_tensor
            output_tensor.data()[i] = std::complex<double>(
                __real__ evaluate_output[i], __imag__ evaluate_output[i]);
            fit_tensor.data()[i] = std::complex<double>(__real__ fit_output[i],
                                                        __imag__ fit_output[i]);
        }
        // Compare results
        for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
            for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                    for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                        REQUIRE(gtau_cpp(i, j, k, l) ==
                                output_tensor(i, j, k, l));
                        REQUIRE(gl_cpp_fit(i, j, k, l) ==
                                fit_tensor(i, j, k, l));
                    }
                }
            }
        }
    }

    // Clean up
    spir_sampling_release(sampling);
    spir_basis_release(basis);
    free(evaluate_output);
    free(fit_output);
}

template <typename S>
void test_matsubara_sampling_evaluation_4d_row_major_complex()
{
    double beta = 1.0;
    double wmax = 10.0;
    bool positive_only = false;

    auto stat = get_stat<S>();

    // Create basis
    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-10, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    // Create sampling
    spir_sampling *sampling = create_matsubara_sampling<S>(basis, positive_only);
    REQUIRE(sampling != nullptr);

    // Test getting number of sampling points
    int n_points;
    int points_status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points > 0);

    // Create equivalent C++ objects for comparison
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, 1e-10);
    sparseir::MatsubaraSampling<S> cpp_sampling(cpp_basis);

    int basis_size = cpp_basis.size();

    int d1 = 2;
    int d2 = 3;
    int d3 = 4;
    Eigen::Tensor<std::complex<double>, 4> rhol_tensor(basis_size, d1, d2, d3);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 1);

    for (int i = 0; i < rhol_tensor.size(); ++i) {
        rhol_tensor.data()[i] = std::complex<double>(dis(gen), dis(gen));
    }

    c_complex *evaluate_output =
        (c_complex *)malloc(n_points * d1 * d2 * d3 * sizeof(c_complex));
    c_complex *fit_output =
        (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

    int ndim = 4;
    int dims1[4] = {basis_size, d1, d2, d3};
    int dims2[4] = {d1, basis_size, d2, d3};
    int dims3[4] = {d1, d2, basis_size, d3};
    int dims4[4] = {d1, d2, d3, basis_size};

    std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

    // Test evaluate() and fit() along each dimension
    for (int dim = 0; dim < 4; ++dim) {
        // Move the "frequency" dimension around
        Eigen::Tensor<std::complex<double>, 4> gl_cpp =
            sparseir::movedim(rhol_tensor, 0, dim);
        // Evaluate from real-time/tau to imaginary-time/tau
        Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
            cpp_sampling.evaluate(gl_cpp, dim);
        Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
            cpp_sampling.fit(gtau_cpp, dim);
        int *dims = dims_list[dim];
        int target_dim = dim;

        // Note that we need to specify Eigen::RowMajor here
        // because Eigen::Tensor<double, 4> is column-major by default
        Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> gl_cpp_rowmajor(
            dims[0], dims[1], dims[2], dims[3]);

        // Fill row-major tensor in the correct order
        for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
            for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                    for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                        gl_cpp_rowmajor(i, j, k, l) = gl_cpp(i, j, k, l);
                    }
                }
            }
        }
        c_complex *evaluate_input =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
        for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
            evaluate_input[i] = (c_complex){gl_cpp_rowmajor.data()[i].real(),
                                            gl_cpp_rowmajor.data()[i].imag()};
        }
        // Evaluate using C API
        int evaluate_status = spir_sampling_eval_zz(
            sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
            evaluate_input, evaluate_output);
        REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

        int fit_status =
            spir_sampling_fit_zz(sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims,
                                 target_dim, evaluate_output, fit_output);
        REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

        // Compare results
        // Note that we need to specify Eigen::RowMajor here
        // because Eigen::Tensor<T, 4> is column-major by default
        Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> output_tensor(
            dims[0], dims[1], dims[2], dims[3]);
        Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> fit_tensor(
            dims[0], dims[1], dims[2], dims[3]);
        for (int i = 0; i < output_tensor.size(); ++i) {
            // store output data to output_tensor
            output_tensor.data()[i] = std::complex<double>(
                __real__ evaluate_output[i], __imag__ evaluate_output[i]);
            fit_tensor.data()[i] = std::complex<double>(__real__ fit_output[i],
                                                        __imag__ fit_output[i]);
        }
        // Compare results
        for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
            for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                    for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                        REQUIRE(gtau_cpp(i, j, k, l) ==
                                output_tensor(i, j, k, l));
                        REQUIRE(gl_cpp_fit(i, j, k, l) ==
                                fit_tensor(i, j, k, l));
                    }
                }
            }
        }
        free(evaluate_input);
    }

    // Clean up
    spir_sampling_release(sampling);
    spir_basis_release(basis);
    free(evaluate_output);
    free(fit_output);
}

template <typename S>
void test_matsubara_sampling_error_status()
{
    double beta = 1.0;
    double wmax = 10.0;
    bool positive_only = false;

    auto stat = get_stat<S>();

    // Create basis
    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, 1e-10, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    // Create sampling
    spir_sampling *sampling = create_matsubara_sampling<S>(basis, positive_only);
    REQUIRE(sampling != nullptr);

    // Test getting number of sampling points
    int n_points;
    int points_status = spir_sampling_get_npoints(sampling, &n_points);
    REQUIRE(points_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_points > 0);

    // Create equivalent C++ objects for comparison
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, 1e-10);
    sparseir::MatsubaraSampling<S> cpp_sampling(cpp_basis);

    int basis_size = cpp_basis.size();

    int d1 = 2;
    int d2 = 3;
    int d3 = 4;
    Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 1);

    // Fill rhol_tensor with random complex values
    for (int i = 0; i < rhol_tensor.size(); ++i) {
        rhol_tensor.data()[i] = dis(gen);
    }

    c_complex *output_complex =
        (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
    double *output_double =
        (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));
    double *fit_output_double =
        (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));
    c_complex *fit_output_complex =
        (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

    int ndim = 4;
    int dims1[4] = {basis_size, d1, d2, d3};
    int dims2[4] = {d1, basis_size, d2, d3};
    int dims3[4] = {d1, d2, basis_size, d3};
    int dims4[4] = {d1, d2, d3, basis_size};

    std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

    // Test evaluate() and fit() along each dimension
    for (int dim = 0; dim < 4; ++dim) {
        // Move the "frequency" dimension around
        Eigen::Tensor<double, 4> gl_cpp =
            sparseir::movedim(rhol_tensor, 0, dim);

        // Set up parameters for evaluation
        int *dims = dims_list[dim];
        int target_dim = dim;

        // Evaluate using C API that is not supported
        int status_not_supported = spir_sampling_eval_dd(
            sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
            gl_cpp.data(), output_double);
        REQUIRE(status_not_supported == SPIR_NOT_SUPPORTED);

        int fit_status_not_supported =
            spir_sampling_fit_dd(sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims,
                                 target_dim, output_double, fit_output_double);
        REQUIRE(fit_status_not_supported == SPIR_NOT_SUPPORTED);

        if (dim == 0) {
            continue;
        }

        // Evaluate using C API that has dimension mismatch
        int status_dimension_mismatch = spir_sampling_eval_dz(
            sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims1, target_dim,
            gl_cpp.data(), output_complex);
        REQUIRE(status_dimension_mismatch == SPIR_INPUT_DIMENSION_MISMATCH);

        int fit_status_dimension_mismatch = spir_sampling_fit_zz(
            sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims1, target_dim,
            output_complex, fit_output_complex);
        REQUIRE(fit_status_dimension_mismatch == SPIR_INPUT_DIMENSION_MISMATCH);
    }

    // Clean up
    spir_sampling_release(sampling);
    spir_basis_release(basis);
    free(output_complex);
    free(output_double);
    free(fit_output_double);
    free(fit_output_complex);
}


TEST_CASE("MatsubaraSampling", "[cinterface]")
{
    SECTION("MatsubaraSampling Constructor (fermionic)")
    {
        test_matsubara_sampling_constructor<sparseir::Fermionic>();
    }

    SECTION("MatsubaraSampling Constructor (bosonic)")
    {
        test_matsubara_sampling_constructor<sparseir::Bosonic>();
    }

    /*
    SECTION("MatsubaraSampling Evaluation 4-dimensional input COLUMN-MAJOR "
            "(fermionic)")
    {
        test_matsubara_sampling_evaluation_4d_column_major<
            sparseir::Fermionic>();
    }
    */

    SECTION("MatsubaraSampling Evaluation 4-dimensional input COLUMN-MAJOR "
            "(bosonic)")
    {
        test_matsubara_sampling_evaluation_4d_column_major<sparseir::Bosonic>(false);
        test_matsubara_sampling_evaluation_4d_column_major<sparseir::Bosonic>(true);
    }

    /*
    SECTION("MatsubaraSampling Evaluation 4-dimensional complex input "
            "COLUMN-MAJOR (fermionic)")
    {
        test_matsubara_sampling_evaluation_4d_column_major_complex<
            sparseir::Fermionic>();
    }

    SECTION("MatsubaraSampling Evaluation 4-dimensional complex input "
            "COLUMN-MAJOR (bosonic)")
    {
        test_matsubara_sampling_evaluation_4d_column_major_complex<
            sparseir::Bosonic>();
    }

    SECTION("MatsubaraSampling Evaluation 4-dimensional input ROW-MAJOR "
            "(fermionic)")
    {
        test_matsubara_sampling_evaluation_4d_row_major<sparseir::Fermionic>();
    }

    SECTION(
        "MatsubaraSampling Evaluation 4-dimensional input ROW-MAJOR (bosonic)")
    {
        test_matsubara_sampling_evaluation_4d_row_major<sparseir::Bosonic>();
    }

    SECTION("MatsubaraSampling Evaluation 4-dimensional complex input "
            "ROW-MAJOR (fermionic)")
    {
        test_matsubara_sampling_evaluation_4d_row_major_complex<
            sparseir::Fermionic>();
    }

    SECTION("MatsubaraSampling Evaluation 4-dimensional complex input "
            "ROW-MAJOR (bosonic)")
    {
        test_matsubara_sampling_evaluation_4d_row_major_complex<
            sparseir::Bosonic>();
    }

    SECTION("MatsubaraSampling Error Status (fermionic)")
    {
        test_matsubara_sampling_error_status<sparseir::Fermionic>();
    }

    SECTION("MatsubaraSampling Error Status (bosonic)")
    {
        test_matsubara_sampling_error_status<sparseir::Bosonic>();
    }
    */
}