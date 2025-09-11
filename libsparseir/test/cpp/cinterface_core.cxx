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

TEST_CASE("Kernel Accuracy Tests", "[cinterface]")
{
    // Test individual kernels
    SECTION("LogisticKernel(9)")
    {
        auto cpp_kernel = sparseir::LogisticKernel(9);
        int status;
        spir_kernel *kernel = spir_logistic_kernel_new(9, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(kernel != nullptr);
    }

    SECTION("RegularizedBoseKernel(10)")
    {
        auto cpp_kernel = sparseir::RegularizedBoseKernel(10);
        int status;
        spir_kernel *kernel = spir_reg_bose_kernel_new(10, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(kernel != nullptr);
    }

    SECTION("Kernel Domain")
    {
        // Create a kernel through C API
        int kernel_status;
        spir_kernel *kernel = spir_logistic_kernel_new(9, &kernel_status);
        REQUIRE(kernel_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(kernel != nullptr);

        // Get domain bounds
        double xmin, xmax, ymin, ymax;
        int domain_status = spir_kernel_domain(kernel, &xmin, &xmax, &ymin, &ymax);
        REQUIRE(domain_status == SPIR_COMPUTATION_SUCCESS);

        // Compare with C++ implementation
        auto cpp_kernel = sparseir::LogisticKernel(9);
        auto xrange = cpp_kernel.xrange();
        auto yrange = cpp_kernel.yrange();
        auto cpp_xmin = xrange.first;
        auto cpp_xmax = xrange.second;
        auto cpp_ymin = yrange.first;
        auto cpp_ymax = yrange.second;

        REQUIRE(xmin == cpp_xmin);
        REQUIRE(xmax == cpp_xmax);
        REQUIRE(ymin == cpp_ymin);
        REQUIRE(ymax == cpp_ymax);

        // Clean up
        spir_kernel_release(kernel);
    }
}

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
void test_finite_temp_basis_constructor()
{
    double beta = 2.0;
    double wmax = 5.0;
    double epsilon = 1e-6;

    auto stat = get_stat<S>();

    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, epsilon);
    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, epsilon, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    int basis_size;
    int size_status = spir_basis_get_size(basis, &basis_size);
    REQUIRE(size_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis_size == cpp_basis.size());
}

template <typename S>
void test_finite_temp_basis_constructor_with_sve()
{
    double beta = 2.0;
    double wmax = 5.0;
    double Lambda = 10.0;
    double epsilon = 1e-6;

    int kernel_status;
    spir_kernel *kernel = spir_logistic_kernel_new(Lambda, &kernel_status);
    REQUIRE(kernel_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);

    int sve_status;
    double cutoff = -1.0;
    int lmax = -1;
    int n_gauss = -1;
    int Twork = SPIR_TWORK_FLOAT64X2;
    spir_sve_result *sve_result = spir_sve_result_new(kernel, epsilon, cutoff, lmax, n_gauss, Twork, &sve_status);
    REQUIRE(sve_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(sve_result != nullptr);

    auto stat = get_stat<S>();

    int basis_status;
    int max_size = -1;
    spir_basis *basis = spir_basis_new(
        stat, beta, wmax, kernel, sve_result, max_size, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    int stats;
    int stats_status = spir_basis_get_stats(basis, &stats);
    REQUIRE(stats_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(stats == stat);

    // Clean up
    spir_kernel_release(kernel);
    spir_sve_result_release(sve_result);
    spir_basis_release(basis);
}

TEST_CASE("FiniteTempBasis", "[cinterface]")
{
    SECTION("FiniteTempBasis Constructor Fermionic")
    {
        test_finite_temp_basis_constructor<sparseir::Fermionic>();
    }

    SECTION("FiniteTempBasis Constructor Bosonic")
    {
        test_finite_temp_basis_constructor<sparseir::Bosonic>();
    }

    SECTION("FiniteTempBasis Constructor with SVE Fermionic/LogisticKernel")
    {
        test_finite_temp_basis_constructor_with_sve<sparseir::Fermionic>();
    }

    SECTION("FiniteTempBasis Constructor with SVE Bosonic/LogisticKernel")
    {
        test_finite_temp_basis_constructor_with_sve<sparseir::Bosonic>();
    }

    SECTION(
        "FiniteTempBasis Constructor with SVE Bosonic/RegularizedBoseKernel")
    {
        double beta = 2.0;
        double wmax = 5.0;
        double Lambda = 10.0;
        double epsilon = 1e-6;

        int kernel_status;
        spir_kernel *kernel = spir_reg_bose_kernel_new(Lambda, &kernel_status);
        REQUIRE(kernel_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(kernel != nullptr);

        int sve_status;
        double cutoff = -1.0;
        int lmax = -1;
        int n_gauss = -1;
        int Twork = SPIR_TWORK_FLOAT64X2;
        spir_sve_result *sve_result = spir_sve_result_new(kernel, epsilon, cutoff, lmax, n_gauss, Twork, &sve_status);
        REQUIRE(sve_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(sve_result != nullptr);

        int basis_status;
        int max_size = -1;
        spir_basis *basis = spir_basis_new(
            SPIR_STATISTICS_BOSONIC, beta, wmax, kernel, sve_result, max_size, &basis_status);
        REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(basis != nullptr);

        int stats;
        int stats_status = spir_basis_get_stats(basis, &stats);
        REQUIRE(stats_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(stats == SPIR_STATISTICS_BOSONIC);

        // Clean up
        spir_kernel_release(kernel);
        spir_sve_result_release(sve_result);
        spir_basis_release(basis);
    }
}

template <typename S>
void test_finite_temp_basis_basis_functions()
{
    double beta = 2.0;
    double wmax = 5.0;
    double epsilon = 1e-6;

    auto stat = get_stat<S>();

    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, epsilon, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    int basis_size;
    int size_status = spir_basis_get_size(basis, &basis_size);
    REQUIRE(size_status == SPIR_COMPUTATION_SUCCESS);

    int u_status;
    spir_funcs *u = spir_basis_get_u(basis, &u_status);
    REQUIRE(u_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(u != nullptr);

    int uhat_status;
    spir_funcs *uhat = spir_basis_get_uhat(basis, &uhat_status);
    REQUIRE(uhat_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(uhat != nullptr);


    // Test basis function evaluation
    double x = 0.5;        // Test point for u basis (imaginary time)
    double y = 0.5 * wmax; // Test point for v basis (real frequency)
    std::vector<double> out_vec(basis_size);
    int eval_status = spir_funcs_eval(u, x, out_vec.data());
    REQUIRE(eval_status == SPIR_COMPUTATION_SUCCESS);

    std::vector<int> slice_u{1, 2};
    int slice_u_status;
    std::cout << "basis_size: " << basis_size << std::endl;
    spir_funcs *u_slice = spir_funcs_get_slice(u, slice_u.size(), slice_u.data(), &slice_u_status);
    REQUIRE(slice_u_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(u_slice != nullptr);
    std::vector<double> out_vec_slice(slice_u.size());
    int eval_status_slice = spir_funcs_eval(u_slice, x, out_vec_slice.data());
    REQUIRE(eval_status_slice == SPIR_COMPUTATION_SUCCESS);
    for (int i = 0; i < slice_u.size(); ++i) {
        REQUIRE(out_vec_slice[i] == Approx(out_vec[slice_u[i]]));
    }

    // Compare with C++ implementation for u basis
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, epsilon);
    Eigen::VectorXd cpp_result = (*cpp_basis.u)(x);
    for (int i = 0; i < basis_size; ++i) {
        REQUIRE(out_vec[i] == Approx(cpp_result(i)));
    }

    // Test roots
    int nroots;
    int nroots_status = spir_funcs_get_n_roots(u, &nroots);
    REQUIRE(nroots_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(nroots == cpp_basis.u->nroots());
    std::vector<double> roots(nroots);
    int roots_status = spir_funcs_get_roots(u, roots.data());
    REQUIRE(roots_status == SPIR_COMPUTATION_SUCCESS);
    Eigen::VectorXd cpp_roots = cpp_basis.u->roots();
    for (int i = 0; i < nroots; ++i) {
        REQUIRE(roots[i] == cpp_roots(i));
    }
    int npositive_roots = std::count_if(cpp_roots.data(), cpp_roots.data() + nroots, [](double x) { return x > 0; });
    REQUIRE(npositive_roots == nroots/2);

    // Test v basis functions
    int v_status;
    spir_funcs *v = spir_basis_get_v(basis, &v_status);
    REQUIRE(v_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(v != nullptr);

    // Test roots
    nroots_status = spir_funcs_get_n_roots(v, &nroots);
    REQUIRE(nroots_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(nroots == cpp_basis.v->nroots());
    roots.resize(nroots);
    roots_status = spir_funcs_get_roots(v, roots.data());
    REQUIRE(roots_status == SPIR_COMPUTATION_SUCCESS);
    cpp_roots = cpp_basis.v->roots();
    for (int i = 0; i < nroots; ++i) {
        REQUIRE(roots[i] == cpp_roots(i));
    }

    // Test v basis function evaluation
    eval_status = spir_funcs_eval(v, y, out_vec.data());
    REQUIRE(eval_status == SPIR_COMPUTATION_SUCCESS);

    // Compare with C++ implementation for v basis
    cpp_result = (*cpp_basis.v)(y);
    for (int i = 0; i < basis_size; ++i) {
        REQUIRE(out_vec.data()[i] == Approx(cpp_result(i)));
    }

    // Test batch evaluation at multiple points
    const int num_points = 5;
    std::vector<double> xs(num_points);
    std::vector<double> batch_out_vec(num_points * basis_size);

    // Generate test points
    for (int i = 0; i < num_points; ++i) {
        xs[i] = 0.2 * (i + 1); // Points at 0.2, 0.4, 0.6, 0.8, 1.0
    }

    // Test row-major order for u basis
    int batch_status = spir_funcs_batch_eval(u, SPIR_ORDER_ROW_MAJOR, num_points, xs.data(), batch_out_vec.data());
    REQUIRE(batch_status == SPIR_COMPUTATION_SUCCESS);

    // Compare with C++ implementation for multiple points
    Eigen::VectorXd cpp_xs = Eigen::Map<Eigen::VectorXd>(xs.data(), num_points);
    Eigen::MatrixXd cpp_batch_result = (*cpp_basis.u)(cpp_xs);

    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < basis_size; ++j) {
            REQUIRE(batch_out_vec[i * basis_size + j] == Approx(cpp_batch_result(j, i)));
        }
    }

    // Test column-major order for u basis
    batch_status = spir_funcs_batch_eval(u, SPIR_ORDER_COLUMN_MAJOR, num_points, xs.data(), batch_out_vec.data());
    REQUIRE(batch_status == SPIR_COMPUTATION_SUCCESS);

    // Compare with C++ implementation for column-major order
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < basis_size; ++j) {
            REQUIRE(batch_out_vec[j * num_points + i] == Approx(cpp_batch_result(j, i)));
        }
    }

    // Test row-major order for v basis
    batch_status = spir_funcs_batch_eval(v, SPIR_ORDER_ROW_MAJOR, num_points, xs.data(), batch_out_vec.data());
    REQUIRE(batch_status == SPIR_COMPUTATION_SUCCESS);

    // Compare with C++ implementation for v basis
    cpp_batch_result = (*cpp_basis.v)(cpp_xs);
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < basis_size; ++j) {
            REQUIRE(batch_out_vec[i * basis_size + j] == Approx(cpp_batch_result(j, i)));
        }
    }

    // Test column-major order for v basis
    batch_status = spir_funcs_batch_eval(v, SPIR_ORDER_COLUMN_MAJOR, num_points, xs.data(), batch_out_vec.data());
    REQUIRE(batch_status == SPIR_COMPUTATION_SUCCESS);

    // Compare with C++ implementation for column-major order
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < basis_size; ++j) {
            REQUIRE(batch_out_vec[j * num_points + i] == Approx(cpp_batch_result(j, i)));
        }
    }

    spir_funcs_release(u);
    spir_funcs_release(v);
    spir_funcs_release(uhat);
    spir_funcs_release(u_slice);

    // Test error cases
    eval_status = spir_funcs_eval(nullptr, x, out_vec.data());
    REQUIRE(eval_status != SPIR_COMPUTATION_SUCCESS);

    eval_status = spir_funcs_eval(u, x, nullptr);
    REQUIRE(eval_status != SPIR_COMPUTATION_SUCCESS);

    // Clean up
    spir_basis_release(basis);
}

TEST_CASE("FiniteTempBasis Basis Functions", "[cinterface]")
{
    SECTION("Basis Functions Fermionic")
    {
        test_finite_temp_basis_basis_functions<sparseir::Fermionic>();
    }

    SECTION("Basis Functions Bosonic")
    {
        test_finite_temp_basis_basis_functions<sparseir::Bosonic>();
    }
}