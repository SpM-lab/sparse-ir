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


// Compression Tests
template <typename Statistics>
void test_finite_temp_basis_dlr()
{
    const double beta = 10000.0;
    const auto stat = get_stat<Statistics>();
    const double wmax = 1.0;
    const double epsilon = 1e-12;

    int status;

    int basis_status;
    spir_basis *basis = _spir_basis_new(stat, beta, wmax, epsilon, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);
    int basis_size;
    basis_status = spir_basis_get_size(basis, &basis_size);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis_size >= 0);

    // Poles
    int num_default_poles;
    status = spir_basis_get_n_default_ws(basis, &num_default_poles);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(num_default_poles >= 0);
    std::vector<double> default_poles(num_default_poles);
    status = spir_basis_get_default_ws(basis, default_poles.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // DLR constructor using the default poles
    spir_basis *dlr = spir_dlr_new(basis, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(dlr != nullptr);

    // DLR constructor using custom poles
    spir_basis *dlr_with_poles = spir_dlr_new_with_poles(basis, num_default_poles, default_poles.data(), &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(dlr_with_poles != nullptr);

    int num_poles;
    status = spir_dlr_get_npoles(dlr, &num_poles);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(num_poles == num_default_poles);

    status = spir_dlr_get_npoles(dlr_with_poles, &num_poles);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(num_poles == num_default_poles);

    std::vector<double> poles_reconst(num_poles);
    status = spir_dlr_get_poles(dlr, poles_reconst.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    for (int i = 0; i < num_poles; i++) {
        REQUIRE(poles_reconst[i] == Approx(default_poles[i]));
    }

    spir_basis_release(basis);
    spir_basis_release(dlr);
    spir_basis_release(dlr_with_poles);
}

TEST_CASE("DiscreteLehmannRepresentation", "[cinterface]")
{
    SECTION("DiscreteLehmannRepresentation Constructor Fermionic")
    {
        test_finite_temp_basis_dlr<sparseir::Fermionic>();
    }

    SECTION("DiscreteLehmannRepresentation Constructor Bosonic")
    {
        test_finite_temp_basis_dlr<sparseir::Bosonic>();
    }
}