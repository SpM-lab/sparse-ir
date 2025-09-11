#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir.hpp>
#include <xprec/ddouble-header-only.hpp>

// test_piecewise_legendre_poly.cpp

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

TEST_CASE("svd.cpp", "[svd]")
{
    using namespace Eigen;
    using namespace xprec;

    // Create a standard matrix
    Eigen::MatrixX<xprec::DDouble> mat =
        Eigen::MatrixX<xprec::DDouble>::Random(5, 6);
    auto svd_result = sparseir::compute_svd(mat, 0, "default");
    auto U = std::get<0>(svd_result);
    auto S = std::get<1>(svd_result);
    auto V = std::get<2>(svd_result);
    auto diff = (mat - U * S.asDiagonal() * V.transpose()).norm() / mat.norm();
    REQUIRE(diff < 1e-28);

    /*
    REQUIRE_THROWS_AS(compute_svd(mat, "fast"), std::domain_error);
    */
}