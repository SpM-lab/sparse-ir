#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <cstdint>
#include <iostream>
#include <sparseir/sparseir.hpp>
#include <vector>
#include <xprec/ddouble-header-only.hpp>

using Catch::Approx;
using xprec::DDouble;

TEST_CASE("legendre", "[specfuncs]")
{
    SECTION("legval")
    {
        std::vector<double> c = {1.0, 2.0, 3.0};
        double x = 0.5;
        double result = sparseir::legval(x, c);
        REQUIRE(result == 1.625);
    }

    SECTION("legvander")
    {
        std::vector<double> x = {0.0, 0.5, 1.0};
        int deg = 2;
        Eigen::MatrixXd result = sparseir::legvander<double>(x, deg);
        Eigen::MatrixXd expected(3, 3);
        expected << 1, 0, -0.5, 1.0, 0.5, -0.125, 1, 1, 1;
        REQUIRE(result.isApprox(expected, 1e-9));

        Eigen::VectorXd x_eigen =
            Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
        Eigen::MatrixXd result_eigen = sparseir::legvander<double>(x_eigen, deg);
        REQUIRE(result_eigen.isApprox(expected, 1e-9));
    }
}

TEST_CASE("bessel", "[specfuncs]")
{
    // Reference values from Julia
    // julia> using Bessels
    // julia> for i in 0:15; println(sphericalbesselj(i, 1.)); end
    std::vector<double> refs = {
        0.8414709848078965,     0.30116867893975674,   0.06203505201137386,
        0.009006581117112517,   0.0010110158084137527, 9.256115861125818e-5,
        7.156936310087086e-6,   4.790134198739489e-7,  2.82649880221473e-8,
        1.4913765025551456e-9,  7.116552640047314e-11, 3.09955185479008e-12,
        1.2416625969871055e-13, 4.604637677683788e-15, 1.5895759875169764e-16,
        5.1326861154437626e-18};
    double x = 1.0;
    for (int l = 0; l < static_cast<int>(refs.size()); ++l) {
        double result = sparseir::sphericalbesselj(l, x);
        REQUIRE(result == Approx(refs[l]));
    }
}