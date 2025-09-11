#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir.hpp>

using std::invalid_argument;

TEST_CASE("freq", "[freq]")
{
    REQUIRE(sparseir::create_statistics(0)->zeta() == 0); // Bosonic
    REQUIRE(sparseir::create_statistics(1)->zeta() == 1); // Fermionic
}

TEST_CASE("Integer value of FermionicFreq and BosonicFreq", "[freq]")
{
    sparseir::FermionicFreq f(3);
    sparseir::BosonicFreq b(-2);
    REQUIRE(f.get_n() == 3);
    REQUIRE(b.get_n() == -2);
}

TEST_CASE("Exceptions for invalid FermionicFreq and BosonicFreq values",
          "[freq]")
{
    REQUIRE_THROWS_AS(sparseir::FermionicFreq(4), std::domain_error);
    REQUIRE_THROWS_AS(sparseir::BosonicFreq(-7), std::domain_error);
}

TEST_CASE("Comparison operators for FermionicFreq and BosonicFreq", "[freq]")
{
    sparseir::FermionicFreq f(5);
    sparseir::BosonicFreq b(6);

    REQUIRE(f.get_n() < b.get_n());
    REQUIRE(b.get_n() >= b.get_n());
}

TEST_CASE("Value calculation for MatsubaraFreq", "[freq]")
{
    double beta = 3.0;
    sparseir::FermionicFreq bf(1);
    double expected_value = M_PI / beta;
    REQUIRE(bf.value(beta) == expected_value);
}

TEST_CASE("Imaginary value calculation for MatsubaraFreq", "[freq]")
{
    double beta = 3.0;
    sparseir::BosonicFreq bf(2);
    std::complex<double> expected_valueim(0, 2 * M_PI / beta);
    // std::cout << bf.valueim(beta) << std::endl;
    REQUIRE(bf.valueim(beta) == expected_valueim);
}

TEST_CASE("iszero check for MatsubaraFreq", "[freq]")
{
    sparseir::FermionicFreq f(-3);
    REQUIRE(!is_zero(f));

    sparseir::BosonicFreq bf(0);
    REQUIRE(is_zero(bf));

    sparseir::BosonicFreq bf_nonzero(2);
    REQUIRE(!is_zero(bf_nonzero));
}
