#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include <sparseir/root.hpp>

// Include the previously implemented functions here

// template <typename F, typename T>
// std::vector<T> discrete_extrema(F f, const std::vector<T>& xgrid);

TEST_CASE("bisect")
{
    SECTION("Simple linear function")
    {
        std::function<double(double)> f_linear = [](double x) {
            return x - 0.5;
        };
        double root_linear =
            sparseir::bisect(f_linear, 0.0, 1.0, f_linear(0.0), 1e-10);
        REQUIRE(std::abs(root_linear - 0.5) < 1e-9);
    }

    SECTION("Quadratic function")
    {
        std::function<double(double)> f_quad = [](double x) {
            return x * x - 2.0;
        };
        double root_quad =
            sparseir::bisect(f_quad, 1.0, 2.0, f_quad(1.0), 1e-10);
        REQUIRE(std::abs(root_quad - std::sqrt(2.0)) < 1e-9);
    }

    SECTION("Function with multiple roots but one in interval")
    {
        std::function<double(double)> f_sin = [](double x) {
            return std::sin(x);
        };
        double root_sin = sparseir::bisect(f_sin, 3.0, 3.5, f_sin(3.0), 1e-10);
        REQUIRE(std::abs(root_sin - M_PI) < 1e-9);
    }

    SECTION("Test with integer inputs")
    {
        std::function<double(double)> f_int = [](double x) { return x - 5.; };
        double root_int = sparseir::bisect(f_int, 0., 10., f_int(0.), 1e-10);
        REQUIRE(std::abs(root_int - 5.) < 1e-10);
    }

    SECTION("Test with different epsilon values")
    {
        std::function<double(double)> f_precise = [](double x) {
            return x - M_PI;
        };
        double root_precise1 =
            sparseir::bisect(f_precise, 3.0, 4.0, f_precise(3.0), 1e-15);
        REQUIRE(std::abs(root_precise1 - M_PI) < 1e-14);

        double root_precise2 =
            sparseir::bisect(f_precise, 3.0, 4.0, f_precise(3.0), 1e-5);
        REQUIRE(std::abs(root_precise2 - M_PI) < 1e-4);
    }

    SECTION("Test with floating point edge cases")
    {
        double eps = std::numeric_limits<double>::epsilon();
        std::function<double(double)> f_small = [=](double x) {
            return x - eps;
        };
        double root_small =
            sparseir::bisect(f_small, 0.0, 1.0, f_small(0.0), eps);
        REQUIRE(std::abs(root_small - eps) < 1e-15);
    }
}

TEST_CASE("find_all")
{
    SECTION("Basic function roots")
    {
        std::vector<double> xgrid = {-2.0, -1.0, 0.0, 1.0, 2.0};

        // Simple linear function
        std::function<double(double)> linear = [](double x) { return x; };
        auto linear_roots = sparseir::find_all(linear, xgrid);
        REQUIRE(linear_roots == std::vector<double>{0.0});

        // Quadratic function
        std::function<double(double)> quadratic = [](double x) {
            return x * x - 1;
        };
        std::vector<double> xgrid2 = {-2.0, -1.2, -0.7, 0.0, 0.8, 1.1, 2.0};
        auto quad_roots = sparseir::find_all(quadratic, xgrid2);
        std::vector<double> expected_quad = {-1.0, 1.0};
        REQUIRE(quad_roots.size() == expected_quad.size());
        for (size_t i = 0; i < quad_roots.size(); ++i) {
            REQUIRE(std::abs(quad_roots[i] - expected_quad[i]) < 1e-10);
        }
    }

    SECTION("Direct hits and sign changes")
    {
        std::vector<double> xgrid = {-1.0, -0.5, 0.0, 0.5, 1.0};

        // Function with exact zeros at grid points
        std::function<double(double)> exact_zeros = [](double x) {
            return x * (x - 0.5) * (x + 0.5);
        };
        auto zeros_roots = sparseir::find_all(exact_zeros, xgrid);
        std::vector<double> expected_zeros = {-0.5, 0.0, 0.5};
        REQUIRE(zeros_roots == expected_zeros);
    }

    SECTION("No roots")
    {
        std::vector<double> xgrid = {-1.0, -0.5, 0.0, 0.5, 1.0};

        // Constant positive function
        std::function<double(double)> constant = [](double) { return 1.0; };
        auto const_roots = sparseir::find_all(constant, xgrid);
        REQUIRE(const_roots.empty());
    }

    SECTION("Edge cases")
    {
        // Empty grid
        std::vector<double> empty_grid;
        std::function<double(double)> f = [](double x) { return x; };
        auto empty_roots = sparseir::find_all(f, empty_grid);
        REQUIRE(empty_roots.empty());

        // Single point grid
        std::vector<double> single_grid = {0.0};
        auto single_roots = sparseir::find_all(f, single_grid);
        REQUIRE(single_roots == std::vector<double>{0.0});
    }

    SECTION("Multiple close roots")
    {
        std::vector<double> xgrid;
        for (double x = -1.0; x <= 1.0; x += 0.1) {
            xgrid.push_back(x);
        }

        // Function with multiple close roots
        std::function<double(double)> multi_roots = [](double x) {
            return std::sin(10 * x);
        };
        auto roots = sparseir::find_all(multi_roots, xgrid);

        // Check that each found root is actually close to zero
        for (double root : roots) {
            REQUIRE(std::abs(multi_roots(root)) < 1e-10);
        }
    }
}

TEST_CASE("midpoint")
{
    SECTION("Integer midpoints")
    {
        // REQUIRE(sparseir::midpoint(std::numeric_limits<int>::max(),
        //                 std::numeric_limits<int>::max()) ==
        //                 std::numeric_limits<int>::max());
        // REQUIRE(sparseir::midpoint(std::numeric_limits<int>::min(),
        //                 std::numeric_limits<int>::max()) == -1);
        REQUIRE(sparseir::midpoint(std::numeric_limits<int>::min(),
                                   std::numeric_limits<int>::min()) ==
                std::numeric_limits<int>::min());
        REQUIRE(sparseir::midpoint<int>(1000, 2000) == 1500);
    }

    SECTION("Floating point midpoints")
    {
        REQUIRE(sparseir::midpoint(std::numeric_limits<double>::max(),
                                   std::numeric_limits<double>::max()) ==
                std::numeric_limits<double>::max());
        REQUIRE(sparseir::midpoint(0.0, std::numeric_limits<float>::max()) ==
                std::numeric_limits<float>::max() / 2.0f);
        REQUIRE(sparseir::midpoint(-10.0, 1.0) == -4.5);
    }
}

TEST_CASE("discrete_extrema")
{
    std::vector<double> nonnegative = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<double> symmetric = {-8, -7, -6, -5, -4, -3, -2, -1, 0,
                                     1,  2,  3,  4,  5,  6,  7,  8};

    SECTION("Identity function")
    {
        std::function<double(double)> identity = [](double x) { return x; };
        auto result = sparseir::discrete_extrema(identity, nonnegative);
        REQUIRE(result == std::vector<double>{8});
    }

    SECTION("Shifted identity function")
    {
        std::function<double(double)> shifted_identity = [](double x) {
            return static_cast<double>(x) -
                   std::numeric_limits<double>::epsilon();
        };
        auto result = sparseir::discrete_extrema(shifted_identity, nonnegative);
        REQUIRE(result == std::vector<double>{0, 8});
    }

    SECTION("Square function")
    {
        std::function<double(double)> square = [](double x) { return x * x; };
        auto result = sparseir::discrete_extrema(square, symmetric);
        REQUIRE(result == std::vector<double>{-8, 0, 8});
    }

    SECTION("Constant function")
    {
        std::function<double(double)> constant = [](double) { return 1; };
        auto result = sparseir::discrete_extrema(constant, symmetric);
        REQUIRE(result.empty());
    }
}