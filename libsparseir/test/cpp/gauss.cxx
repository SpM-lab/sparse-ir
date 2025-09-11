#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <catch2/catch_approx.hpp> // for Approx
#include <catch2/catch_test_macros.hpp>
#include <sparseir/sparseir.hpp>
#include <xprec/ddouble-header-only.hpp>

using Catch::Approx;
using std::invalid_argument;
using xprec::DDouble;

TEST_CASE("Rule constructor", "[gauss]")
{
    std::vector<DDouble> x, w;
    sparseir::Rule<DDouble> r = sparseir::Rule<DDouble>(x, w);
    REQUIRE(1 == 1);
}

template <typename T>
void gaussValidate(const sparseir::Rule<T> &rule)
{
    if (!(rule.a <= rule.b)) {
        throw invalid_argument("a,b must be a valid interval");
    }
    if (!std::all_of(rule.x.begin(), rule.x.end(),
                     [rule](T xi) { return xi <= rule.b; })) {
        throw invalid_argument("x must be smaller than b");
    }
    if (!std::all_of(rule.x.begin(), rule.x.end(),
                     [rule](T xi) { return xi >= rule.a; })) {
        throw invalid_argument("x must be larger than a");
    }
    if (!std::is_sorted(rule.x.begin(), rule.x.end())) {
        throw invalid_argument("x must be well-ordered");
    }
    if (rule.x.size() != rule.w.size()) {
        throw invalid_argument("shapes are inconsistent");
    }

    for (size_t i = 0; i < rule.x_forward.size(); ++i) {
        REQUIRE(rule.x_forward[i] == Approx(rule.x[i] - rule.a));
    }
    for (size_t i = 0; i < rule.x_backward.size(); ++i) {
        REQUIRE(rule.x_backward[i] == Approx(rule.b - rule.x[i]));
    }
}

TEST_CASE("gauss.cpp", "[gauss]")
{
    SECTION("Rule constructor")
    {
        std::vector<DDouble> x(20), w(20);
        DDouble a = -1, b = 1;
        xprec::gauss_legendre(20, x.data(), w.data());
        Eigen::VectorX<DDouble> x_eigen =
            Eigen::Map<Eigen::VectorX<DDouble>>(x.data(), x.size());
        Eigen::VectorX<DDouble> w_eigen =
            Eigen::Map<Eigen::VectorX<DDouble>>(w.data(), w.size());
        sparseir::Rule<DDouble> r1 = sparseir::Rule<DDouble>(x_eigen, w_eigen);
        sparseir::Rule<DDouble> r2 =
            sparseir::Rule<DDouble>(x_eigen, w_eigen, a, b);
        REQUIRE(r1.a == r2.a);
        REQUIRE(r1.b == r2.b);
        REQUIRE(r1.x == r2.x);
        REQUIRE(r1.w == r2.w);
    }

    SECTION("legendre")
    {
        int n = 16;
        sparseir::Rule<double> rule = sparseir::legendre(n);

        Eigen::VectorX<double> x_expected(16);
        x_expected << -0.9894009349916499, -0.9445750230732325,
            -0.8656312023878318, -0.755404408355003, -0.6178762444026438,
            -0.45801677765722737, -0.2816035507792589, -0.09501250983763743,
            0.09501250983763743, 0.2816035507792589, 0.45801677765722737,
            0.6178762444026438, 0.755404408355003, 0.8656312023878318,
            0.9445750230732325, 0.9894009349916499;
        REQUIRE(rule.x.isApprox(x_expected));

        Eigen::VectorX<double> w_expected(16);
        w_expected << 0.027152459411754124, 0.06225352393864806,
            0.0951585116824928, 0.12462897125553389, 0.14959598881657682,
            0.16915651939500254, 0.18260341504492367, 0.18945061045506834,
            0.18945061045506834, 0.18260341504492367, 0.16915651939500254,
            0.14959598881657682, 0.12462897125553389, 0.0951585116824928,
            0.06225352393864806, 0.027152459411754124;
        REQUIRE(rule.w.isApprox(w_expected));

        REQUIRE(rule.a == -1.0);
        REQUIRE(rule.b == 1.0);

        Eigen::VectorX<double> x_forward_expected(16);
        x_forward_expected << 0.010599065008350061, 0.05542497692676751,
            0.13436879761216824, 0.244595591644997, 0.38212375559735623,
            0.5419832223427726, 0.7183964492207411, 0.9049874901623626,
            1.0950125098376375, 1.281603550779259, 1.4580167776572273,
            1.6178762444026438, 1.755404408355003, 1.8656312023878319,
            1.9445750230732326, 1.98940093499165;
        REQUIRE(rule.x_forward.isApprox(x_forward_expected));

        Eigen::VectorX<double> x_backward_expected(16);
        x_backward_expected << 1.98940093499165, 1.9445750230732326,
            1.8656312023878319, 1.755404408355003, 1.6178762444026438,
            1.4580167776572273, 1.281603550779259, 1.0950125098376375,
            0.9049874901623626, 0.7183964492207411, 0.5419832223427726,
            0.38212375559735623, 0.244595591644997, 0.13436879761216824,
            0.05542497692676751, 0.010599065008350061;
        REQUIRE(rule.x_backward.isApprox(x_backward_expected));
    }

    SECTION("legvander6")
    {
        int n = 6;
        auto result = sparseir::legvander(sparseir::legendre(n).x, n - 1);
        Eigen::MatrixX<DDouble> expected(n, n);
        // expected is computed by
        // using SparseIR; m = SparseIR.legvander(SparseIR.sparseir::legendre(6,
        // SparseIR.Float64x2).x, 5); foreach(x -> println(x, ","), vec(m'))
        expected << 1.0, -0.9324695142031520278123015544939835,
            0.8042490923773935119886600608277198,
            -0.6282499246436887457708844976782951,
            0.4220050092706226656844451152082432,
            -0.2057123110596225258297870187140517, 1.0,
            -0.6612093864662645136613995950198845,
            0.1557967791266409127010318094210897,
            0.26911576974459911112396181357848563,
            -0.428245862097120739542522563281485,
            0.2943957149254374170467243373494173, 1.0,
            -0.2386191860831969086305017216807169,
            -0.4145913260494889701442373247943369,
            0.3239618653539352481441754340495602,
            0.1756623404298037786973215350969389,
            -0.33461902074104083146186699361445111, 1.0,
            0.2386191860831969086305017216807169,
            -0.4145913260494889701442373247943369,
            -0.3239618653539352481441754340495602,
            0.1756623404298037786973215350969389,
            0.33461902074104083146186699361445111, 1.0,
            0.6612093864662645136613995950198845,
            0.1557967791266409127010318094210897,
            -0.26911576974459911112396181357848563,
            -0.428245862097120739542522563281485,
            -0.2943957149254374170467243373494173, 1.0,
            0.9324695142031520278123015544939835,
            0.8042490923773935119886600608277198,
            0.6282499246436887457708844976782951,
            0.4220050092706226656844451152082432,
            0.2057123110596225258297870187140517;
        DDouble e = 1e-13;
        REQUIRE(result.isApprox(expected, e));
    }

    SECTION("legendre_collocation")
    {
        sparseir::Rule<DDouble> r = sparseir::legendre(2);
        Eigen::MatrixX<DDouble> result = sparseir::legendre_collocation(r);
        Eigen::MatrixX<DDouble> expected(2, 2);
        // expected is computed by
        // julia> using SparseIR; m =
        // SparseIR.legendre_collocation(SparseIR.sparseir::legendre(2,
        // SparseIR.Float64x2))
        expected << 0.5, 0.5, -0.8660254037844386467637231707528938,
            0.8660254037844386467637231707528938;
        DDouble e = 1e-13;
        REQUIRE(result.isApprox(expected, e));
    }

    SECTION("collocate")
    {
        int n = 6;
        sparseir::Rule<DDouble> r = sparseir::legendre(n);

        Eigen::MatrixX<DDouble> cmat = legendre_collocation(
            r); // Assuming legendre_collocation function is defined

        Eigen::MatrixX<DDouble> emat = sparseir::legvander(r.x, r.x.size() - 1);
        DDouble e = 1e-13;
        Eigen::MatrixX<DDouble> out = emat * cmat;
        REQUIRE(
            (emat * cmat).isApprox(Eigen::MatrixX<DDouble>::Identity(n, n), e));
    }

    SECTION("gauss sparseir::legendre")
    {
        int n = 200;
        sparseir::Rule<DDouble> rule = sparseir::legendre(n);
        gaussValidate(rule);
        std::vector<DDouble> x(n), w(n);
        xprec::gauss_legendre(n, x.data(), w.data());
        Eigen::VectorX<DDouble> x_eigen =
            Eigen::Map<Eigen::VectorX<DDouble>>(x.data(), n);
        Eigen::VectorX<DDouble> w_eigen =
            Eigen::Map<Eigen::VectorX<DDouble>>(w.data(), n);
        REQUIRE(rule.x == x_eigen);
        REQUIRE(rule.w == w_eigen);
    }

    SECTION("piecewise")
    {
        std::vector<DDouble> edges = {-4, -1, 1, 3};
        sparseir::Rule<DDouble> rule = sparseir::legendre(20).piecewise(edges);
        gaussValidate(rule);
    }
}