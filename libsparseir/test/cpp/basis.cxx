#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <catch2/catch_approx.hpp> // for Approx
#include <catch2/catch_test_macros.hpp>

#include "sve_cache.hpp"
#include <sparseir/sparseir.hpp>
#include <xprec/ddouble-header-only.hpp>
#include "_utils.hpp"

using Catch::Approx;
using std::complex;
using std::make_shared;
using std::pair;
using std::shared_ptr;
using std::vector;

using ComplexF64 = complex<double>;

TEST_CASE("basis.u[0] test", "[basis]")
{
    /*
    using SparseIR
    begin
        β = 1.0
        Λ = 10.
        sve_result = SparseIR.SVEResult(LogisticKernel(Λ), ε=1e-15)
        basis = FiniteTempBasis{Bosonic}(β, Λ; sve_result)
        u = basis.u[begin]
    end
    */
    double beta = 1.0;
    double wmax = 10.0;
    auto kernel = sparseir::LogisticKernel(beta * wmax);
    auto sve_result = SVECache::get_sve_result(kernel, 1e-15);
    auto basis = make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
        beta, wmax, 1e-15, kernel, sve_result);

    vector<double> sampling_points_ref_vec = {
        -0.9926231612199574,     -0.9612900365085015,  -0.9055709783447798,
        -0.8265940603194348,     -0.7260516410338869,  -0.6062233701847619,
        -0.46999176945389465,    -0.32082393925747893, -0.16270277264190053,
        -1.8244248383569015e-16, 0.16270277264190053,  0.32082393925747893,
        0.46999176945389465,     0.606223370184762,    0.7260516410338869,
        0.8265940603194348,      0.9055709783447796,   0.9612900365085015,
        0.9926231612199574,
    };

    auto sampling_points = sparseir::default_sampling_points(
        *(basis->sve_result->u), basis->size());
    REQUIRE(sampling_points.isApprox(Eigen::Map<Eigen::VectorXd>(
        sampling_points_ref_vec.data(), sampling_points_ref_vec.size())));

    vector<double> s_ref_vec = {
        1.2621489299919293,     0.8363588547029699,     0.3462622585830318,
        0.12082626967769121,    0.03387861935965415,    0.00796130085778543,
        0.0015966925515801561,  0.00027823051505153205, 4.276437930624593e-5,
        5.871774564004103e-6,   7.279433734090833e-7,   8.221881611234219e-8,
        8.525219704563244e-9,   8.168448057712933e-10,  7.273189647675939e-11,
        6.0477895415959716e-12, 4.716593209003674e-13,  3.4631886072022945e-14,
        2.4022217530858486e-15,
    };
    Eigen::Map<Eigen::VectorXd> s(basis->s.data(), basis->s.size());
    REQUIRE(s.size() == s_ref_vec.size());
    Eigen::VectorXd s_ref =
        Eigen::Map<Eigen::VectorXd>(s_ref_vec.data(), s_ref_vec.size());
    REQUIRE(s.isApprox(s_ref));

    double x = 0.3;
    sparseir::PiecewiseLegendrePoly u0 = _singleout_poly_from_irtaufuncs(*(basis->u), 0);
    auto u0x = u0(x);
    REQUIRE(std::is_same<decltype(u0x), double>::value);
    REQUIRE(u0.get_xmin() == 0.0);
    REQUIRE(u0.get_xmax() == 1.0);
    REQUIRE(u0.get_polyorder() == 16);
    REQUIRE(u0.get_l() == 0);
    REQUIRE(u0.get_symm() == 1);

    vector<double> u_knot_ref_vec = {
        0.0,
        0.013623446212733203,
        0.0292485379483044,
        0.04713527623719438,
        0.0675600047521634,
        0.09080712995504114,
        0.11715549463550101,
        0.14685782470149988,
        0.18011200903273533,
        0.21702410043575965,
        0.25756521020520734,
        0.3015280008924536,
        0.3484926000123403,
        0.3978146340870691,
        0.44864700611131486,
        0.5,
        0.5513529938886851,
        0.6021853659129309,
        0.6515073999876597,
        0.6984719991075464,
        0.7424347897947927,
        0.7829758995642404,
        0.8198879909672647,
        0.8531421752985001,
        0.882844505364499,
        0.9091928700449589,
        0.9324399952478366,
        0.9528647237628056,
        0.9707514620516956,
        0.9863765537872669,
        1.0,
    };
    Eigen::Map<Eigen::VectorXd> u_knots(u0.knots.data(), u0.knots.size());
    REQUIRE(u_knots.isApprox(Eigen::Map<Eigen::VectorXd>(
        u_knot_ref_vec.data(), u_knot_ref_vec.size())));

    vector<double> u_delta_x_ref_vec = {
        0.013623446212733203, 0.015625091735571195, 0.01788673828888998,
        0.020424728514969015, 0.02324712520287775,  0.026348364680459868,
        0.029702330065998872, 0.03325418433123545,  0.036912091403024316,
        0.04054110976944769,  0.043962790687246206, 0.04696459911988671,
        0.049322034074728835, 0.050832372024245794, 0.05135299388868512,
        0.05135299388868512,  0.050832372024245794, 0.049322034074728835,
        0.04696459911988671,  0.043962790687246206, 0.04054110976944769,
        0.036912091403024316, 0.03325418433123545,  0.029702330065998872,
        0.026348364680459868, 0.02324712520287775,  0.020424728514969015,
        0.01788673828888998,  0.015625091735571195, 0.013623446212733203,
    };
    Eigen::Map<Eigen::VectorXd> u_delta_x(u0.delta_x.data(), u0.delta_x.size());
    REQUIRE(u_delta_x.isApprox(Eigen::Map<Eigen::VectorXd>(
        u_delta_x_ref_vec.data(), u_delta_x_ref_vec.size())));

    REQUIRE(u0(0.3) == Approx(0.8209004724107448));
    int i;
    double x_tilde;
    std::tie(i, x_tilde) = u0.split(0.3);
    REQUIRE(i == 10);
    REQUIRE(x_tilde == Approx(0.9304866288710429));
}

TEST_CASE("basis.u(x)", "[basis]")
{
    /*
    # Julia implementation
    begin
        Λ = 10.
        sve_result = SparseIR.SVEResult(LogisticKernel(Λ), ε=1e-15)
        basis = FiniteTempBasis{Bosonic}(
            1., Λ; sve_result
        )
        u0 = basis.u[1]
        x = 0.3
        u0(x)
    end
    */

    double beta = 1.0;
    double wmax = 10.0;
    auto kernel = sparseir::LogisticKernel(beta * wmax);
    auto sve_result = SVECache::get_sve_result(kernel, 1e-15);

    auto basis = make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
        beta, wmax, 1e-15, kernel, sve_result);
    double x = 0.3;
    sparseir::PiecewiseLegendrePoly u0 = _singleout_poly_from_irtaufuncs(*basis->u, 0);
    auto u0x = u0(x);
    REQUIRE(std::is_same<decltype(u0x), double>::value);

    vector<double> u0_data_vec = {
        0.1306343289249387,      -0.003088620328392153,
        4.332434116041884e-5,    -4.051803535484306e-7,
        3.0294780746417638e-9,   -1.8454824890861207e-11,
        9.593259697606462e-14,   -4.318378229918751e-16,
        -1.6057152611363738e-17, 8.034298644739209e-19,
        -5.215449610952754e-17,  1.6770637398117038e-18,
        -9.088555845096977e-17,  1.3601547584026748e-18,
        -2.1290268153903202e-17, 2.2812299069925636e-19,
        0.133117010114246,       -0.0034677634482834954,
        5.524039190213751e-5,    -5.852928194892295e-7,
        4.992377187882264e-9,    -3.470094120915678e-11,
        2.0626937965651714e-13,  -1.0621125444648503e-15,
        -1.327011574477474e-17,  8.891860052322482e-19,
        -5.3150032070815465e-17, 1.8829704560145743e-18,
        -9.261720518028769e-17,  1.5271597311845498e-18,
        -2.1698376474365316e-17, 2.5619103723903433e-19,
        0.1348677927781422,      -0.0038359347262132584,
        6.91944672290212e-5,     -8.268613421443971e-7,
        8.024718568697224e-9,    -6.345427395840508e-11,
        4.3032181031196856e-13,  -2.5281727376047486e-15,
        -5.1747925868067095e-18, 9.440394707282658e-19,
        -5.3854136235768805e-17, 2.0829380040824132e-18,
        -9.384055332144593e-17,  1.6893519670818779e-18,
        -2.1987920900745524e-17, 2.8347999649558843e-19,
        0.13583520528428533,     -0.004171580503739914,
        8.493448081651559e-5,    -1.138085125623402e-6,
        1.2526233369579828e-8,   -1.122452205822655e-10,
        8.659014584499268e-13,   -5.785591933646663e-15,
        1.587938940904511e-17,   9.109693637060432e-19,
        -5.4245865298256037e-17, 2.2652649115211116e-18,
        -9.451970000002734e-17,  1.8372430819635655e-18,
        -2.2150439035417448e-17, 3.084034195336992e-19,
        0.13600309349640446,     -0.004449238725087478,
        0.00010188835454079239,  -1.519447054286535e-6,
        1.889170501672303e-8,    -1.9087128529968257e-10,
        1.668854643854283e-12,   -1.2627907764313899e-14,
        6.665421392806912e-17,   6.530289128320217e-19,
        -5.431766546550717e-17,  2.416121691783754e-18,
        -9.464318251380017e-17,  1.9596213517746516e-18,
        -2.218312002743061e-17,  3.290826047521968e-19,
        0.13539867705132935,     -0.004641176436720682,
        0.00011912116720466503,  -1.957779132706826e-6,
        2.7370414622389277e-8,   -3.097334872911258e-10,
        3.0555739278178214e-12,  -2.6039961676418555e-14,
        1.8009454727178994e-16,  -1.3416575385737644e-19,
        -5.40777083077729e-17,   2.520431207688298e-18,
        -9.422956521670746e-17,  2.044270970118777e-18,
        -2.2090103166217916e-17, 3.434644834131977e-19,
        0.13409715895853275,     -0.004719986969004002,
        0.00013536907031716337,  -2.420436638151997e-6,
        3.785300630728854e-8,    -4.755666885288881e-10,
        5.265379848297904e-12,   -5.0169717001576626e-14,
        4.118401065780764e-16,   -2.0446457822121647e-18,
        -5.35501633054833e-17,   2.5632616361585686e-18,
        -9.33306591515526e-17,   2.0791139201621523e-18,
        -2.1883239227246204e-17, 3.4951037695345937e-19,
        0.13222103792506845,     -0.004661940139242021,
        0.00014918832923537266,  -2.852394734987402e-6,
        4.964092300409242e-8,    -6.841888454869215e-10,
        8.45142444088915e-12,    -8.915612056827141e-14,
        8.362314469143777e-16,   -6.0234895174712546e-18,
        -5.277316485609017e-17,  2.5316204613489296e-18,
        -9.203110228857883e-17,  2.053684309629933e-18,
        -2.1582025252842366e-17, 3.4544177495993926e-19,
        0.1299329360557847,      -0.0044505687852619,
        0.00015923080726201497,  -3.1799170775970573e-6,
        6.133479591365844e-8,    -9.121828673040286e-10,
        1.249871300365485e-11,   -1.4402027451892738e-13,
        1.5147956156101485e-15,  -1.3107025245287422e-17,
        -5.179657846894099e-17,  2.416391390722045e-18,
        -9.044347906363985e-17,  1.9607087542271804e-18,
        -2.1212523005683314e-17, 3.300068208808041e-19,
        0.12742205869276685,     -0.004079739261690625,
        0.00016460133801271778,  -3.323384001363083e-6,
        7.102844349567234e-8,    -1.1129202291336217e-9,
        1.6854888730035335e-11,  -2.0802811534878425e-13,
        2.4289300367321258e-15,  -2.35162299615964e-17,
        -5.068452880161132e-17,  2.214070031982241e-18,
        -8.869904343539805e-17,  1.797462302645294e-18,
        -2.080526618099368e-17,  3.0271327424360563e-19,
        0.12488604691311368,     -0.0035554511594525565,
        0.0001651837328374035,   -3.2179846422232465e-6,
        7.691267497167832e-8,    -1.2240947764811543e-9,
        2.0562588873292238e-11,  -2.6368003120635255e-13,
        3.4125649732034824e-15,  -3.525694467765447e-17,
        -4.9524847106933214e-17, 1.927953455490532e-18,
        -8.69352262317446e-17,   1.566567186371069e-18,
        -2.039240753381822e-17,  2.639716125971533e-19,
        0.12251133488059741,     -0.0028959196219000448,
        0.00016178189481639923,  -2.8354837701945592e-6,
        7.811993243177101e-8,    -1.1893726727069478e-9,
        2.2651634863163037e-11,  -2.864356292789231e-13,
        4.17643217199609e-15,    -4.3530851579701145e-17,
        -4.8432887661721154e-17, 1.568435980322896e-18,
        -8.528203040124262e-17,  1.2760356815022537e-18,
        -2.0004564431181052e-17, 2.151119394338036e-19,
        0.12045550335931388,     -0.0021299971737511955,
        0.00015595793131832877,  -2.1971718263068114e-6,
        7.531412168714067e-8,    -9.855388935566871e-10,
        2.275227647285319e-11,   -2.5684758474727664e-13,
        4.4812639187570635e-15,  -4.271913858540445e-17,
        -4.752579106509054e-17,  1.1520658876395328e-18,
        -8.384969222532824e-17,  9.385791170908876e-19,
        -1.9667894078773607e-17, 1.5827397556551558e-19,
        0.11883426988578132,     -0.0012944145276902674,
        0.00014957121718186817,  -1.371361155558782e-6,
        7.049246064311859e-8,    -6.381023775808443e-10,
        2.1419621010469043e-11,  -1.740303200914425e-13,
        4.337356464969145e-15,   -3.051082300661339e-17,
        -4.6871616133348213e-17, 6.99397965248822e-19,
        -8.271944407490659e-17,  5.703920911526572e-19,
        -1.940182948222789e-17,  9.620220822710325e-20,
        0.11771412671520903,     -0.0004304408867896644,
        0.00014417587775830828,  -4.572235733754134e-7,
        6.593973286016377e-8,    -2.1435848824301272e-10,
        1.9738899595315872e-11,  -5.917713099809887e-14,
        3.988701201825088e-15,   -1.0536464222579925e-17,
        -4.6468226059630736e-17, 2.3250576202408365e-19,
        -8.19381504674851e-17,   1.8967689986618523e-19,
        -1.9217690339092148e-17, 3.199140100593713e-20,
        0.11771412671520903,     0.0004304408867896644,
        0.00014417587775830828,  4.572235733754134e-7,
        6.593973286016377e-8,    2.1435848824301272e-10,
        1.9738899595315872e-11,  5.917713099809887e-14,
        3.988701201825088e-15,   1.0536464222579925e-17,
        -4.6468226059630736e-17, -2.3250576202408365e-19,
        -8.19381504674851e-17,   -1.8967689986618523e-19,
        -1.9217690339092148e-17, -3.199140100593713e-20,
        0.11883426988578132,     0.0012944145276902674,
        0.00014957121718186817,  1.371361155558782e-6,
        7.049246064311859e-8,    6.381023775808443e-10,
        2.1419621010469043e-11,  1.740303200914425e-13,
        4.337356464969145e-15,   3.051082300661339e-17,
        -4.6871616133348213e-17, -6.99397965248822e-19,
        -8.271944407490659e-17,  -5.703920911526572e-19,
        -1.940182948222789e-17,  -9.620220822710325e-20,
        0.12045550335931388,     0.0021299971737511955,
        0.00015595793131832877,  2.1971718263068114e-6,
        7.531412168714067e-8,    9.855388935566871e-10,
        2.275227647285319e-11,   2.5684758474727664e-13,
        4.4812639187570635e-15,  4.271913858540445e-17,
        -4.752579106509054e-17,  -1.1520658876395328e-18,
        -8.384969222532824e-17,  -9.385791170908876e-19,
        -1.9667894078773607e-17, -1.5827397556551558e-19,
        0.12251133488059741,     0.0028959196219000448,
        0.00016178189481639923,  2.8354837701945592e-6,
        7.811993243177101e-8,    1.1893726727069478e-9,
        2.2651634863163037e-11,  2.864356292789231e-13,
        4.17643217199609e-15,    4.3530851579701145e-17,
        -4.8432887661721154e-17, -1.568435980322896e-18,
        -8.528203040124262e-17,  -1.2760356815022537e-18,
        -2.0004564431181052e-17, -2.151119394338036e-19,
        0.12488604691311368,     0.0035554511594525565,
        0.0001651837328374035,   3.2179846422232465e-6,
        7.691267497167832e-8,    1.2240947764811543e-9,
        2.0562588873292238e-11,  2.6368003120635255e-13,
        3.4125649732034824e-15,  3.525694467765447e-17,
        -4.9524847106933214e-17, -1.927953455490532e-18,
        -8.69352262317446e-17,   -1.566567186371069e-18,
        -2.039240753381822e-17,  -2.639716125971533e-19,
        0.12742205869276685,     0.004079739261690625,
        0.00016460133801271778,  3.323384001363083e-6,
        7.102844349567234e-8,    1.1129202291336217e-9,
        1.6854888730035335e-11,  2.0802811534878425e-13,
        2.4289300367321258e-15,  2.35162299615964e-17,
        -5.068452880161132e-17,  -2.214070031982241e-18,
        -8.869904343539805e-17,  -1.797462302645294e-18,
        -2.080526618099368e-17,  -3.0271327424360563e-19,
        0.1299329360557847,      0.0044505687852619,
        0.00015923080726201497,  3.1799170775970573e-6,
        6.133479591365844e-8,    9.121828673040286e-10,
        1.249871300365485e-11,   1.4402027451892738e-13,
        1.5147956156101485e-15,  1.3107025245287422e-17,
        -5.179657846894099e-17,  -2.416391390722045e-18,
        -9.044347906363985e-17,  -1.9607087542271804e-18,
        -2.1212523005683314e-17, -3.300068208808041e-19,
        0.13222103792506845,     0.004661940139242021,
        0.00014918832923537266,  2.852394734987402e-6,
        4.964092300409242e-8,    6.841888454869215e-10,
        8.45142444088915e-12,    8.915612056827141e-14,
        8.362314469143777e-16,   6.0234895174712546e-18,
        -5.277316485609017e-17,  -2.5316204613489296e-18,
        -9.203110228857883e-17,  -2.053684309629933e-18,
        -2.1582025252842366e-17, -3.4544177495993926e-19,
        0.13409715895853275,     0.004719986969004002,
        0.00013536907031716337,  2.420436638151997e-6,
        3.785300630728854e-8,    4.755666885288881e-10,
        5.265379848297904e-12,   5.0169717001576626e-14,
        4.118401065780764e-16,   2.0446457822121647e-18,
        -5.35501633054833e-17,   -2.5632616361585686e-18,
        -9.33306591515526e-17,   -2.0791139201621523e-18,
        -2.1883239227246204e-17, -3.4951037695345937e-19,
        0.13539867705132935,     0.004641176436720682,
        0.00011912116720466503,  1.957779132706826e-6,
        2.7370414622389277e-8,   3.097334872911258e-10,
        3.0555739278178214e-12,  2.6039961676418555e-14,
        1.8009454727178994e-16,  1.3416575385737644e-19,
        -5.40777083077729e-17,   -2.520431207688298e-18,
        -9.422956521670746e-17,  -2.044270970118777e-18,
        -2.2090103166217916e-17, -3.434644834131977e-19,
        0.13600309349640446,     0.004449238725087478,
        0.00010188835454079239,  1.519447054286535e-6,
        1.889170501672303e-8,    1.9087128529968257e-10,
        1.668854643854283e-12,   1.2627907764313899e-14,
        6.665421392806912e-17,   -6.530289128320217e-19,
        -5.431766546550717e-17,  -2.416121691783754e-18,
        -9.464318251380017e-17,  -1.9596213517746516e-18,
        -2.218312002743061e-17,  -3.290826047521968e-19,
        0.13583520528428533,     0.004171580503739914,
        8.493448081651559e-5,    1.138085125623402e-6,
        1.2526233369579828e-8,   1.122452205822655e-10,
        8.659014584499268e-13,   5.785591933646663e-15,
        1.587938940904511e-17,   -9.109693637060432e-19,
        -5.4245865298256037e-17, -2.2652649115211116e-18,
        -9.451970000002734e-17,  -1.8372430819635655e-18,
        -2.2150439035417448e-17, -3.084034195336992e-19,
        0.1348677927781422,      0.0038359347262132584,
        6.91944672290212e-5,     8.268613421443971e-7,
        8.024718568697224e-9,    6.345427395840508e-11,
        4.3032181031196856e-13,  2.5281727376047486e-15,
        -5.1747925868067095e-18, -9.440394707282658e-19,
        -5.3854136235768805e-17, -2.0829380040824132e-18,
        -9.384055332144593e-17,  -1.6893519670818779e-18,
        -2.1987920900745524e-17, -2.8347999649558843e-19,
        0.133117010114246,       0.0034677634482834954,
        5.524039190213751e-5,    5.852928194892295e-7,
        4.992377187882264e-9,    3.470094120915678e-11,
        2.0626937965651714e-13,  1.0621125444648503e-15,
        -1.327011574477474e-17,  -8.891860052322482e-19,
        -5.3150032070815465e-17, -1.8829704560145743e-18,
        -9.261720518028769e-17,  -1.5271597311845498e-18,
        -2.1698376474365316e-17, -2.5619103723903433e-19,
        0.1306343289249387,      0.003088620328392153,
        4.332434116041884e-5,    4.051803535484306e-7,
        3.0294780746417638e-9,   1.8454824890861207e-11,
        9.593259697606462e-14,   4.318378229918751e-16,
        -1.6057152611363738e-17, -8.034298644739209e-19,
        -5.215449610952754e-17,  -1.6770637398117038e-18,
        -9.088555845096977e-17,  -1.3601547584026748e-18,
        -2.1290268153903202e-17, -2.2812299069925636e-19,
    };
    Eigen::MatrixXd u0_data_mat_eigen =
        Eigen::Map<Eigen::MatrixXd>(u0_data_vec.data(), 16, 30);
    REQUIRE(u0.data.isApprox(u0_data_mat_eigen));

    std::vector<double> u0_knots_vec = {
        0.0,
        0.013623446212733203,
        0.0292485379483044,
        0.04713527623719438,
        0.0675600047521634,
        0.09080712995504114,
        0.11715549463550101,
        0.14685782470149988,
        0.18011200903273533,
        0.21702410043575965,
        0.25756521020520734,
        0.3015280008924536,
        0.3484926000123403,
        0.3978146340870691,
        0.44864700611131486,
        0.5,
        0.5513529938886851,
        0.6021853659129309,
        0.6515073999876597,
        0.6984719991075464,
        0.7424347897947927,
        0.7829758995642404,
        0.8198879909672647,
        0.8531421752985001,
        0.882844505364499,
        0.9091928700449589,
        0.9324399952478366,
        0.9528647237628056,
        0.9707514620516956,
        0.9863765537872669,
        1.0,
    };
    Eigen::VectorXd u0_knots_vec_eigen =
        Eigen::Map<Eigen::VectorXd>(u0_knots_vec.data(), u0_knots_vec.size());
    REQUIRE(u0_knots_vec_eigen.size() == u0.knots.size());
    REQUIRE(u0_knots_vec_eigen.isApprox(u0.knots));
    REQUIRE(u0.xmin == 0.0);
    REQUIRE(u0.xmax == 1.0);
    REQUIRE(u0.symm == 1);
    REQUIRE(u0.polyorder == 16);
    REQUIRE(u0.l == 0);
    REQUIRE(u0.xm.size() == 30);
    REQUIRE(u0.inv_xs.size() == 30);
    REQUIRE(u0.norms.size() == 30);
}

TEST_CASE("FiniteTempBasis consistency tests", "[basis]")
{
    SECTION("Basic consistency")
    {
        double beta = 1.0;
        double omega_max = 1.0;
        double epsilon = 1e-5;

        // Define the kernel
        auto kernel = sparseir::LogisticKernel(beta * omega_max);

        // Specify both template parameters: S and K
        using FermKernel = sparseir::FiniteTempBasis<sparseir::Fermionic>;
        using BosKernel = sparseir::FiniteTempBasis<sparseir::Bosonic>;

        std::pair<std::shared_ptr<FermKernel>, std::shared_ptr<BosKernel>>
            bases = sparseir::finite_temp_bases(beta, omega_max, epsilon);

        // Create the basis set without template parameter
        sparseir::FiniteTempBasisSet bs(bases.first, bases.second,
                                        Eigen::VectorXd(),  // Empty tau vector
                                        std::vector<int>(), // Empty wn_f vector
                                        std::vector<int>()  // Empty wn_b vector
        );

        // Use s (singular values) instead of singular_values() method
        REQUIRE(bases.first->s.size() == bs.basis_f->s.size());
        REQUIRE(bases.second->s.size() == bs.basis_b->s.size());
    }

    SECTION("Sampling consistency")
    {
        double beta = 2.0;
        double omega_max = 5.0;
        double epsilon = 1e-5;

        auto kernel = sparseir::LogisticKernel(beta * omega_max);

        // Specify the template argument for SVEResult
        sparseir::SVEResult sve_result = sparseir::compute_sve(kernel, epsilon);

        using FermKernel = sparseir::FiniteTempBasis<sparseir::Fermionic>;
        using BosKernel = sparseir::FiniteTempBasis<sparseir::Bosonic>;

        std::pair<std::shared_ptr<FermKernel>, std::shared_ptr<BosKernel>>
            bases = sparseir::finite_temp_bases(beta, omega_max, epsilon,
                                                sve_result);

        // Create shared pointers for the bases
        std::shared_ptr<FermKernel> basis_f_ptr = bases.first;
        std::shared_ptr<BosKernel> basis_b_ptr = bases.second;

        // Create the basis set without template parameter
        sparseir::FiniteTempBasisSet bs(
            basis_f_ptr, basis_b_ptr,
            Eigen::VectorXd(),   // Empty tau vector
            std::vector<int>(),  // Empty wn_f vector
            std::vector<int>()); // Empty wn_b vector

        // Check sampling points consistency
        sparseir::TauSampling<sparseir::Fermionic> smpl_tau_f(basis_f_ptr);
        sparseir::TauSampling<sparseir::Bosonic> smpl_tau_b(basis_b_ptr);

        REQUIRE(smpl_tau_f.sampling_points() == smpl_tau_b.sampling_points());
    }

    SECTION("Singular value scaling")
    {
        double beta = 1e-3;
        double omega_max = 1e-3;
        double epsilon = 1e-100;
        auto kernel = sparseir::LogisticKernel(beta * omega_max);
        auto sve_result = sparseir::compute_sve(kernel, epsilon);
        sparseir::FiniteTempBasis<sparseir::Fermionic> basis(
            beta, omega_max, epsilon, kernel, sve_result);
        REQUIRE(sve_result.s.size() > 0);
        REQUIRE(basis.s.size() > 0);
        double scale = std::sqrt(beta / 2.0 * omega_max);
        // Ensure the correct function or member is used for singular values
        Eigen::VectorXd scaled_s_eigen = sve_result.s * scale;
        REQUIRE(basis.s.size() == sve_result.s.size());
        REQUIRE(basis.s.isApprox(scaled_s_eigen));
        // Access accuracy as a member variable if it's not a function
        REQUIRE(
            std::abs(basis.accuracy - sve_result.s(sve_result.s.size() - 1) /
                                          sve_result.s(0)) < 1e-10);
    }

    SECTION("Rescaling test")
    {
        double beta = 3.0;
        double omega_max = 4.0;
        double epsilon = 1e-6;

        // Specify both template parameters
        auto kernel = sparseir::LogisticKernel(beta * omega_max);
        auto sve_result = sparseir::compute_sve(kernel, epsilon);
        sparseir::FiniteTempBasis<sparseir::Fermionic> basis(
            beta, omega_max, epsilon, kernel, sve_result);
        sparseir::FiniteTempBasis<sparseir::Fermionic> rescaled_basis =
            basis.rescale(2.0);
        REQUIRE(rescaled_basis.sve_result->s.size() ==
                basis.sve_result->s.size());
        REQUIRE(rescaled_basis.get_wmax() == 6.0);
    }

    SECTION("default_sampling_points")
    {
        auto beta = 3.0;
        auto omega_max = 4.0;
        auto epsilon = 1e-6;
        auto kernel = sparseir::LogisticKernel(beta * omega_max);
        auto sve_result = sparseir::compute_sve(kernel, epsilon);
        auto basis = sparseir::FiniteTempBasis<sparseir::Fermionic>(
            beta, omega_max, epsilon, kernel, sve_result);
        auto s = sve_result.s;
        // REQUIRE(s.size() == 32);

        std::vector<double> s_ref = {
            0.5242807065966564,     0.361040299707525,
            0.1600617039313169,     0.06192139783088188,
            0.019641646995563183,   0.005321140031657106,
            0.001245435134907047,   0.0002553808249508306,
            4.6445392784931696e-5,  7.57389586542119e-6,
            1.1180101601552092e-6,  1.506251988966008e-7,
            1.8653991892840962e-8,  2.136773728637427e-9,
            2.276179221544401e-10,  2.2655690134240947e-11,
            2.115880115422964e-12,  1.861108037178489e-13,
            1.5466716841180263e-14, 1.218212516630768e-15,
            5.590657287253601e-16,  4.656548094642833e-16,
            4.552528808131262e-16,  4.341440592462354e-16,
            3.744993780121804e-16,  3.549006072192367e-16,
            3.277985748785467e-16,  3.2621304578629284e-16,
            3.2046691517654354e-16, 3.097729851576022e-16,
            2.4973730182025644e-16, 2.476022474231314e-16};
        Eigen::VectorXd s_double = s.template cast<double>();
        REQUIRE(std::fabs(s_double[0] - s_ref[0]) < 1e-10);
        REQUIRE(std::fabs(s_double[1] - s_ref[1]) < 1e-10);
        REQUIRE(std::fabs(s_double[2] - s_ref[2]) < 1e-10);
        REQUIRE(std::fabs(s_double[3] - s_ref[3]) < 1e-10);
        REQUIRE(std::fabs(s_double[4] - s_ref[4]) < 1e-10);
        REQUIRE(std::fabs(s_double[5] - s_ref[5]) < 1e-10);
        REQUIRE(std::fabs(s_double[6] - s_ref[6]) < 1e-10);
        REQUIRE(std::fabs(s_double[7] - s_ref[7]) < 1e-10);
        REQUIRE(std::fabs(s_double[8] - s_ref[8]) < 1e-10);
        REQUIRE(std::fabs(s_double[9] - s_ref[9]) < 1e-10);
        REQUIRE(std::fabs(s_double[10] - s_ref[10]) < 1e-10);
        REQUIRE(std::fabs(s_double[11] - s_ref[11]) < 1e-10);
        REQUIRE(std::fabs(s_double[12] - s_ref[12]) < 1e-10);
        REQUIRE(std::fabs(s_double[13] - s_ref[13]) < 1e-10);
        REQUIRE(std::fabs(s_double[14] - s_ref[14]) < 1e-10);
        REQUIRE(std::fabs(s_double[15] - s_ref[15]) < 1e-10);
        REQUIRE(std::fabs(s_double[16] - s_ref[16]) < 1e-10);
        REQUIRE(std::fabs(s_double[17] - s_ref[17]) < 1e-10);
        REQUIRE(std::fabs(s_double[18] - s_ref[18]) < 1e-10);
        REQUIRE(std::fabs(s_double[19] - s_ref[19]) < 1e-10);
        REQUIRE(std::fabs(s_double[20] - s_ref[20]) < 1e-10);
        REQUIRE(std::fabs(s_double[21] - s_ref[21]) < 1e-10);
        REQUIRE(std::fabs(s_double[22] - s_ref[22]) < 1e-10);
        REQUIRE(std::fabs(s_double[23] - s_ref[23]) < 1e-10);
        REQUIRE(std::fabs(s_double[24] - s_ref[24]) < 1e-10);
        REQUIRE(std::fabs(s_double[25] - s_ref[25]) < 1e-10);
        REQUIRE(std::fabs(s_double[26] - s_ref[26]) < 1e-10);
        REQUIRE(std::fabs(s_double[27] - s_ref[27]) < 1e-10);
        REQUIRE(std::fabs(s_double[28] - s_ref[28]) < 1e-10);
        REQUIRE(std::fabs(s_double[29] - s_ref[29]) < 1e-10);
        REQUIRE(std::fabs(s_double[30] - s_ref[30]) < 1e-10);
        REQUIRE(std::fabs(s_double[31] - s_ref[31]) < 1e-10);

        REQUIRE((*sve_result.u)[0].data.rows() == 10);
        REQUIRE((*sve_result.u)[0].data.cols() == 32);

        std::vector<double> u_knots_ref = {-1.0,
                                           -0.9768276289532026,
                                           -0.9502121116288913,
                                           -0.9196860690044226,
                                           -0.8847415486995369,
                                           -0.8448386704449569,
                                           -0.7994218020611462,
                                           -0.7479461808659303,
                                           -0.6899180675604202,
                                           -0.6249508554943133,
                                           -0.552837354044473,
                                           -0.4736340017820308,
                                           -0.38774586460365346,
                                           -0.2959932285976203,
                                           -0.19963497739688743,
                                           -0.10032604651986517,
                                           0.0,
                                           0.10032604651986517,
                                           0.19963497739688743,
                                           0.2959932285976203,
                                           0.38774586460365346,
                                           0.4736340017820308,
                                           0.552837354044473,
                                           0.6249508554943133,
                                           0.6899180675604202,
                                           0.7479461808659303,
                                           0.7994218020611462,
                                           0.8448386704449569,
                                           0.8847415486995369,
                                           0.9196860690044226,
                                           0.9502121116288913,
                                           0.9768276289532026,
                                           1.0};
        Eigen::VectorXd u_knots_ref_eigen =
            Eigen::Map<Eigen::VectorXd>(u_knots_ref.data(), u_knots_ref.size());
        REQUIRE((*sve_result.u)[0].knots.isApprox(u_knots_ref_eigen));

        std::vector<double> v_knots_ref = {-1.0,
                                           -0.9833147686254275,
                                           -0.9470082310185116,
                                           -0.8938959515018162,
                                           -0.8283053538395936,
                                           -0.7548706158857138,
                                           -0.6778753393916265,
                                           -0.600858151891138,
                                           -0.5264593296556019,
                                           -0.45644270870032133,
                                           -0.39184906182935686,
                                           -0.3331494756803358,
                                           -0.2804096832724622,
                                           -0.23343248554812435,
                                           -0.19185635090170117,
                                           -0.15524305920516734,
                                           -0.12312152382089525,
                                           -0.0950206581112576,
                                           -0.070491286445028,
                                           -0.04911709970058231,
                                           -0.03050369976269751,
                                           -0.014178372359576086,
                                           0.0,
                                           0.014178372359576086,
                                           0.03050369976269751,
                                           0.04911709970058231,
                                           0.070491286445028,
                                           0.0950206581112576,
                                           0.12312152382089525,
                                           0.15524305920516734,
                                           0.19185635090170117,
                                           0.23343248554812435,
                                           0.2804096832724622,
                                           0.3331494756803358,
                                           0.39184906182935686,
                                           0.45644270870032133,
                                           0.5264593296556019,
                                           0.600858151891138,
                                           0.6778753393916265,
                                           0.7548706158857138,
                                           0.8283053538395936,
                                           0.8938959515018162,
                                           0.9470082310185116,
                                           0.9833147686254275,
                                           1.0};
        Eigen::VectorXd v_knots_ref_eigen =
            Eigen::Map<Eigen::VectorXd>(v_knots_ref.data(), v_knots_ref.size());
        REQUIRE((*sve_result.v)[0].knots.isApprox(v_knots_ref_eigen));

        REQUIRE((*sve_result.u)[1].xmin == -1.0);
        REQUIRE((*sve_result.u)[1].xmax == 1.0);

        REQUIRE((*sve_result.v)[1].xmin == -1.0);
        REQUIRE((*sve_result.v)[1].xmax == 1.0);

        REQUIRE((*sve_result.u)[0].l == 0);
        REQUIRE((*sve_result.u)[1].l == 1);
        REQUIRE((*sve_result.u)[2].l == 2);

        REQUIRE((*sve_result.v)[0].l == 0);
        REQUIRE((*sve_result.v)[1].l == 1);
        REQUIRE((*sve_result.v)[2].l == 2);

        REQUIRE((*sve_result.u)[0].symm == 1);
        REQUIRE((*sve_result.u)[1].symm == -1);
        REQUIRE((*sve_result.u)[2].symm == 1);
        REQUIRE((*sve_result.u)[3].symm == -1);

        REQUIRE((*sve_result.v)[0].symm == 1);
        REQUIRE((*sve_result.v)[1].symm == -1);
        REQUIRE((*sve_result.v)[2].symm == 1);
        REQUIRE((*sve_result.v)[3].symm == -1);

        // std::cout << "Singular values: " << s.transpose() << std::endl;

        int L = 10;
        Eigen::VectorXd pts_L = default_sampling_points(*(sve_result.u), L);
        REQUIRE(pts_L.size() == L);
        /*
        Eigen::VectorXd pts_100 = default_sampling_points(sve_result.u, 100);
        REQUIRE(pts_100.size() == 24);
        */
    }

    SECTION("U_l(beta) > 0")
    {
        double betas[] = {1.0, 10.0, 100.0};
        double omega_max = 10.0;
        double epsilon = 1e-10;
        for (double beta : betas) {
            auto kernel = sparseir::LogisticKernel(beta * omega_max);
            auto sve_result = SVECache::get_sve_result(kernel, 1e-15);

            auto basis = make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
                beta, omega_max, 1e-15, kernel, sve_result);
    
            auto ulx_tau0 = (*basis->u)(beta);
            for (int l = 0; l < basis->size(); ++l) {
                REQUIRE(ulx_tau0[l] > 0.0);
            }
        }
    }

    SECTION("LogisticKernel reconstruction, epsilon = 1e-10")
    {
        double beta = 10.0;
        double omega_max = 10.0;
        double epsilon = 1e-10;

        auto kernel = sparseir::LogisticKernel(beta * omega_max);
        auto sve_result = SVECache::get_sve_result(kernel, epsilon);
        auto basis_f =
            make_shared<sparseir::FiniteTempBasis<sparseir::Fermionic>>(
                beta, omega_max, epsilon, kernel, sve_result);

        auto taus = std::vector<double>({-beta, -0.0, 0.0, 1e-1 * beta, 0.9 * beta, beta});
        auto taus_regularized = std::vector<double>({0.0, beta, 0.0, 1e-1 * beta, 0.9 * beta, beta});
        auto signs = std::vector<double>({-1.0, -1.0, 1.0, 1.0, 1.0, 1.0});

        REQUIRE(taus.size() == signs.size());

        for (int i = 0; i < taus.size(); ++i) {
            double tau = taus[i];
            double tau_regularized = taus_regularized[i];
            double sign = signs[i];
            for (double omega :
                 //{1e-1 * omega_max, 0.5 * omega_max, 0.9 * omega_max}) {
                 {0.9 * omega_max}) {
                double x = 2.0 * tau_regularized / beta - 1.0;
                double y = omega / omega_max;

                // Compute kernel value directly
                double kernel_value = kernel.compute(x, y);

                // Compute reconstruction using basis functions
                double reconstruction = 0.0;
                for (int l = 0; l < basis_f->size(); ++l) {
                    auto ulx = (*basis_f->u)[l];
                    auto vly = (*basis_f->v)[l];
                    auto ulx_tau = ulx(tau);
                    auto vly_omega = vly(omega);

                    reconstruction += basis_f->s[l] * ulx_tau[0] * vly_omega;
                }

                REQUIRE(std::abs(kernel_value * sign - reconstruction) < 10 * epsilon);
            }
        }
    }

    SECTION("RegularizedBoseKernel reconstruction, epsilon = 1e-10")
    {
        double beta = 10.0;
        double omega_max = 10.0;
        double epsilon = 1e-10;

        auto kernel = sparseir::RegularizedBoseKernel(beta * omega_max);
        auto sve_result = SVECache::get_sve_result(kernel, epsilon);
        auto basis_b =
            make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
                beta, omega_max, epsilon, kernel, sve_result);

        for (double tau :
             {1e-1 * beta, 0.5 * beta, 0.9 * beta}) {
            for (double omega :
                 {1e-1 * omega_max, 0.5 * omega_max, 0.9 * omega_max}) {
                double x = 2.0 * tau / beta - 1.0;
                double y = omega / omega_max;

                // Compute kernel value directly
                double kernel_value = omega_max * kernel.compute(x, y);

                // Compute reconstruction using basis functions
                double reconstruction = 0.0;
                for (int l = 0; l < basis_b->size(); ++l) {
                    auto ulx = (*basis_b->u)[l];
                    auto vly = (*basis_b->v)[l];
                    auto ulx_tau = ulx(tau);
                    auto vly_omega = vly(omega);
                    reconstruction += basis_b->s[l] * ulx_tau[0] * vly_omega;
                }

                REQUIRE(std::abs(kernel_value - reconstruction) < 10 * epsilon);
            }
        }
    }
}

TEST_CASE("FiniteTempBasis error handling", "[basis]")
{

    double beta = 2.0;
    double omega_max = 5.0;
    double epsilon = 1e-10;
    auto rbk = sparseir::RegularizedBoseKernel(10.0);
    auto sve_result = SVECache::get_sve_result(rbk, epsilon);

    double wrong_beta = 120.0;
    REQUIRE_THROWS_AS(sparseir::FiniteTempBasis<sparseir::Bosonic>(
                          wrong_beta, omega_max, rbk, sve_result),
                      std::runtime_error);

    double wrong_omega_max = 1000.;
    REQUIRE_THROWS_AS(sparseir::FiniteTempBasis<sparseir::Bosonic>(
                          beta, wrong_omega_max, rbk, sve_result),
                      std::runtime_error);

    double negative_beta = -10.0;
    REQUIRE_THROWS_AS(sparseir::FiniteTempBasis<sparseir::Bosonic>(
                          negative_beta, omega_max, rbk, sve_result),
                      std::domain_error);

    double negative_wmax = -10.0;
    REQUIRE_THROWS_AS(sparseir::FiniteTempBasis<sparseir::Bosonic>(
                          beta, negative_wmax, rbk, sve_result),
                      std::domain_error);
}
