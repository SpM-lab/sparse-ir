#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>
#include <xprec/ddouble-header-only.hpp>

namespace sparseir {

template <typename T>
T legval(T x, const std::vector<T> &c);

template <typename T>
Eigen::MatrixX<T> legvander(const Eigen::VectorX<T> &x, int deg);

// Add legder for accepting std::vector<T>
template <typename T>
Eigen::MatrixX<T> legvander(const std::vector<T> &x, int deg);

template <typename T>
Eigen::MatrixX<T> legder(Eigen::MatrixX<T> c, int cnt = 1);

// Constants
const double PI = 3.14159265358979323846;

double SQPIO2(double /*x*/);

// Helper: evaluate polynomial using Horner's method.
double evalpoly(double x, const std::vector<double> &coeffs);

// Helper: compute sin(pi*x)
double sinpi(double x);

// Gamma function approximation (for real x)
double gamma_func(double _x);

// Cylindrical Bessel function of the first kind, J_nu(x)
// Uses the series expansion:
//   J_nu(x) = sum_{m=0}^∞ (-1)^m / (m! * Gamma(nu+m+1)) * (x/2)^(2m+nu)
double cyl_bessel_j(double nu, double x);

// sphericalbesselj_generic:
// Computes the spherical Bessel function j_n(x) using the relation:
//   j_n(x) = sqrt(pi/(2x)) * J_{n+1/2}(x)
double sphericalbesselj_generic(double nu, double x);

// sphericalbesselj_small_args:
// Approximation for small x.
double sphericalbesselj_small_args(double nu, double x);

// sphericalbesselj_small_args_cutoff:
// Determines when the small-argument expansion is accurate.
bool sphericalbesselj_small_args_cutoff(double nu, double x);

// besselj_ratio_jnu_jnum1:
// Computes the continued-fraction for the ratio J_{ν}(x) / J_{ν-1}(x)
double besselj_ratio_jnu_jnum1(double n, double x);

// sphericalbessely_forward_recurrence:
// Computes forward recurrence for spherical Bessel y.
// Returns a pair: (sY_{n-1}, sY_n)
std::pair<double, double> sphericalbessely_forward_recurrence(int nu, double x);

// sphericalbesselj_recurrence:
// Uses forward recurrence if stable; otherwise uses spherical Bessel y
// recurrence.
double sphericalbesselj_recurrence(int nu, double x);

// sphericalbesselj_positive_args:
// Selects the proper method for computing j_n(x) for positive arguments.
double sphericalbesselj_positive_args(int nu, double x);

// Main function to calculate spherical Bessel function of the first kind
double sphericalbesselj(int n, double x);

} // namespace sparseir