#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

namespace sparseir {

// Midpoint function for floating-point types
template <typename T1, typename T2>
typename std::enable_if<std::is_floating_point<T1>::value &&
                            std::is_floating_point<T2>::value,
                        typename std::common_type<T1, T2>::type>::type
midpoint(T1 a, T2 b);

// Midpoint function for integral types
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type midpoint(T a, T b);

// For floating point types, returns true if absolute difference is within
// epsilon. For integer types, performs exact equality comparison
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
closeenough(T a, T b, T eps);

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
closeenough(T a, T b, T eps);

// Signbit function (handles both floating-point and integral types)
template <typename T>
inline bool signbit(T x);

// Bisection method to find root in interval [a,b] where f(a) and f(b) have
// opposite signs f: Function (T -> double) a, b: Interval endpoints fa: Value
// of f(a) eps: Error tolerance (for floating point)
template <typename T, typename F>
T bisect(const F &f, T a, T b, T fa, T eps);

template <typename F, typename T>
std::vector<T> find_all(F f, const std::vector<T> &xgrid);

template <typename T>
std::vector<T> refine_grid(const std::vector<T> &grid, int alpha);

template <typename F>
double bisect_discr_extremum(F absf, double a, double b, double absf_a,
                             double absf_b);

template <typename F, typename T>
std::vector<T> discrete_extrema(F f, const std::vector<T> &xgrid);

} // namespace sparseir