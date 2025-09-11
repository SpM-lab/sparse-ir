#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <numeric>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory>
#include "xprec/ddouble-header-only.hpp"

namespace sparseir {

namespace util {

// Implementation of make_unique for C++11
// This will be used when std::make_unique is not available
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace util

// Type trait to check if a type is a std::shared_ptr
template <typename T>
struct is_shared_ptr : std::false_type
{
};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type
{
};

// julia> sort = sortperm(s; rev=true)
// Implement sortperm in C++
std::vector<size_t> sortperm_rev(const Eigen::VectorXd &vec);

/*
This function ports Julia's implementation of the `invperm` function to C++.
*/
Eigen::VectorXi invperm(const Eigen::VectorXi &a);

template <typename Container>
bool issorted(const Container &c)
{
    if (c.size() <= 1) {
        return true;
    }

    return std::is_sorted(c.begin(), c.end());
}

// Overload for Eigen vectors
template <typename Derived>
bool issorted(const Eigen::MatrixBase<Derived> &vec)
{
    if (vec.size() <= 1) {
        return true;
    }

    for (Eigen::Index i = 1; i < vec.size(); ++i) {
        if (vec[i] < vec[i - 1]) {
            return false;
        }
    }
    return true;
}

template <typename T, int N>
bool tensorIsApprox(const Eigen::Tensor<T, N> &a, const Eigen::Tensor<T, N> &b,
                    double tol = 1e-12)
{
    // Ensure dimensions match (assuming dimensions() returns an
    // Eigen::DSizes<>)
    if (a.dimensions() != b.dimensions()) {
        return false;
    }
    for (int i = 0; i < a.size(); ++i) {
        // Compare the absolute difference of each complex number.
        if (std::abs(a.data()[i] - b.data()[i]) > tol) {
            return false;
        }
    }
    return true;
}

template <typename T>
inline std::vector<T> diff(const std::vector<T> &xs)
{
    std::vector<T> diff(xs.size() - 1);
    for (size_t i = 0; i < xs.size() - 1; ++i) {
        diff[i] = xs[i + 1] - xs[i];
    }
    return diff;
}

template <typename T>
inline Eigen::VectorX<T> diff(const Eigen::VectorX<T> &xs)
{
    Eigen::VectorX<T> diff(xs.size() - 1);
    for (Eigen::Index i = 0; i < xs.size() - 1; ++i) {
        diff[i] = xs[i + 1] - xs[i];
    }
    return diff;
}

template <typename T>
inline T sqrt_impl(const T &x)
{
    return std::sqrt(x);
}

// Specialization for DDouble
template <>
inline xprec::DDouble sqrt_impl(const xprec::DDouble &x)
{
    return xprec::sqrt(x);
}

template <typename T>
inline T cosh_impl(const T &x)
{
    return std::cosh(x);
}

template <>
inline xprec::DDouble cosh_impl(const xprec::DDouble &x)
{
    return xprec::cosh(x);
}

template <typename T>
inline T sinh_impl(const T &x)
{
    return std::sinh(x);
}

template <>
inline xprec::DDouble sinh_impl(const xprec::DDouble &x)
{
    return xprec::sinh(x);
}

template <typename T>
inline T exp_impl(const T &x)
{
    return std::exp(x);
}

template <>
inline xprec::DDouble exp_impl(const xprec::DDouble &x)
{
    return xprec::exp(x);
}

template <int N>
Eigen::array<int, N> getperm(int src, int dst)
{
    Eigen::array<int, N> perm;
    if (src == dst) {
        for (int i = 0; i < N; ++i) {
            perm[i] = i;
        }
        return perm;
    }

    int pos = 0;
    for (int i = 0; i < N; ++i) {
        if (i == dst) {
            perm[i] = src;
        } else {
            // Skip src position
            if (pos == src)
                ++pos;
            perm[i] = pos;
            ++pos;
        }
    }
    return perm;
}

template <typename T, int N>
Eigen::Tensor<T, N> movedim(const Eigen::Tensor<T, N> &arr, int src, int dst)
{
    if (src == dst) {
        return arr;
    }
    auto perm = getperm<N>(src, dst);
    return arr.shuffle(perm);
}

template <typename T, int N, int Options>
Eigen::Tensor<T, N, Options> movedim(const Eigen::Tensor<T, N, Options> &arr, int src, int dst)
{
    if (src == dst) {
        return arr;
    }
    auto perm = getperm<N>(src, dst);
    return arr.shuffle(perm);
}

} // namespace sparseir