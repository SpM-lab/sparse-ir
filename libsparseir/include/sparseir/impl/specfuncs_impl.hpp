#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>

namespace sparseir {

template <typename T>
T legval(T x, const std::vector<T> &c)
{
    int nd = c.size();
    if (nd < 2) {
        return c.back();
    }

    T c0 = c[nd - 2];
    T c1 = c[nd - 1];
    for (int j = nd - 2; j >= 1; --j) {
        T k = T(j) / T(j + 1);
        T temp = c0;
        c0 = c[j - 1] - c1 * k;
        c1 = temp + c1 * x * (k + 1);
    }
    return c0 + c1 * x;
}

template <typename T>
Eigen::MatrixX<T> legvander(const Eigen::VectorX<T> &x, int deg)
{
    if (deg < 0) {
        throw std::domain_error("degree needs to be non-negative");
    }

    int n = x.size();
    Eigen::MatrixX<T> v(n, deg + 1);

    // Use forward recursion to generate the entries
    for (int i = 0; i < n; ++i) {
        v(i, 0) = T(1);
    }
    if (deg > 0) {
        for (int i = 0; i < n; ++i) {
            v(i, 1) = x[i];
        }
        for (int i = 2; i <= deg; ++i) {
            T invi = T(1) / T(i);
            for (int j = 0; j < deg + 1; ++j) {
                v(j, i) =
                    v(j, i - 1) * x[j] * (2 - invi) - v(j, i - 2) * (1 - invi);
            }
        }
    }

    return v;
}

// Add legder for accepting std::vector<T>
template <typename T>
Eigen::MatrixX<T> legvander(const std::vector<T> &x, int deg)
{
    Eigen::VectorX<T> x_eigen =
        Eigen::Map<const Eigen::VectorX<T>>(x.data(), x.size());
    return legvander(x_eigen, deg);
}

template <typename T>
Eigen::MatrixX<T> legder(Eigen::MatrixX<T> c, int cnt)
{
    if (cnt < 0) {
        throw std::domain_error(
            "The order of derivation needs to be non-negative");
    }
    if (cnt == 0) {
        return c;
    }

    int n = c.rows();
    int m = c.cols();
    if (cnt >= n) {
        return Eigen::MatrixX<T>::Zero(1, m);
    }

    for (int k = 0; k < cnt; ++k) {
        n -= 1;
        Eigen::MatrixX<T> der(n, m);
        for (int j = n; j >= 2; --j) {
            der.row(j - 1) = (2 * j - 1) * c.row(j);
            c.row(j - 2) += c.row(j);
        }
        if (n > 1) {
            der.row(1) = 3 * c.row(2);
        }
        der.row(0) = c.row(1);
        c = der;
    }
    return c;
}

} // namespace sparseir