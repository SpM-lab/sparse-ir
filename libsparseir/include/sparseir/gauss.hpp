#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "xprec/ddouble-header-only.hpp"

#include "sparseir/specfuncs.hpp"

namespace sparseir {

using xprec::DDouble;

class slice {
public:
    size_t start, stop, step;

    slice(size_t start, size_t stop, size_t step = 1)
        : start(start), stop(stop), step(step)
    {
    }

    std::vector<size_t> indices() const
    {
        std::vector<size_t> result;
        for (size_t i = start; i < stop; i += step) {
            result.push_back(i);
        }
        return result;
    }
};

/*
Quadrature rule.

    Approximation of an integral by a weighted sum over discrete points:

         ∫ f(x) * omega(x) * dx ~ sum(f(xi) * wi for (xi, wi) in zip(x, w))

    where we generally have superexponential convergence for smooth ``f(x)``
    with the number of quadrature points.
*/

template <typename T>
class Rule {
public:
    // TODO: Define x and w as Eigen::VectorXd
    Eigen::VectorX<T> x, w, x_forward, x_backward;
    T a, b;
    // Default constructor
    Rule() { };

    Rule(const std::vector<T> &x, const std::vector<T> &w, T a = -1, T b = 1)
        : x(Eigen::Map<const Eigen::VectorX<T>>(x.data(), x.size())),
          w(Eigen::Map<const Eigen::VectorX<T>>(w.data(), w.size())),
          a(a),
          b(b)
    {
        this->x_forward = Eigen::VectorX<T>::Zero(x.size());
        this->x_backward = Eigen::VectorX<T>::Zero(x.size());
        std::transform(this->x.data(), this->x.data() + this->x.size(),
                       this->x_forward.data(), [a](T xi) { return xi - a; });
        std::transform(this->x.data(), this->x.data() + this->x.size(),
                       this->x_backward.data(), [b](T xi) { return b - xi; });
    }

    // Constructor with x, w, x_forward, x_backward, a, b
    Rule(const Eigen::VectorX<T> &x, const Eigen::VectorX<T> &w,
         Eigen::VectorX<T> x_forward, Eigen::VectorX<T> x_backward, T a = -1,
         T b = 1)
        : x(x), w(w), x_forward(x_forward), x_backward(x_backward), a(a), b(b)
    {
    }
    // Constructor with x, w, a, b
    Rule(const Eigen::VectorX<T> &x, const Eigen::VectorX<T> &w, T a = -1,
         T b = 1)
        : x(x), w(w), a(a), b(b)
    {
        this->x_forward = Eigen::VectorX<T>::Zero(x.size());
        this->x_backward = Eigen::VectorX<T>::Zero(x.size());
        std::transform(this->x.data(), this->x.data() + this->x.size(),
                       this->x_forward.data(), [a](T xi) { return xi - a; });
        std::transform(this->x.data(), this->x.data() + this->x.size(),
                       this->x_backward.data(), [b](T xi) { return b - xi; });
    }

    template <typename U>
    Rule(const Rule<U> &other)
        : x(other.x.template cast<T>()),
          w(other.w.template cast<T>()),
          x_forward(other.x_forward.template cast<T>()),
          x_backward(other.x_backward.template cast<T>()),
          a(static_cast<T>(other.a)),
          b(static_cast<T>(other.b))
    {
    }

    Rule<T> reseat(T a, T b) const
    {
        T scaling = (b - a) / (this->b - this->a);
        Eigen::VectorX<T> new_x(x.size()), new_w(w.size()),
            new_x_forward(x_forward.size()), new_x_backward(x_backward.size());
        std::transform(x.data(), x.data() + x.size(), new_x.data(),
                       [this, scaling, a, b](T xi) {
                           return scaling * (xi - (this->b + this->a) / 2) +
                                  (b + a) / 2;
                       });
        std::transform(w.data(), w.data() + w.size(), new_w.data(),
                       [scaling](T wi) { return wi * scaling; });
        std::transform(x_forward.data(), x_forward.data() + x_forward.size(),
                       new_x_forward.data(),
                       [scaling](T xi) { return xi * scaling; });
        std::transform(x_backward.data(), x_backward.data() + x_backward.size(),
                       new_x_backward.data(),
                       [scaling](T xi) { return xi * scaling; });
        return Rule<T>(new_x, new_w, new_x_forward, new_x_backward, a, b);
    }

    Rule<T> scale(T factor) const
    {
        Eigen::VectorX<T> new_w(w.size());
        transform(w.data(), w.data() + w.size(), new_w.data(),
                  [factor](T wi) { return wi * factor; });
        return Rule<T>(x, new_w, x_forward, x_backward, a, b);
    }

    template <typename U>
    Rule<T> piecewise(const std::vector<U> &edges) const
    {
        if (!std::is_sorted(edges.begin(), edges.end())) {
            throw std::invalid_argument(
                "segments ends must be ordered ascendingly");
        }
        std::vector<Rule<T>> rules;
        for (size_t i = 0; i < edges.size() - 1; ++i) {
            auto rule_ = reseat(T(edges[i]), T(edges[i + 1]));
            rules.push_back(rule_);
        }
        return join(rules);
    }

    static Rule<T> join(const std::vector<Rule<T>> &gauss_list)
    {
        if (gauss_list.empty()) {
            return Rule<T>(Eigen::VectorX<T>(), Eigen::VectorX<T>());
        }

        T a = gauss_list.front().a;
        T b = gauss_list.back().b;
        T prev_b = a;
        Eigen::VectorX<T> x, w, x_forward, x_backward;

        for (const auto &curr : gauss_list) {
            if (curr.a != prev_b) {
                throw std::invalid_argument("Gauss rules must be ascending");
            }
            prev_b = curr.b;
            Eigen::VectorX<T> curr_x_forward(curr.x_forward.size()),
                curr_x_backward(curr.x_backward.size());
            std::transform(curr.x_forward.data(),
                           curr.x_forward.data() + curr.x_forward.size(),
                           curr_x_forward.data(),
                           [a, curr](T xi) { return xi + (curr.a - a); });
            std::transform(curr.x_backward.data(),
                           curr.x_backward.data() + curr.x_backward.size(),
                           curr_x_backward.data(),
                           [b, curr](T xi) { return xi + (b - curr.b); });
            x.conservativeResize(x.size() + curr.x.size());
            w.conservativeResize(w.size() + curr.w.size());
            x_forward.conservativeResize(x_forward.size() +
                                         curr_x_forward.size());
            x_backward.conservativeResize(x_backward.size() +
                                          curr_x_backward.size());
            x.tail(curr.x.size()) = curr.x;
            w.tail(curr.w.size()) = curr.w;
            x_forward.tail(curr_x_forward.size()) = curr_x_forward;
            x_backward.tail(curr_x_backward.size()) = curr_x_backward;
        }

        return Rule<T>(x, w, x_forward, x_backward, a, b);
    }
};

/*
    legendre(n[, T])

Gauss-Legendre quadrature with `n` points on [-1, 1].
*/
inline Rule<xprec::DDouble> legendre(int n)
{
    std::vector<xprec::DDouble> x(n), w(n);
    xprec::gauss_legendre(n, x.data(), w.data());
    // cast to T
    // Eigen::VectorX<T> new_x = Eigen::Map<Eigen::VectorX<T>>(x.data(),
    // x.size());
    Eigen::VectorX<xprec::DDouble> new_x(x.size());
    std::transform(x.begin(), x.end(), new_x.data(),
                   [](const xprec::DDouble &xi) { return xi; });
    Eigen::VectorX<xprec::DDouble> new_w(w.size());
    std::transform(w.begin(), w.end(), new_w.data(),
                   [](const xprec::DDouble &wi) { return wi; });
    return Rule<xprec::DDouble>(new_x, new_w);
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
legendre_collocation(const Rule<T> &rule, int n = -1)
{
    if (n < 0) {
        n = rule.x.size();
    }
    // Compute the Legendre Vandermonde matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> lv =
        sparseir::legvander<T>(rule.x, n - 1);
    for (auto i = 0; i < rule.w.size(); ++i) {
        for (auto j = 0; j < lv.cols(); ++j) {
            lv(i, j) *= rule.w[i];
        }
    }
    // !!! do NOT do this !!!
    // res = res.transpose();
    // This is the so-called aliasing issue. In "debug mode", i.e., when
    // assertions have not been disabled, such common pitfalls are automatically
    // detected.

    auto res = lv.transpose();

    // Normalize the matrix rows
    Eigen::VectorXd invnorm = Eigen::VectorXd::LinSpaced(n, 0.5, n - 0.5);

    for (auto i = 0; i < invnorm.size(); ++i) {
        for (auto j = 0; j < lv.cols(); ++j) {
            res(i, j) *= invnorm(i);
        }
    }

    return res;
}

template <typename TargetType, typename SourceType>
inline sparseir::Rule<TargetType>
convert_rule(const sparseir::Rule<SourceType> &rule)
{
    // Convert vectors using Eigen::Map to handle the conversion properly
    Eigen::VectorX<TargetType> x = rule.x.template cast<TargetType>();
    Eigen::VectorX<TargetType> w = rule.w.template cast<TargetType>();
    Eigen::VectorX<TargetType> x_forward =
        rule.x_forward.template cast<TargetType>();
    Eigen::VectorX<TargetType> x_backward =
        rule.x_backward.template cast<TargetType>();
    TargetType a = static_cast<TargetType>(rule.a);
    TargetType b = static_cast<TargetType>(rule.b);

    return sparseir::Rule<TargetType>(x, w, x_forward, x_backward, a, b);
}

} // namespace sparseir
