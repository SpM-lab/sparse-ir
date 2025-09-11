#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "sparseir/sparseir-fwd.hpp"
#include "sparseir/gauss.hpp"
#include "sparseir/poly.hpp"
#include "sparseir/kernel.hpp"
#include "sparseir/svd.hpp"
namespace sparseir {

class SVEResult {
public:
    std::shared_ptr<PiecewiseLegendrePolyVector> u;
    Eigen::VectorXd s;
    std::shared_ptr<PiecewiseLegendrePolyVector> v;
    double epsilon;

    SVEResult();
    SVEResult(const PiecewiseLegendrePolyVector &u_, const Eigen::VectorXd &s_,
              const PiecewiseLegendrePolyVector &v_, double epsilon_);

    std::tuple<PiecewiseLegendrePolyVector, Eigen::VectorXd,
               PiecewiseLegendrePolyVector>
    part(double eps = std::numeric_limits<double>::quiet_NaN(),
         int max_size = -1) const;
};

std::tuple<double, std::string, std::string>
choose_accuracy(double epsilon, const std::string &Twork);

std::tuple<double, std::string, std::string> choose_accuracy(double epsilon,
                                                             std::nullptr_t);

std::tuple<double, std::string, std::string> choose_accuracy(std::nullptr_t,
                                                             std::string Twork);

std::tuple<double, std::string, std::string> choose_accuracy(std::nullptr_t,
                                                             std::nullptr_t);

std::tuple<double, std::string, std::string>
auto_choose_accuracy(double epsilon, std::string Twork,
                     std::string svd_strat = "auto");

void canonicalize(PiecewiseLegendrePolyVector &u,
                  PiecewiseLegendrePolyVector &v);

// Base class for SVE strategies
template <typename T>
class AbstractSVE {
public:
    virtual ~AbstractSVE() { }
    virtual std::vector<Eigen::MatrixX<T>> matrices() const = 0;
    virtual SVEResult
    postprocess(const std::vector<Eigen::MatrixX<T>> &u_list,
                const std::vector<Eigen::VectorX<T>> &s_list,
                const std::vector<Eigen::MatrixX<T>> &v_list) const = 0;
};

// SamplingSVE class declaration
template <typename K, typename T>
class SamplingSVE : public AbstractSVE<T> {
public:
    std::shared_ptr<AbstractKernel> kernel;
    double epsilon;
    int n_gauss;
    int nsvals_hint;
    Rule<T> rule;
    std::vector<T> segs_x;
    std::vector<T> segs_y;
    Rule<T> gauss_x;
    Rule<T> gauss_y;

    SamplingSVE(const K &kernel_, double epsilon_, int n_gauss_ = -1);
    SamplingSVE(const std::shared_ptr<AbstractKernel> &kernel_, double epsilon_,
                int n_gauss_ = -1);

    std::vector<Eigen::MatrixX<T>> matrices() const override;
    SVEResult
    postprocess(const std::vector<Eigen::MatrixX<T>> &u_list,
                const std::vector<Eigen::VectorX<T>> &s_list,
                const std::vector<Eigen::MatrixX<T>> &v_list) const override;
};

// CentrosymmSVE class declaration
template <typename K, typename T>
class CentrosymmSVE : public AbstractSVE<T> {
public:
    std::shared_ptr<AbstractKernel> kernel;
    double epsilon;
    SamplingSVE<
        typename SymmKernelTraits<K, std::integral_constant<int, +1>>::type, T>
        even;
    SamplingSVE<
        typename SymmKernelTraits<K, std::integral_constant<int, -1>>::type, T>
        odd;
    int nsvals_hint;

    CentrosymmSVE(const K &kernel_, double epsilon_, int n_gauss_ = -1);
    CentrosymmSVE(const std::shared_ptr<AbstractKernel> &kernel_,
                  double epsilon_, int n_gauss_ = -1);

    std::vector<Eigen::MatrixX<T>> matrices() const override;
    SVEResult
    postprocess(const std::vector<Eigen::MatrixX<T>> &u_list,
                const std::vector<Eigen::VectorX<T>> &s_list,
                const std::vector<Eigen::MatrixX<T>> &v_list) const override;
};


// Template function declarations
template <typename K, typename T>
std::shared_ptr<AbstractSVE<T>> determine_sve(const K &kernel,
                                              double safe_epsilon, int n_gauss);

//template <typename T>
//std::shared_ptr<AbstractSVE<T>>
//determine_sve(const std::shared_ptr<AbstractKernel> &kernel,
              //double safe_epsilon, int n_gauss);

template <typename T>
std::tuple<std::vector<Eigen::MatrixX<T>>, std::vector<Eigen::VectorX<T>>,
           std::vector<Eigen::MatrixX<T>>>
truncate(const std::vector<Eigen::MatrixX<T>> &u,
         const std::vector<Eigen::VectorX<T>> &s,
         const std::vector<Eigen::MatrixX<T>> &v, T rtol = 0.0,
         int lmax = std::numeric_limits<int>::max());

template <typename K, typename T>
std::tuple<SVEResult, std::shared_ptr<AbstractSVE<T>>>
pre_postprocess(const K &kernel, double safe_epsilon, int n_gauss,
                double cutoff = std::numeric_limits<double>::quiet_NaN(),
                int lmax = std::numeric_limits<int>::max());

//template <typename T>
//std::tuple<SVEResult, std::shared_ptr<AbstractSVE<T>>>
//pre_postprocess(const std::shared_ptr<AbstractKernel> &kernel,
                //double safe_epsilon, int n_gauss,
                ////double cutoff = std::numeric_limits<double>::quiet_NaN(),
                //int lmax = std::numeric_limits<int>::max());

// Restrict compute_sve to concrete kernel types only
template <typename K>
SVEResult compute_sve(const K &kernel, double epsilon,
            double cutoff = std::numeric_limits<double>::quiet_NaN(),
            int lmax = std::numeric_limits<int>::max(),
            int n_gauss = -1, std::string Twork = "Float64x2");


} // namespace sparseir
