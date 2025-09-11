#pragma once

// C++ Standard Library headers
#include <tuple>
#include <string>

// Eigen headers
#include <Eigen/Dense>

namespace sparseir {

using namespace Eigen;

template <typename T>
std::tuple<MatrixX<T>, VectorX<T>, MatrixX<T>>
compute_svd(const MatrixX<T> &A, int n_sv_hint = 0,
            std::string strategy = "default");

} // namespace sparseir
