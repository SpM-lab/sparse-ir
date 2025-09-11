#pragma once

#include <iostream>
#include "../linalg.hpp"
#include "../svd.hpp"

namespace sparseir {

template <typename T>
std::tuple<MatrixX<T>, VectorX<T>, MatrixX<T>>
compute_svd(const MatrixX<T> &A, int n_sv_hint, std::string strategy)
{
    if (n_sv_hint != 0) {
        std::cout << "n_sv_hint is set but will not be used in the current "
                     "implementation!"
                  << std::endl;
    }

    if (strategy != "default") {
        std::cout << "strategy is set but will not be used in the current "
                     "implementation!"
                  << std::endl;
    }

    MatrixX<T> A_copy = A; // Create a copy of matrix A
    return tsvd<T>(A_copy);
}

} // namespace sparseir