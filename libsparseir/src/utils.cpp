#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include <numeric>

namespace sparseir {

// julia> sort = sortperm(s; rev=true)
// Implement sortperm in C++
std::vector<size_t> sortperm_rev(const Eigen::VectorXd &vec)
{
    std::vector<size_t> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&vec](size_t i1, size_t i2) { return vec[i1] > vec[i2]; });
    return indices;
}

Eigen::VectorXi invperm(const Eigen::VectorXi &a)
{
    int n = a.size();
    Eigen::VectorXi b(n);
    b.setConstant(-1);

    for (int i = 0; i < n; i++) {
        int j = a(i);
        if (!(0 <= j && j < n) || b(j) != -1) {
            throw std::invalid_argument("invalid permutation");
        }
        b(j) = i;
    }
    return b;
}

} // namespace sparseir