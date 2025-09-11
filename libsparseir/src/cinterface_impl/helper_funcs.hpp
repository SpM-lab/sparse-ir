#include <complex.h>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include "_util.hpp"

inline bool is_dlr_basis(const spir_basis *b) {
    return std::dynamic_pointer_cast<DLRAdapter<sparseir::Fermionic>>(get_impl_basis(b)) != nullptr ||
           std::dynamic_pointer_cast<DLRAdapter<sparseir::Bosonic>>(get_impl_basis(b)) != nullptr;
}

inline bool is_ir_basis(const spir_basis *b) {
    return std::dynamic_pointer_cast<_IRBasis<sparseir::Fermionic>>(get_impl_basis(b)) != nullptr ||
           std::dynamic_pointer_cast<_IRBasis<sparseir::Bosonic>>(get_impl_basis(b)) != nullptr;
}

// Helper function to convert N-dimensional array to 3D array by collapsing
// dimensions
static std::array<int, 3> collapse_to_3d(int ndim, const int *dims,
                                             int target_dim)
{
    std::array<int, 3> dims_3d = {1, dims[target_dim], 1};
    // Multiply all dimensions before target_dim into first dimension
    for (int i = 0; i < target_dim; ++i) {
        dims_3d[0] *= dims[i];
    }
    // Multiply all dimensions after target_dim into last dimension
    for (int i = target_dim + 1; i < ndim; ++i) {
        dims_3d[2] *= dims[i];
    }
    return dims_3d;
}

// Template function to handle all evaluation cases - moved outside extern "C"
// block
template <typename InputScalar, typename OutputScalar>
static int
evaluate_impl(const spir_sampling *s, int order, int ndim,
              const int *input_dims, int target_dim, const InputScalar *input,
              OutputScalar *out,
              int (sparseir::AbstractSampling::*eval_func)(
                  const Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> &,
                  int, Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> &)
                  const)
{
    auto impl = get_impl_sampling(s);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    // Convert dimensions
    std::array<int, 3> dims_3d =
        collapse_to_3d(ndim, input_dims, target_dim);

    // output ndim, target_dim, dims_3d
    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::array<int, 3> input_dims_3d = dims_3d;
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());
        std::array<int, 3> output_dims_3d = input_dims_3d;
        output_dims_3d[1] = static_cast<int>(impl.get()->n_sampling_points());

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(
            input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(
            out, output_dims_3d);
        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    } else {
        std::array<int, 3> input_dims_3d = dims_3d;
        std::array<int, 3> output_dims_3d = input_dims_3d;
        output_dims_3d[1] = static_cast<int>(impl.get()->n_sampling_points());

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(
            input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(
            out, output_dims_3d);
        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    }
}

template <typename InputScalar, typename OutputScalar>
static int
fit_impl(const spir_sampling *s, int order, int ndim,
         const int *input_dims, int target_dim, const InputScalar *input,
         OutputScalar *out,
         int (sparseir::AbstractSampling::*eval_func)(
             const Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> &, int,
             Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> &) const)
{
    auto impl = get_impl_sampling(s);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;
    if (!impl.get())
        return SPIR_GET_IMPL_FAILED;

    // Convert dimensions
    std::array<int, 3> dims_3d =
        collapse_to_3d(ndim, input_dims, target_dim);

    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::array<int, 3> input_dims_3d = dims_3d;
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());

        std::array<int, 3> output_dims_3d = input_dims_3d;
        output_dims_3d[1] = static_cast<int>(impl.get()->basis_size());

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(
            input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(
            out, output_dims_3d);
        // Convert to column-major order for Eigen
        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    } else {
        std::array<int, 3> input_dims_3d = dims_3d;
        std::array<int, 3> output_dims_3d = input_dims_3d;
        output_dims_3d[1] = static_cast<int>(impl.get()->basis_size());

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(
            input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(
            out, output_dims_3d);

        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    }
}

template <typename S>
spir_funcs *_create_ir_tau_funcs(std::shared_ptr<sparseir::IRTauFuncsType<S>> impl, double beta)
{
    return create_funcs(_safe_static_pointer_cast<AbstractContinuousFunctions>(
        std::make_shared<TauFunctionsAdaptor<sparseir::IRTauFuncsType<S>>>(impl, beta)));
}

template <typename S>
spir_funcs *_create_dlr_tau_funcs(std::shared_ptr<sparseir::DLRTauFuncsType<S>> impl, double beta)
{
    return create_funcs(_safe_static_pointer_cast<AbstractContinuousFunctions>(
        std::make_shared<TauFunctionsAdaptor<sparseir::DLRTauFuncsType<S>>>(impl, beta)));
}

template <typename InternalType>
spir_funcs *_create_omega_funcs(std::shared_ptr<InternalType> impl)
{
    return create_funcs(_safe_static_pointer_cast<AbstractContinuousFunctions>(
        std::make_shared<OmegaFunctionsAdaptor<InternalType>>(impl)));
}

template <typename S, typename T>
int spir_dlr2ir(const spir_basis *dlr, int order, int ndim,
                       const int *input_dims, int target_dim, const T *input, T *out)
{
    std::shared_ptr<DLRAdapter<S>> impl =
        std::dynamic_pointer_cast<DLRAdapter<S>>(get_impl_basis(dlr));
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    std::array<int, 3> input_dims_3d = collapse_to_3d(ndim, input_dims, target_dim);
    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());
    }
    Eigen::Tensor<T, 3> input_tensor(input_dims_3d[0], input_dims_3d[1], input_dims_3d[2]);
    std::size_t total_input_size = input_dims_3d[0] * input_dims_3d[1] * input_dims_3d[2];
    for (std::size_t i = 0; i < total_input_size; i++) {
        input_tensor.data()[i] = input[i];
    }
    Eigen::Tensor<T, 3> out_tensor = impl->get_impl()->to_IR(input_tensor, 1);
    std::size_t total_output_size =
        out_tensor.dimension(0) * out_tensor.dimension(1) * out_tensor.dimension(2);
    for (std::size_t i = 0; i < total_output_size; i++) {
        out[i] = out_tensor.data()[i];
    }
    return SPIR_COMPUTATION_SUCCESS;
}

template <typename S, typename T>
int spir_ir2dlr(const spir_basis *dlr, int order,
                         int ndim, const int *input_dims, int target_dim,
                         const T *input, T *out)
{
    std::shared_ptr<DLRAdapter<S>> impl =
        std::dynamic_pointer_cast<DLRAdapter<S>>(get_impl_basis(dlr));
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    std::array<int, 3> input_dims_3d = collapse_to_3d(ndim, input_dims, target_dim);
    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());
    }

    Eigen::Tensor<T, 3> input_tensor_3d(input_dims_3d[0], input_dims_3d[1], input_dims_3d[2]);
    std::size_t total_input_size = input_dims_3d[0] * input_dims_3d[1] * input_dims_3d[2];
    for (std::size_t i = 0; i < total_input_size; i++) {
        input_tensor_3d.data()[i] = input[i];
    }
    // move the target dimension to the first dimension
    input_tensor_3d = sparseir::movedim(input_tensor_3d, 1, 0);
    // reshape to 2D
    Eigen::array<Eigen::Index, 2> input2d_dims{input_dims_3d[1], input_dims_3d[0] * input_dims_3d[2]};
    Eigen::Tensor<T, 2> input_tensor_2d = input_tensor_3d.reshape(input2d_dims);
    Eigen::Tensor<T, 2> out_tensor_2d = impl->get_impl()->from_IR(input_tensor_2d);
    // move the target dimension to the last dimension
    // reshape to 3D
    Eigen::array<Eigen::Index, 3> out3d_dims{out_tensor_2d.dimension(0), input_dims_3d[0], input_dims_3d[2]};

    Eigen::Tensor<T, 3> out_tensor_3d_ = out_tensor_2d.reshape(out3d_dims);
    Eigen::Tensor<T, 3> out_tensor_3d = sparseir::movedim(out_tensor_3d_, 0, 1);

    // pass data to out
    std::size_t total_output_size =
        out_tensor_3d.dimension(0) * out_tensor_3d.dimension(1) * out_tensor_3d.dimension(2);
    for (std::size_t i = 0; i < total_output_size; i++) {
        out[i] = out_tensor_3d.data()[i];
    }
    return SPIR_COMPUTATION_SUCCESS;
}

template<typename SMPL>
spir_sampling* _spir_tau_sampling_new_with_points(const spir_basis *b, int num_points, const double *points) {
    if (num_points <= 0) {
        DEBUG_LOG("Error: Number of points must be positive");
        return nullptr;
    }

    // Get basis functions
    int status;
    spir_funcs* u = spir_basis_get_u(b, &status);
    if (!u) {
        DEBUG_LOG("Error: Failed to get basis functions");
        return nullptr;
    }

    // Get basis size
    int basis_size;
    status = spir_basis_get_size(b, &basis_size);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        DEBUG_LOG("Error: Failed to get basis size");
        spir_funcs_release(u);
        return nullptr;
    }

    // Evaluate basis functions at sampling points
    Eigen::MatrixXd matrix(num_points, basis_size);
    status = spir_funcs_batch_eval(u, SPIR_ORDER_COLUMN_MAJOR, num_points, points, matrix.data());
    if (status != SPIR_COMPUTATION_SUCCESS) {
        DEBUG_LOG("Error: Failed to evaluate basis functions");
        spir_funcs_release(u);
        return nullptr;
    }

    int statistics;
    int stats_status = spir_basis_get_stats(b, &statistics);
    if (stats_status != SPIR_COMPUTATION_SUCCESS) {
        DEBUG_LOG("Error: Failed to get basis statistics");
        spir_funcs_release(u);
        return nullptr;
    }

    // Create sampling object using the matrix version
    spir_sampling* smpl = spir_tau_sampling_new_with_matrix(
        SPIR_ORDER_COLUMN_MAJOR,
        statistics,
        basis_size,
        num_points, points, matrix.data(), &status);

    spir_funcs_release(u);

    return smpl;
}

template<typename SMPL>
spir_sampling* _spir_tau_sampling_new_with_matrix(int order,
                                                int basis_size,
                                                int num_points, const double *points,
                                                const double *matrix, int *status) {
    if (num_points <= 0) {
        DEBUG_LOG("Error: Number of points must be positive");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    // Create sampling points vector
    Eigen::VectorXd sampling_points(num_points);
    for (int i = 0; i < num_points; i++) {
        sampling_points(i) = points[i];
    }

    // Create matrix from input data
    Eigen::MatrixXd eigen_matrix(num_points, basis_size);
    if (order == SPIR_ORDER_ROW_MAJOR) {
        // Convert row-major to column-major
        for (int i = 0; i < num_points; i++) {
            for (int j = 0; j < basis_size; j++) {
                eigen_matrix(i, j) = matrix[i * basis_size + j];
            }
        }
    } else {
        // Already column-major
        for (int i = 0; i < num_points * basis_size; i++) {
            eigen_matrix.data()[i] = matrix[i];
        }
    }

    // Create sampling object using the new constructor
    auto sampling = std::make_shared<SMPL>(eigen_matrix, sampling_points);
    *status = SPIR_COMPUTATION_SUCCESS;
    return create_sampling(std::static_pointer_cast<sparseir::AbstractSampling>(sampling));
}

template<typename S, typename SMPL>
spir_sampling* _spir_matsu_sampling_new_with_points(const spir_basis *b, bool positive_only, int num_points, const int64_t *points) {
    std::shared_ptr<AbstractFiniteTempBasis> impl = get_impl_basis(b);
    if (!impl)
        return nullptr;

    if (num_points <= 0) {
        DEBUG_LOG("Error: Number of points must be positive");
        return nullptr;
    }

    std::vector<sparseir::MatsubaraFreq<S>> matsubara_points;
    matsubara_points.reserve(num_points);
    for (int i = 0; i < num_points; i++) {
        matsubara_points.emplace_back(points[i]);
    }

    if (is_ir_basis(b)) {
        auto impl_finite_temp_basis = std::dynamic_pointer_cast<_IRBasis<S>>(impl)->get_impl();
        return create_sampling(std::static_pointer_cast<sparseir::AbstractSampling>(
            std::make_shared<SMPL>(impl_finite_temp_basis, matsubara_points, positive_only)));
    } else {
        auto impl_finite_temp_basis = std::dynamic_pointer_cast<DLRAdapter<S>>(impl)->get_impl();
        return create_sampling(std::static_pointer_cast<sparseir::AbstractSampling>(
            std::make_shared<SMPL>(impl_finite_temp_basis, matsubara_points, positive_only)));
    }
}

template<typename S, typename SMPL>
spir_sampling* _spir_matsu_sampling_new_with_matrix(int order,
                                                  int basis_size,
                                                  bool positive_only, int num_points,
                                                  const int64_t *points,
                                                  const c_complex *matrix,
                                                  int *status) {
    if (num_points <= 0) {
        DEBUG_LOG("Error: Number of points must be positive");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    // Create sampling points vector
    std::vector<sparseir::MatsubaraFreq<S>> matsubara_points;
    matsubara_points.reserve(num_points);
    for (int i = 0; i < num_points; i++) {
        matsubara_points.emplace_back(points[i]);
    }

    // Create matrix from input data
    Eigen::MatrixXcd eigen_matrix(num_points, basis_size);
    if (order == SPIR_ORDER_ROW_MAJOR) {
        // Convert row-major to column-major
        for (int i = 0; i < num_points; i++) {
            for (int j = 0; j < basis_size; j++) {
                eigen_matrix(i, j) = *reinterpret_cast<const std::complex<double>*>(&matrix[i * basis_size + j]);
            }
        }
    } else {
        // Already column-major
        for (int i = 0; i < num_points * basis_size; i++) {
            eigen_matrix.data()[i] = *reinterpret_cast<const std::complex<double>*>(&matrix[i]);
        }
    }

    // Create sampling object using the new constructor
    auto sampling = std::make_shared<SMPL>(eigen_matrix, matsubara_points, positive_only);
    *status = SPIR_COMPUTATION_SUCCESS;
    return create_sampling(std::static_pointer_cast<sparseir::AbstractSampling>(sampling));
}

template <typename S>
spir_basis *_spir_dlr_new(const spir_basis *b)
{
    auto impl = get_impl_basis(b);
    if (!impl) {
        return nullptr;
    }

    if(!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return nullptr;
    }

    auto ptr_finite_temp_basis = _safe_static_pointer_cast<_IRBasis<S>>(impl)->get_impl();

    auto ptr_dlr = std::make_shared<DLRAdapter<S>>(
        std::make_shared<sparseir::DiscreteLehmannRepresentation<S>>(
            *ptr_finite_temp_basis));
    auto result = create_basis(std::static_pointer_cast<AbstractFiniteTempBasis>(ptr_dlr));
    return result;
}

template <typename S>
spir_basis *_spir_dlr_new_with_poles(const spir_basis *b,
                                   const int npoles, const double *poles)
{
    auto impl = get_impl_basis(b);
    if (!impl)
        return nullptr;

    if(!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return nullptr;
    }

    auto ptr_finite_temp_basis = _safe_static_pointer_cast<_IRBasis<S>>(impl)->get_impl();

    Eigen::VectorXd poles_vec(npoles);
    for (int i = 0; i < npoles; i++) {
        poles_vec(i) = poles[i];
    }

    auto ptr_dlr = std::make_shared<DLRAdapter<S>>(
        std::make_shared<sparseir::DiscreteLehmannRepresentation<S>>(
            *ptr_finite_temp_basis, poles_vec));
    return create_basis(std::static_pointer_cast<AbstractFiniteTempBasis>(ptr_dlr));
}

template <typename S>
int _spir_basis_get_u(const spir_basis *b,
                                      spir_funcs **u)
{
    try {
        std::shared_ptr<AbstractFiniteTempBasis> impl = get_impl_basis(b);
        if (!impl) {
            return SPIR_GET_IMPL_FAILED;
        }
        if (is_ir_basis(b)) {
            std::shared_ptr<sparseir::IRTauFuncsType<S>> u_impl = _safe_static_pointer_cast<_IRBasis<S>>(impl)->get_impl()->u;
            *u = _create_ir_tau_funcs<S>(u_impl, impl->get_beta());
        } else if (is_dlr_basis(b)) {
            std::shared_ptr<sparseir::DLRTauFuncsType<S>> u_impl = _safe_static_pointer_cast<DLRAdapter<S>>(impl)->get_impl()->u;
            *u = _create_dlr_tau_funcs<S>(u_impl, impl->get_beta());
        } else {
            return SPIR_NOT_SUPPORTED;
        }
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Error: " << e.what());
        return SPIR_GET_IMPL_FAILED;
    }
}

template <typename S>
int _spir_get_v(const spir_basis *b,
                                      spir_funcs **v)
{
    try {
        auto impl = get_impl_basis(b);
        if (!impl) {
            return SPIR_GET_IMPL_FAILED;
        }
        if (!is_ir_basis(b)) {
            DEBUG_LOG("Error: The basis is not an IR basis");
            return SPIR_NOT_SUPPORTED;
        }
        *v = _create_omega_funcs(impl->get_v());
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

template <typename S>
int _spir_basis_get_uhat(const spir_basis *b, spir_funcs **uhat)
{
    try {
        auto impl = get_impl_basis(b);
        if (!impl) {
            return SPIR_GET_IMPL_FAILED;
        }
        *uhat = create_funcs(impl->get_uhat());
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}


template <typename K>
spir_basis* _spir_basis_new(
    int statistics, double beta, double omega_max,
    const K& kernel, const spir_sve_result *sve, int max_size)
{
    try {
        auto sve_impl = get_impl_sve_result(sve);
        if (!sve_impl)
            return nullptr;
        if (statistics == SPIR_STATISTICS_FERMIONIC) {
            using FiniteTempBasisType =
                sparseir::FiniteTempBasis<sparseir::Fermionic>;
            auto impl = std::make_shared<FiniteTempBasisType>(
                beta, omega_max, kernel, *sve_impl, max_size);
            return create_basis(
                std::make_shared<_IRBasis<sparseir::Fermionic>>(impl));
        } else {
            using FiniteTempBasisType =
                sparseir::FiniteTempBasis<sparseir::Bosonic>;
            auto impl = std::make_shared<FiniteTempBasisType>(
                beta, omega_max, kernel, *sve_impl, max_size);
            return create_basis(
                std::make_shared<_IRBasis<sparseir::Bosonic>>(impl));
        }
    } catch (...) {
        return nullptr;
    }
}
