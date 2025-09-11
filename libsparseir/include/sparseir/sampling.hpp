#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/SVD>
#include <memory>
#include <stdexcept>
#include <vector>
#include <complex>
#include <tuple>

namespace sparseir {

// Forward declarations
template <typename S>
class FiniteTempBasis;
template <typename S>
class DiscreteLehmannRepresentation;

template <typename T1, typename T2, int N1, int N2>
Eigen::Tensor<decltype(T1() * T2()), (N1 + N2 - 2)>
_contract(const Eigen::TensorMap<const Eigen::Tensor<T1, N1>> &tensor1,
          const Eigen::TensorMap<const Eigen::Tensor<T2, N2>> &tensor2,
          const Eigen::array<Eigen::IndexPair<int>, 1> &contract_dims)
{
    using ResultType = decltype(T1() * T2());

    // Contract tensors with proper type casting
    auto tensor1_cast = tensor1.template cast<ResultType>();
    auto tensor2_cast = tensor2.template cast<ResultType>();
    auto result = tensor1_cast.contract(tensor2_cast, contract_dims);

    return result;
}

// Overload for regular Tensor in both arguments - delegates to TensorMap
// version
template <typename T1, typename T2, int N1, int N2>
Eigen::Tensor<decltype(T1() * T2()), (N1 + N2 - 2)>
_contract(const Eigen::Tensor<T1, N1> &tensor1,
          const Eigen::Tensor<T2, N2> &tensor2,
          const Eigen::array<Eigen::IndexPair<int>, 1> &contract_dims)
{
    // Create TensorMaps from the Tensors
    Eigen::TensorMap<Eigen::Tensor<T1, N1>> tensor1_map(
        const_cast<T1 *>(tensor1.data()), tensor1.dimensions());
    Eigen::TensorMap<Eigen::Tensor<T2, N2>> tensor2_map(
        const_cast<T2 *>(tensor2.data()), tensor2.dimensions());

    // Delegate to the TensorMap version
    return _contract(tensor1_map, tensor2_map, contract_dims);
}

// Overload for TensorMap in first argument, regular Tensor in second
template <typename T1, typename T2, int N1, int N2>
Eigen::Tensor<decltype(T1() * T2()), (N1 + N2 - 2)>
_contract(const Eigen::TensorMap<const Eigen::Tensor<T1, N1>> &tensor1,
          const Eigen::Tensor<T2, N2> &tensor2,
          const Eigen::array<Eigen::IndexPair<int>, 1> &contract_dims)
{
    // Create TensorMap from the second Tensor
    Eigen::TensorMap<const Eigen::Tensor<T2, N2>> tensor2_map(
        const_cast<T2 *>(tensor2.data()), tensor2.dimensions());

    // Delegate to the TensorMap version
    return _contract(tensor1, tensor2_map, contract_dims);
}

// Overload for regular Tensor in first argument, TensorMap in second
template <typename T1, typename T2, int N1, int N2>
Eigen::Tensor<decltype(T1() * T2()), (N1 + N2 - 2)>
_contract(const Eigen::Tensor<T1, N1> &tensor1,
          const Eigen::TensorMap<const Eigen::Tensor<T2, N2>> &tensor2,
          const Eigen::array<Eigen::IndexPair<int>, 1> &contract_dims)
{
    // Create TensorMap from the first Tensor
    Eigen::TensorMap<const Eigen::Tensor<T1, N1>> tensor1_map(
        const_cast<T1 *>(tensor1.data()), tensor1.dimensions());

    // Delegate to the TensorMap version
    return _contract(tensor1_map, tensor2, contract_dims);
}

template <typename T1, typename T2, int N2>
Eigen::Tensor<decltype(T1() * T2()), N2> _matop_along_dim(
    const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const Eigen::Tensor<T2, N2> &tensor2, int dim = 0)
{
    if (dim < 0 || dim >= N2) {
        throw std::runtime_error(
            "evaluate: dimension must be in [0..N2). Got dim=" +
            std::to_string(dim));
    }

    if (matrix.cols() != tensor2.dimension(dim)) {
        throw std::runtime_error(
            "Mismatch: matrix.cols()=" + std::to_string(matrix.cols()) +
            ", but tensor2.dimension(" + std::to_string(dim) +
            ")=" + std::to_string(tensor2.dimension(dim)));
    }

    // Create a temporary tensor from the matrix
    Eigen::Tensor<T1, 2> matrix_tensor(matrix.rows(), matrix.cols());
    // Copy data from matrix to tensor
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            matrix_tensor(i, j) = matrix(i, j);
        }
    }

    // specify contraction dimensions
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
        Eigen::IndexPair<int>(1, dim)};

    auto result = _contract(matrix_tensor, tensor2, contract_dims);
    return movedim(result, 0, dim);
}

// Add overload for TensorMap
template <typename T1, typename T2, int N2>
Eigen::Tensor<decltype(T1() * T2()), N2> _matop_along_dim(
    const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const Eigen::TensorMap<const Eigen::Tensor<T2, N2>> &tensor2, int dim = 0)
{
    if (dim < 0 || dim >= N2) {
        throw std::runtime_error(
            "evaluate: dimension must be in [0..N2). Got dim=" +
            std::to_string(dim));
    }

    if (matrix.cols() != tensor2.dimension(dim)) {
        throw std::runtime_error(
            "Mismatch: matrix.cols()=" + std::to_string(matrix.cols()) +
            ", but tensor2.dimension(" + std::to_string(dim) +
            ")=" + std::to_string(tensor2.dimension(dim)));
    }

    // Create a temporary tensor from the matrix
    Eigen::Tensor<T1, 2> matrix_tensor(matrix.rows(), matrix.cols());
    // Copy data from matrix to tensor
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            matrix_tensor(i, j) = matrix(i, j);
        }
    }

    // specify contraction dimensions
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
        Eigen::IndexPair<int>(1, dim)};

    auto result = _contract(matrix_tensor, tensor2, contract_dims);
    return movedim(result, 0, dim);
}

template <typename T, typename S, int N>
Eigen::Matrix<decltype(T() * S()), Eigen::Dynamic, Eigen::Dynamic>
_fit_impl_first_dim(const Eigen::JacobiSVD<Eigen::MatrixX<S>> &svd,
                    const Eigen::MatrixX<T> &B)
{
    using ResultType = decltype(T() * S());

    Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic> UHB =
        svd.matrixU().adjoint() * B;

    // Apply inverse singular values to the rows of UHB
    for (int i = 0; i < svd.singularValues().size(); ++i) {
        UHB.row(i) /= ResultType(svd.singularValues()(i));
    }
    return svd.matrixV() * UHB;
}

inline Eigen::MatrixXcd
_fit_impl_first_dim_split_svd(const Eigen::JacobiSVD<Eigen::MatrixXcd> &svd,
                              const Eigen::MatrixXcd &B, bool has_zero)
{
    auto U = svd.matrixU();

    Eigen::Index U_halfsize =
        U.rows() % 2 == 0 ? U.rows() / 2 : U.rows() / 2 + 1;

    Eigen::MatrixXcd U_realT;
    U_realT = U.block(0, 0, U_halfsize, U.cols()).transpose();

    // Create a properly sized matrix first
    Eigen::MatrixXcd U_imag = Eigen::MatrixXcd::Zero(U_halfsize, U.cols());

    // Get the blocks we need
    if (has_zero) {
        U_imag = Eigen::MatrixXcd::Zero(U_halfsize, U.cols());
        auto U_imag_ = U.block(U_halfsize, 0, U_halfsize - 1, U.cols());
        auto U_imag_1 = U.block(0, 0, 1, U.cols());

        // Now do the assignments
        U_imag.topRows(1) = U_imag_1;
        U_imag.bottomRows(U_imag_.rows()) = U_imag_;
    } else {
        U_imag = U.block(U_halfsize, 0, U_halfsize, U.cols());
    }

    auto U_imagT = U_imag.transpose();

    Eigen::MatrixXcd UHB = U_realT * B.real();
    UHB += U_imagT * B.imag();

    // Apply inverse singular values to the rows of UHB
    for (int i = 0; i < svd.singularValues().size(); ++i) {
        UHB.row(i) /= std::complex<double>(svd.singularValues()(i));
    }
    return svd.matrixV() * UHB;
}

template <typename T, typename S, int N>
Eigen::Tensor<decltype(T() * S()), N>
fit_impl(const Eigen::JacobiSVD<Eigen::MatrixX<S>> &svd,
         const Eigen::Tensor<T, N> &arr, int dim)
{
    if (dim < 0 || dim >= N) {
        throw std::domain_error("Dimension must be in [0, N).");
    }

    // First move the dimension to the first
    auto arr_ = movedim(arr, dim, 0);
    // Create a view of the tensor as a matrix
    Eigen::MatrixX<T> arr_view = Eigen::Map<Eigen::MatrixX<T>>(
        arr_.data(), arr_.dimension(0), arr_.size() / arr_.dimension(0));
    // output matrix size
    Eigen::MatrixX<T> result = _fit_impl_first_dim<T, S, N>(svd, arr_view);
    // Copy the result to a tensor
    Eigen::array<Eigen::Index, N> dims;
    dims[0] = result.rows();
    for (int i = 1; i < N; ++i) {
        dims[i] = arr_.dimension(i);
    }
    Eigen::Tensor<T, N> result_tensor(dims);
    std::copy(result.data(), result.data() + result.size(),
              result_tensor.data());

    return movedim(result_tensor, 0, dim);
}

template <int N>
Eigen::Tensor<std::complex<double>, N>
fit_impl_split_svd(const Eigen::JacobiSVD<Eigen::MatrixXcd> &svd,
                   const Eigen::Tensor<std::complex<double>, N> &arr, int dim,
                   bool has_zero)
{
    if (dim < 0 || dim >= N) {
        throw std::domain_error("Dimension must be in [0, N).");
    }

    // First move the dimension to the first
    auto arr_ = movedim(arr, dim, 0);
    // Create a view of the tensor as a matrix
    Eigen::MatrixXcd arr_view = Eigen::Map<Eigen::MatrixXcd>(
        arr_.data(), arr_.dimension(0), arr_.size() / arr_.dimension(0));
    Eigen::MatrixXcd result =
        _fit_impl_first_dim_split_svd(svd, arr_view, has_zero);
    // Copy the result to a tensor
    Eigen::array<Eigen::Index, N> dims;
    dims[0] = result.rows();
    for (int i = 1; i < N; ++i) {
        dims[i] = arr_.dimension(i);
    }
    Eigen::Tensor<std::complex<double>, N> result_tensor(dims);
    std::copy(result.data(), result.data() + result.size(),
              result_tensor.data());

    return movedim(result_tensor, 0, dim);
}

class AbstractSampling {
public:
    virtual ~AbstractSampling() = default;

    // Get the number of sampling points
    virtual int32_t n_sampling_points() const = 0;

    // Get the basis size
    virtual std::size_t basis_size() const = 0;

    // Get the condition number of the sampling matrix
    virtual double get_cond_num() const = 0;

    // Evaluate the basis functions at the sampling points with double input
    virtual int evaluate_inplace_dd(
        const Eigen::TensorMap<const Eigen::Tensor<double, 3>> & /*input*/, int /*dim*/,
        Eigen::TensorMap<Eigen::Tensor<double, 3>> & /*output*/) const
    {
        return SPIR_NOT_SUPPORTED;
    }

    // Evaluate the basis functions at the sampling points with complex input
    virtual int evaluate_inplace_zz(
        const Eigen::TensorMap<const Eigen::Tensor<std::complex<double>, 3>> & /*input*/,
        int /*dim*/,
        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 3>> & /*output*/) const
    {
        return SPIR_NOT_SUPPORTED;
    }

    // Evaluate the basis functions at the sampling points with double input and
    // complex output
    virtual int evaluate_inplace_dz(
        const Eigen::TensorMap<const Eigen::Tensor<double, 3>> & /*input*/, int /*dim*/,
        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 3>> & /*output*/) const
    {
        return SPIR_NOT_SUPPORTED;
    }

    // Fit basis coefficients from the sparse sampling points with double input
    virtual int fit_inplace_dd(
        const Eigen::TensorMap<const Eigen::Tensor<double, 3>> & /*input*/, int /*dim*/,
        Eigen::TensorMap<Eigen::Tensor<double, 3>> & /*output*/) const
    {
        return SPIR_NOT_SUPPORTED;
    }

    // Fit basis coefficients from the sparse sampling points with complex input
    virtual int fit_inplace_zz(
        const Eigen::TensorMap<const Eigen::Tensor<std::complex<double>, 3>> & /*input*/,
        int /*dim*/,
        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 3>> & /*output*/) const
    {
        return SPIR_NOT_SUPPORTED;
    }

    // Fit basis coefficients from the sparse sampling points with complex input
    // and double output
    virtual int fit_inplace_dz(
        const Eigen::TensorMap<const Eigen::Tensor<std::complex<double>, 3>> & /*input*/,
        int /*dim*/,
        Eigen::TensorMap<Eigen::Tensor<double, 3>> & /*output*/) const
    {
        return SPIR_NOT_SUPPORTED;
    }
};

// Helper function declarations
// Forward declarations
template <typename S>
class TauSampling;

template <typename S>
class MatsubaraSampling;

template <typename S>
class AugmentedBasis;

// Common implementation for evaluate_inplace
template <typename Sampler, typename InputScalar = double,
          typename OutputScalar = std::complex<double>, int Dim = 3>
int evaluate_inplace_impl(
    const Sampler &sampler,
    const Eigen::TensorMap<const Eigen::Tensor<InputScalar, Dim>> &input,
    int dim, Eigen::TensorMap<Eigen::Tensor<OutputScalar, Dim>> &output)
{

    if (dim < 0 || dim >= Dim) {
        // Invalid dimension
        return SPIR_INVALID_DIMENSION;
    }

    if (sampler.basis_size() !=
        static_cast<std::size_t>(input.dimension(dim))) {
        // Dimension mismatch
        return SPIR_INPUT_DIMENSION_MISMATCH;
    }

    // Calculate result using the existing evaluate method
    auto result = sampler.evaluate(input, dim);

    // Check if output dimensions match result dimensions
    if (output.dimensions() != result.dimensions()) {
        // Output tensor has wrong dimensions
        return SPIR_OUTPUT_DIMENSION_MISMATCH;
    }

    // Copy the result to the output tensor
    std::copy(result.data(), result.data() + result.size(), output.data());

    return SPIR_COMPUTATION_SUCCESS; // Success
}

// Common implementation for evaluate_inplace
template <typename Sampler, typename InputScalar = double,
          typename OutputScalar = std::complex<double>, int Dim = 3>
int fit_inplace_impl(
    const Sampler &sampler,
    const Eigen::TensorMap<const Eigen::Tensor<InputScalar, Dim>> &input,
    int dim, Eigen::TensorMap<Eigen::Tensor<OutputScalar, Dim>> &output)
{

    if (dim < 0 || dim >= Dim) {
        // Invalid dimension
        return SPIR_INVALID_DIMENSION;
    }

    if (sampler.n_sampling_points() !=
        static_cast<int32_t>(input.dimension(dim))) {
        // Dimension mismatch
        return SPIR_INPUT_DIMENSION_MISMATCH;
    }

    // Calculate result using the existing evaluate method
    auto result = sampler.fit(input, dim);

    // Check if output dimensions match result dimensions
    if (output.dimensions() != result.dimensions()) {
        // Output tensor has wrong dimensions
        return SPIR_OUTPUT_DIMENSION_MISMATCH;
    }

    // Copy the result to the output tensor
    std::copy(result.data(), result.data() + result.size(), output.data());

    return SPIR_COMPUTATION_SUCCESS; // Success
}

template <typename Basis>
inline Eigen::MatrixXd eval_matrix(const Basis &basis,
                                   const Eigen::VectorXd &x)
{
    // Initialize matrix with correct dimensions
    Eigen::MatrixXd matrix(x.size(), basis.size());

    // Evaluate basis functions at sampling points
    auto u_eval = (*basis.u)(x);
    // Transpose
    matrix = u_eval.transpose();

    return matrix;
}

// Return a matrix of size (sampling_points.size(), basis_size)
template <typename S, typename Base>
inline Eigen::MatrixXcd
eval_matrix(const Base &basis,
            const std::vector<MatsubaraFreq<S>> &sampling_points)
{
    // check if basis->uhat is a valid shared_ptr
    if (!(basis.uhat)) {
        throw std::runtime_error("uhat is not a valid shared_ptr");
    }
    Eigen::Vector<int64_t, Eigen::Dynamic> freqs(sampling_points.size());
    for (size_t i = 0; i < sampling_points.size(); ++i) {
        freqs(i) = sampling_points[i].get_n();
    }
    auto m = (*(basis.uhat))(freqs);
    return m.transpose();
}

template <typename S>
class TauSampling : public AbstractSampling {
private:
    Eigen::VectorXd sampling_points_;
    //a matrix of size (sampling_points.size(), basis_size)
    Eigen::MatrixXd matrix_;
    mutable std::shared_ptr<Eigen::JacobiSVD<Eigen::MatrixXd>> matrix_svd_;

public:
    template <typename Basis>
    TauSampling(const std::shared_ptr<Basis> &basis, const Eigen::VectorXd& sampling_points) : sampling_points_(sampling_points)
    {
        // Ensure matrix dimensions are correct
        if (sampling_points.size() == 0) {
            throw std::runtime_error("No sampling points given");
        }

        // Initialize evaluation matrix with correct dimensions
        matrix_ = eval_matrix(*basis, sampling_points);

        // Check matrix dimensions
        if (matrix_.rows() != sampling_points.size() ||
            matrix_.cols() != static_cast<long>(basis->size())) {
            throw std::runtime_error("Matrix dimensions mismatch: got " +
                                     std::to_string(matrix_.rows()) + "x" +
                                     std::to_string(matrix_.cols()) +
                                     ", expected " +
                                     std::to_string(sampling_points.size()) +
                                     "x" + std::to_string(basis->size()));
        }
    }

    // Constructor that takes matrix and sampling_points directly
    TauSampling(const Eigen::MatrixXd& matrix, const Eigen::VectorXd& sampling_points) 
        : sampling_points_(sampling_points), matrix_(matrix)
    {
        if (sampling_points.size() == 0 || matrix_.rows() != sampling_points.size()) {
            throw std::runtime_error("Matrix dimensions mismatch: got " +
                                     std::to_string(matrix_.rows()) + "x" +
                                     std::to_string(matrix_.cols()) +
                                     ", expected " +
                                     std::to_string(sampling_points.size()) +
                                     "x" + std::to_string(matrix_.cols()));
        }
    }

    // Implement the pure virtual method from AbstractSampling
    int32_t n_sampling_points() const override
    {
        return sampling_points_.size();
    }

    // Implement the pure virtual method from AbstractSampling
    std::size_t basis_size() const override
    {
        return static_cast<std::size_t>(this->matrix_.cols());
    }

    // Implement evaluate_inplace_dd method using the common implementation
    // Error code: -1: invalid dimension, -2: dimension mismatch, -3: type not
    // supported
    int evaluate_inplace_dd(
        const Eigen::TensorMap<const Eigen::Tensor<double, 3>> &input, int dim,
        Eigen::TensorMap<Eigen::Tensor<double, 3>> &output) const override
    {
        return evaluate_inplace_impl<TauSampling<S>, double, double, 3>(
            *this, input, dim, output);
    }

    // Implement fit_inplace_dd method using the common implementation
    // Error code: -1: invalid dimension, -2: dimension mismatch, -3: type not
    // supported
    int fit_inplace_dd(
        const Eigen::TensorMap<const Eigen::Tensor<double, 3>> &input, int dim,
        Eigen::TensorMap<Eigen::Tensor<double, 3>> &output) const override
    {
        return fit_inplace_impl<TauSampling<S>, double, double, 3>(*this, input,
                                                                   dim, output);
    }
    // Implement evaluate_inplace_dd method using the common implementation
    // Error code: -1: invalid dimension, -2: dimension mismatch, -3: type not
    // supported
    int evaluate_inplace_zz(
        const Eigen::TensorMap<const Eigen::Tensor<std::complex<double>, 3>>
            &input,
        int dim,
        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 3>> &output)
        const override
    {
        return evaluate_inplace_impl<TauSampling<S>, std::complex<double>,
                                     std::complex<double>, 3>(*this, input, dim,
                                                              output);
    }
    // Implement fit_inplace_zz method using the common implementation
    // Error code: -1: invalid dimension, -2: dimension mismatch, -3: type not
    // supported
    int fit_inplace_zz(
        const Eigen::TensorMap<const Eigen::Tensor<std::complex<double>, 3>>
            &input,
        int dim,
        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 3>> &output)
        const override
    {
        return fit_inplace_impl<TauSampling<S>, std::complex<double>,
                                std::complex<double>, 3>(*this, input, dim,
                                                         output);
    }

    template <typename Basis>
    TauSampling(const std::shared_ptr<Basis> &basis)
    {
        // Get default sampling points from basis
        sampling_points_ = basis->default_tau_sampling_points();

        *this = TauSampling(basis, sampling_points_);
    }

    // Add constructor that takes a direct reference to FiniteTempBasis<S>
    template <typename Basis, typename = typename std::enable_if<
                                  !is_shared_ptr<Basis>::value>::type>
    TauSampling(const Basis &basis)
        : TauSampling(std::make_shared<FiniteTempBasis<S>>(basis))
    {
    }

    // Evaluate the basis coefficients at sampling points
    template <typename T, int N>
    Eigen::Tensor<T, N>
    evaluate(const Eigen::TensorMap<const Eigen::Tensor<T, N>> &al,
             int dim = 0) const
    {
        return _matop_along_dim(matrix_, al, dim);
    }

    // Overload for Tensor (converts to TensorMap)
    template <typename T, int N>
    Eigen::Tensor<T, N> evaluate(const Eigen::Tensor<T, N> &al,
                                 int dim = 0) const
    {
        // Create a TensorMap from the Tensor
        Eigen::TensorMap<const Eigen::Tensor<T, N>> al_map(al.data(),
                                                           al.dimensions());
        return evaluate(al_map, dim);
    }

    // Fit values at sampling points to basis coefficients
    template <typename T, int N>
    Eigen::Tensor<T, N>
    fit(const Eigen::TensorMap<const Eigen::Tensor<T, N>> &ax,
        int dim = 0) const
    {
        if (dim < 0 || dim >= N) {
            throw std::runtime_error(
                "fit: dimension must be in [0..N). Got dim=" +
                std::to_string(dim));
        }
        auto svd = get_matrix_svd();
        return fit_impl<T, double, N>(svd, ax, dim);
    }

    // Overload for Tensor (converts to TensorMap)
    template <typename T, int N>
    Eigen::Tensor<T, N> fit(const Eigen::Tensor<T, N> &al, int dim = 0) const
    {
        // Create a TensorMap from the Tensor
        Eigen::TensorMap<const Eigen::Tensor<T, N>> al_map(al.data(),
                                                           al.dimensions());
        return fit(al_map, dim);
    }

    Eigen::MatrixXd get_matrix() const { return matrix_; }

    const Eigen::VectorXd &sampling_points() const { return sampling_points_; }

    const Eigen::VectorXd &tau() const { return sampling_points_; }
    const Eigen::JacobiSVD<Eigen::MatrixXd>& get_matrix_svd() const
    {
        if (!matrix_svd_) {
            matrix_svd_ = std::make_shared<Eigen::JacobiSVD<Eigen::MatrixXd>>(
                matrix_, Eigen::ComputeThinU | Eigen::ComputeThinV);
        }
        return *matrix_svd_;
    }

    // Get the condition number of the sampling matrix
    double get_cond_num() const override {
        auto _matrix_svd = get_matrix_svd();
        const auto& singular_values = _matrix_svd.singularValues();
        if (singular_values.size() == 0) {
            throw std::runtime_error("No singular values found");
        }
        return singular_values(0) / singular_values(singular_values.size() - 1);
    }
};

inline Eigen::JacobiSVD<Eigen::MatrixXcd>
make_split_svd(const Eigen::MatrixXcd &mat, bool has_zero = false)
{
    const int m = mat.rows(); // Number of rows in the input complex matrix
    const int n = mat.cols(); // Number of columns in the input complex matrix

    // Determine the starting row for the imaginary part.
    // If has_zero is true, skip the first row (offset = 1); otherwise, start at
    // row 0.
    const int offset_imag = has_zero ? 1 : 0;

    // Calculate the number of rows to be used from the imaginary part.
    const int imag_rows = m - offset_imag;

    // Total number of rows in the resulting real matrix:
    // all rows from the real part plus the selected rows from the imaginary
    // part.
    const int total_rows = m + imag_rows;

    // Create a real matrix with 'total_rows' rows and 'n' columns.
    Eigen::MatrixXcd rmat(total_rows, n);

    // Top part: assign the real part of the input matrix.
    rmat.topRows(m) = mat.real();

    // Bottom part: assign the selected block of the imaginary part.
    rmat.bottomRows(imag_rows) = mat.imag().block(offset_imag, 0, imag_rows, n);

    // Compute the SVD of the real matrix.
    // The options 'ComputeThinU' and 'ComputeThinV' compute the thin
    // (economical) versions of U and V.
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(rmat, Eigen::ComputeThinU |
                                                     Eigen::ComputeThinV);
    return svd;
}

template <typename S>
class MatsubaraSampling : public AbstractSampling {
private:
    std::vector<MatsubaraFreq<S>> sampling_points_;
    Eigen::MatrixXcd matrix_;
    mutable std::shared_ptr<Eigen::JacobiSVD<Eigen::MatrixXcd>> matrix_svd_;
    bool positive_only_;
    bool has_zero_;

public:
    // Implement the pure virtual method from AbstractSampling
    int32_t n_sampling_points() const override
    {
        return sampling_points_.size();
    }

    std::size_t basis_size() const override
    {
        return static_cast<std::size_t>(matrix_.cols());
    }

    // Implement evaluate_inplace_dz method using the common implementation
    // Error code: -1: invalid dimension, -2: dimension mismatch, -3: type not
    // supported
    int evaluate_inplace_dz(
        const Eigen::TensorMap<const Eigen::Tensor<double, 3>> &input, int dim,
        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 3>> &output)
        const override
    {
        return evaluate_inplace_impl<MatsubaraSampling<S>, double,
                                     std::complex<double>, 3>(*this, input, dim,
                                                              output);
    }

    // Implement evaluate_inplace_zz method using the common implementation
    // Error code: -1: invalid dimension, -2: dimension mismatch, -3: type not
    // supported
    int evaluate_inplace_zz(
        const Eigen::TensorMap<const Eigen::Tensor<std::complex<double>, 3>>
            &input,
        int dim,
        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 3>> &output)
        const override
    {
        return evaluate_inplace_impl<MatsubaraSampling<S>, std::complex<double>,
                                     std::complex<double>, 3>(*this, input, dim,
                                                              output);
    }

    // Implement fit_inplace_zz method using the common implementation
    // Error code: -1: invalid dimension, -2: dimension mismatch, -3: type not
    // supported
    int fit_inplace_zz(
        const Eigen::TensorMap<const Eigen::Tensor<std::complex<double>, 3>>
            &input,
        int dim,
        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 3>> &output)
        const override
    {
        return fit_inplace_impl<MatsubaraSampling<S>, std::complex<double>,
                                std::complex<double>, 3>(*this, input, dim,
                                                         output);
    }

    template <typename Basis>
    MatsubaraSampling(const std::shared_ptr<Basis> &basis,
                      const std::vector<MatsubaraFreq<S>> &sampling_points,
                      bool positive_only = false)
        : positive_only_(positive_only), has_zero_(false)
    {
        sampling_points_ = sampling_points;
        std::sort(sampling_points_.begin(), sampling_points_.end());

        // Ensure matrix dimensions are correct
        if (sampling_points_.size() == 0) {
            throw std::runtime_error("No sampling points given");
        }

        // Initialize evaluation matrix with correct dimensions
        matrix_ = eval_matrix(*basis, sampling_points_);

        // Check matrix dimensions
        if (matrix_.rows() != static_cast<long>(sampling_points_.size()) ||
            matrix_.cols() != static_cast<long>(basis->size())) {
            throw std::runtime_error("Matrix dimensions mismatch: got " +
                                     std::to_string(matrix_.rows()) + "x" +
                                     std::to_string(matrix_.cols()) +
                                     ", expected " +
                                     std::to_string(sampling_points_.size()) +
                                     "x" + std::to_string(basis->size()));
        }
        has_zero_ = sampling_points_[0].n == 0;
    }

    template <typename Basis>
    MatsubaraSampling(const std::shared_ptr<Basis> &basis,
                      bool positive_only = false)
        : positive_only_(positive_only), has_zero_(false)
    {
        // Get default sampling points from basis
        bool fence = false;
        sampling_points_ = basis->default_matsubara_sampling_points(
            basis->size(), fence, positive_only);
        std::sort(sampling_points_.begin(), sampling_points_.end());

        // Ensure matrix dimensions are correct
        if (sampling_points_.size() == 0) {
            throw std::runtime_error("No sampling points generated");
        }

        // Initialize evaluation matrix with correct dimensions
        matrix_ = eval_matrix(*basis, sampling_points_);

        // Check matrix dimensions
        if (matrix_.rows() != sampling_points_.size() ||
            matrix_.cols() != static_cast<std::size_t>(basis->size())) {
            throw std::runtime_error("Matrix dimensions mismatch: got " +
                                     std::to_string(matrix_.rows()) + "x" +
                                     std::to_string(matrix_.cols()) +
                                     ", expected " +
                                     std::to_string(sampling_points_.size()) +
                                     "x" + std::to_string(basis->size()));
        }
        has_zero_ = sampling_points_[0].n == 0;
    }

    // Add constructor that takes a direct reference to FiniteTempBasis<S>
    template <typename Basis, typename = typename std::enable_if<
                                  !is_shared_ptr<Basis>::value>::type>
    MatsubaraSampling(const Basis &basis, bool positive_only = false)
        : MatsubaraSampling(std::make_shared<Basis>(basis), positive_only)
    {
    }

    template <typename Basis, typename = typename std::enable_if<
                                  !is_shared_ptr<Basis>::value>::type>
    MatsubaraSampling(const Basis &basis,
                      const std::vector<MatsubaraFreq<S>> &sampling_points,
                      bool positive_only = false)
        : MatsubaraSampling(std::make_shared<Basis>(basis), sampling_points,
                            positive_only)
    {
    }

    // Constructor that takes matrix and sampling_points directly
    MatsubaraSampling(const Eigen::MatrixXcd& matrix, 
                      const std::vector<MatsubaraFreq<S>>& sampling_points,
                      bool positive_only = false)
        : sampling_points_(sampling_points), matrix_(matrix), 
          positive_only_(positive_only), has_zero_(false)
    {
        if (sampling_points.size() == 0 || matrix_.rows() != sampling_points.size()) {
            throw std::runtime_error("Matrix dimensions mismatch: got " +
                                     std::to_string(matrix_.rows()) + "x" +
                                     std::to_string(matrix_.cols()) +
                                     ", expected " +
                                     std::to_string(sampling_points.size()) +
                                     "x" + std::to_string(matrix_.cols()));
        }
        has_zero_ = sampling_points_[0].n == 0;
    }

    template <typename T, int N>
    Eigen::Tensor<std::complex<double>, N>
    evaluate(const Eigen::TensorMap<const Eigen::Tensor<T, N>> &al,
             int dim = 0) const
    {
        return _matop_along_dim(matrix_, al, dim);
    }

    // Overload for Tensor (converts to TensorMap)
    template <typename T, int N>
    Eigen::Tensor<std::complex<double>, N>
    evaluate(const Eigen::Tensor<T, N> &al, int dim = 0) const
    {
        // Create a TensorMap from the Tensor
        Eigen::TensorMap<const Eigen::Tensor<T, N>> al_map(al.data(),
                                                           al.dimensions());
        return evaluate(al_map, dim);
    }

    // Primary template for complex input tensors
    template <typename T, int N>
    Eigen::Tensor<std::complex<T>, N>
    fit(const Eigen::TensorMap<const Eigen::Tensor<std::complex<T>, N>> &ax,
        int dim = 0) const
    {
        if (dim < 0 || dim >= N) {
            throw std::runtime_error(
                "fit: dimension must be in [0..N). Got dim=" +
                std::to_string(dim));
        }
        auto svd = get_matrix_svd();
        if (positive_only_) {
            return fit_impl_split_svd<N>(svd, ax, dim, has_zero_);
        } else {
            return fit_impl<std::complex<T>, std::complex<T>, N>(svd, ax, dim);
        }
    }

    // Overload for Tensor (converts to TensorMap)
    template <typename T, int N>
    Eigen::Tensor<std::complex<double>, N> fit(const Eigen::Tensor<T, N> &al,
                                               int dim = 0) const
    {
        // Create a TensorMap from the Tensor
        Eigen::TensorMap<const Eigen::Tensor<T, N>> al_map(al.data(),
                                                           al.dimensions());
        return fit(al_map, dim);
    }

    Eigen::MatrixXcd get_matrix() const { return matrix_; }

    const std::vector<MatsubaraFreq<S>> &sampling_points() const
    {
        return sampling_points_;
    }

    const Eigen::JacobiSVD<Eigen::MatrixXcd>& get_matrix_svd() const
    {
        if (!matrix_svd_) {
            if (positive_only_) {
                matrix_svd_ = std::make_shared<Eigen::JacobiSVD<Eigen::MatrixXcd>>(
                    make_split_svd(matrix_, has_zero_));
            } else {
                matrix_svd_ = std::make_shared<Eigen::JacobiSVD<Eigen::MatrixXcd>>(
                    matrix_, Eigen::ComputeThinU | Eigen::ComputeThinV);
            }
        }
        return *matrix_svd_;
    }

    // Get the condition number of the sampling matrix
    double get_cond_num() const override {
        auto _matrix_svd = get_matrix_svd();
        const auto& singular_values = _matrix_svd.singularValues();
        if (singular_values.size() == 0) {
            throw std::runtime_error("No singular values found");
        }
        return singular_values(0) / singular_values(singular_values.size() - 1);
    }
};

} // namespace sparseir
