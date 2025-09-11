#pragma once

#include <stdbool.h>
#include <complex.h>
#include <stdint.h>

#include "spir_status.h"

// Define a C-compatible type alias for the C99 complex number.
typedef double _Complex c_complex;

// Status codes
#define SPIR_COMPUTATION_SUCCESS 0
#define SPIR_GET_IMPL_FAILED -1
#define SPIR_INVALID_DIMENSION -2
#define SPIR_INPUT_DIMENSION_MISMATCH -3
#define SPIR_OUTPUT_DIMENSION_MISMATCH -4
#define SPIR_NOT_SUPPORTED -5
#define SPIR_INVALID_ARGUMENT -6
#define SPIR_INTERNAL_ERROR -7

// Statistics type constants
#define SPIR_STATISTICS_FERMIONIC 1
#define SPIR_STATISTICS_BOSONIC 0

// Order type constants
#define SPIR_ORDER_COLUMN_MAJOR 1
#define SPIR_ORDER_ROW_MAJOR 0

// Twork type constants
#define SPIR_TWORK_FLOAT64 0
#define SPIR_TWORK_FLOAT64X2 1

#ifdef __cplusplus
extern "C" {
#endif

/* Macro for declaring opaque types and their functions */
#define DECLARE_OPAQUE_TYPE(name)                                              \
    struct _spir_##name;                                                       \
    typedef struct _spir_##name spir_##name;                                   \
                                                                               \
    /* Destroy function */                                                     \
    void spir_##name##_release(spir_##name *obj);                              \
                                                                               \
    /* Clone function */                                                       \
    spir_##name *spir_##name##_clone(const spir_##name *src);                  \
                                                                               \
    /* Check if the shared_ptr is assigned to a valid object */                \
    int spir_##name##_is_assigned(const spir_##name *obj);                     \
                                                                               \
    /* Get the raw pointer to the shared_ptr (only for debugging) */           \
    void *_spir_##name##_get_raw_ptr(const spir_##name *obj);

/* Declare opaque types */
struct _spir_kernel;

DECLARE_OPAQUE_TYPE(kernel);
DECLARE_OPAQUE_TYPE(funcs);
DECLARE_OPAQUE_TYPE(basis);
DECLARE_OPAQUE_TYPE(sampling);
DECLARE_OPAQUE_TYPE(sve_result);

/**
 * @brief Creates a new logistic kernel for fermionic/bosonic analytical
 * continuation.
 *
 * In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is
 * a function on [-1, 1] × [-1, 1]:
 *
 * K(x, y) = exp(-Λy(x + 1)/2)/(1 + exp(-Λy))
 *
 * While LogisticKernel is primarily a fermionic analytic continuation kernel,
 * it can also model the τ dependence of a bosonic correlation function as:
 *
 * ∫ [exp(-Λy(x + 1)/2)/(1 - exp(-Λy))] ρ(y) dy = ∫ K(x, y) ρ'(y) dy
 *
 * where ρ'(y) = w(y)ρ(y) and the weight function w(y) = 1/tanh(Λy/2)
 *
 * @param lambda The cutoff parameter Λ (must be non-negative)
 * @param status Pointer to store the status code
 * @return Pointer to the newly created kernel object, or NULL if creation fails
 *
 * @note The kernel is implemented using piecewise Legendre polynomial expansion
 *       for numerical stability and accuracy.
 */
spir_kernel *spir_logistic_kernel_new(double lambda, int *status);

/**
 * @brief Creates a new regularized bosonic kernel for analytical continuation.
 *
 * In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is
 * a function on [-1, 1] × [-1, 1]:
 *
 * K(x, y) = y * exp(-Λy(x + 1)/2)/(exp(-Λy) - 1)
 *
 * Special care is taken in evaluating this expression around y = 0 to handle
 * the singularity. The kernel is specifically designed for bosonic functions
 * and includes proper regularization to handle numerical stability issues.
 *
 * @param lambda The cutoff parameter Λ (must be non-negative)
 * @param status Pointer to store the status code
 * @return Pointer to the newly created kernel object, or NULL if creation fails
 *
 * @note This kernel is specifically designed for bosonic correlation functions
 *       and should not be used for fermionic cases.
 * @note The kernel is implemented using piecewise Legendre polynomial expansion
 *       for numerical stability and accuracy.
 */
spir_kernel *spir_reg_bose_kernel_new(double lambda, int *status);

/**
 * @brief Retrieves the domain boundaries of a kernel function.
 *
 * This function obtains the domain boundaries (ranges) for both the x and y
 * variables of the specified kernel function. The kernel domain is typically
 * defined as a rectangle in the (x,y) plane.
 *
 * @param k Pointer to the kernel object whose domain is to be retrieved.
 * @param xmin Pointer to store the minimum value of the x-range.
 * @param xmax Pointer to store the maximum value of the x-range.
 * @param ymin Pointer to store the minimum value of the y-range.
 * @param ymax Pointer to store the maximum value of the y-range.
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note For the logistic and regularized bosonic kernels, the domain is
 *       typically [-1, 1] × [-1, 1] in dimensionless variables.
 */
int spir_kernel_domain(const spir_kernel *k, double *xmin, double *xmax,
                       double *ymin, double *ymax);

/**
 * @brief Perform truncated singular value expansion (SVE) of a kernel.
 *
 * Computes a truncated singular value expansion of an integral kernel
 * K: [xmin, xmax] × [ymin, ymax] → ℝ in the form:
 *
 * K(x, y) = ∑ s[l] * u[l](x) * v[l](y)  for l = 1, 2, 3, ...
 *
 * where:
 * - s[l] are singular values in non-increasing order
 * - u[l](x) are left singular functions, forming an orthonormal system on
 * [xmin, xmax]
 * - v[l](y) are right singular functions, forming an orthonormal system on
 * [ymin, ymax]
 *
 * The SVE is computed by mapping it onto a singular value decomposition (SVD)
 * of a matrix using piecewise Legendre polynomial expansion. The accuracy of
 * the computation is controlled by the epsilon parameter, which determines:
 * - The relative magnitude of included singular values
 * - The accuracy of computed singular values and vectors
 *
 * @param k Pointer to the kernel object for which to compute SVE
 * @param epsilon Accuracy target for the basis. Determines:
 *               - The relative magnitude of included singular values
 *               - The accuracy of computed singular values and vectors
 * @param status Pointer to store the status code
 * @return Pointer to the newly created SVE result, or NULL if creation fails
 *
 * @note The computation automatically uses optimized strategies:
 *       - For centrosymmetric kernels, specialized algorithms are employed
 *       - The working precision is adjusted to meet accuracy requirements
 *       - If epsilon is below √ε (where ε is machine epsilon), a warning is
 *         issued and higher precision arithmetic is used
 *
 * @note The returned object must be freed using spir_release_sve_result when no
 * longer needed
 * @see spir_release_sve_result
 */
spir_sve_result *spir_sve_result_new(
    const spir_kernel *k,
    double epsilon,
    double cutoff,
    int lmax,
    int n_gauss,
    int work_dtype,
    int *status
);

/**
 * @brief Gets the number of singular values/vectors in an SVE result.
 *
 * This function returns the number of singular values and corresponding
 * singular vectors contained in the specified SVE result object. This number is
 * needed to allocate arrays of the correct size when retrieving singular values
 * or evaluating singular vectors.
 *
 * @param sve Pointer to the SVE result object
 * @param size Pointer to store the number of singular values/vectors
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 */
int spir_sve_result_get_size(const spir_sve_result *sve, int *size);

/**
 * @brief Gets the singular values from an SVE result.
 *
 * This function retrieves all singular values from the specified SVE result
 * object. The singular values are stored in descending order in the output
 * array.
 *
 * @param sve Pointer to the SVE result object
 * @param svals Pre-allocated array to store the singular values.
 *              Must have size at least equal to the value returned by
 *              spir_sve_result_get_size()
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @see spir_sve_result_get_size
 */
int spir_sve_result_get_svals(const spir_sve_result *sve, double *svals);

/**
 * @brief Gets the number of functions in a functions object.
 *
 * This function returns the number of functions contained in the specified
 * functions object. This number is needed to allocate arrays of the correct
 * size when evaluating the functions.
 *
 * @param funcs Pointer to the functions object
 * @param size Pointer to store the number of functions
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 */
int spir_funcs_get_size(const spir_funcs *funcs, int *size);

/**
 * @brief Creates a new function object containing a subset of functions from
 * the input.
 *
 * This function creates a new function object that contains only the functions
 * specified by the indices array. The indices must be valid (within range and
 * no duplicates).
 *
 * @param funcs Pointer to the source function object
 * @param nslice Number of functions to select (length of indices array)
 * @param indices Array of indices specifying which functions to include in the
 * slice
 * @param status Pointer to store the status code (0 for success, non-zero for
 * error)
 * @return Pointer to the new function object containing the selected functions,
 * or NULL on error
 *
 * @note The caller is responsible for freeing the returned object using
 * spir_funcs_free
 * @note If status is non-zero, the returned pointer will be NULL
 */
spir_funcs *spir_funcs_get_slice(const spir_funcs *funcs, int nslice,
                                 int *indices, int *status);

/**
 * @brief Evaluates functions at a single point in the imaginary-time domain or
 * the real frequency domain.
 *
 * This function evaluates all functions at a specified point x.
 * The values of each basis function at x are stored in the output array.
 * The output array out[j] contains the value of the j-th function evaluated at
 * x.
 *
 * @param funcs Pointer to a functions object
 * @param x Point at which to evaluate the functions
 * @param out Pre-allocated array to store the evaluation results.
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The output array must be pre-allocated with sufficient size to store
 *       all function values
 */
int spir_funcs_eval(const spir_funcs *funcs, double x, double *out);

/**
 * @brief Evaluate a funcs object at a single Matsubara frequency
 *
 * This function evaluates the basis functions at a single Matsubara frequency
 * index. The output array will contain the values of all basis functions at the
 * specified frequency.
 *
 * @param funcs Pointer to the funcs object to evaluate
 * @param x The Matsubara frequency index (integer)
 * @param out Pointer to the output array where the results will be stored.
 *            The array must have enough space to store all basis function
 * values. The values are stored in the order of basis functions.
 * @return int SPIR_COMPUTATION_SUCCESS on success, or an error code on failure
 */
int spir_funcs_eval_matsu(const spir_funcs *funcs, int64_t x, c_complex *out);

/**
 * @brief Evaluate a funcs object at multiple points in the imaginary-time
 * domain or the real frequency domain
 *
 * This function evaluates the basis functions at multiple points. The points
 * can be either in the imaginary-time domain or the real frequency domain,
 * depending on the type of the funcs object (u or v basis functions).
 *
 * The output array can be stored in either row-major or column-major order,
 * specified by the order parameter. In row-major order, the output is stored as
 * (num_points, nfuncs), while in column-major order, it is stored as (nfuncs,
 * num_points).
 *
 * @param funcs Pointer to the funcs object to evaluate
 * @param order Memory layout of the output array:
 *             - SPIR_ORDER_ROW_MAJOR: (num_points, nfuncs)
 *             - SPIR_ORDER_COLUMN_MAJOR: (nfuncs, num_points)
 * @param num_points Number of points to evaluate
 * @param xs Array of points to evaluate at. The points should be in the
 * appropriate domain (imaginary time for u basis, real frequency for v basis)
 * @param out Pointer to the output array where the results will be stored.
 *            The array must have enough space to store num_points * nfuncs
 * values, where nfuncs is the number of basis functions.
 * @return int SPIR_COMPUTATION_SUCCESS on success, or an error code on failure
 */
int spir_funcs_batch_eval(const spir_funcs *funcs, int order, int num_points,
                          const double *xs, double *out);

/**
 * @brief Evaluates basis functions at multiple Matsubara frequencies.
 *
 * This function evaluates all functions contained in a functions object
 * at the specified Matsubara frequency indices. The values of each
 * function at each frequency are stored in the output array.
 *
 * @param funcs Pointer to the functions object
 * @param order Specifies the memory layout of the output array:
 *             SPIR_ORDER_ROW_MAJOR for row-major order (frequency index varies
 * fastest), SPIR_ORDER_COLUMN_MAJOR for column-major order (function index
 * varies fastest)
 * @param num_freqs Number of Matsubara frequencies at which to evaluate
 * @param matsubara_freq_indices Array of Matsubara frequency indices
 * @param out Pre-allocated array to store the evaluation results. The results
 * are stored as a 2D array of size num_freqs x n_funcs.
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The output array must be pre-allocated with sufficient size to store
 *       all function values at all requested frequencies. Indices n correspond
 * to ωn = nπ/β, where n are odd for fermionic frequencies and even for bosonic
 * frequencies.
 */
int spir_funcs_batch_eval_matsu(const spir_funcs *funcs, int order,
                                int num_freqs,
                                const int64_t *matsubara_freq_indices,
                                c_complex *out);

/**
 * @brief Gets the number of roots of a funcs object.
 *
 * This function returns the number of roots of the specified funcs object.
 * This function is only available for continuous functions.
 *
 * @param funcs Pointer to the funcs object
 * @param n_roots Pointer to store the number of roots
 * @return An integer status code:
 */
int spir_funcs_get_n_roots(const spir_funcs *funcs, int *n_roots);

/**
 * @brief Gets the roots of a funcs object.
 *
 * This function returns the roots of the specified funcs object in the
 * non-ascending order. If the size of the funcs object is greater than 1, the
 * roots for all the functions are returned. This function is only available for
 * continuous functions.
 *
 * @param funcs Pointer to the funcs object
 * @param n_roots Pointer to store the number of roots
 * @param roots Pointer to store the roots
 * @return An integer status code:
 */
int spir_funcs_get_roots(const spir_funcs *funcs, double *roots);

/**
 * @brief Creates a new finite temperature IR basis using a
 * pre-computed SVE result.
 *
 * This function creates a intermediate representation (IR) basis
 * using a pre-computed singular value expansion (SVE) result. This allows for
 * reusing an existing SVE computation, which can be more efficient than
 * recomputing it.
 *
 * @param statistics Statistics type (SPIR_STATISTICS_FERMIONIC or
 * SPIR_STATISTICS_BOSONIC)
 * @param beta Inverse temperature β (must be positive)
 * @param omega_max Frequency cutoff ωmax (must be non-negative)
 * @param k Pointer to the kernel object used for the basis construction
 * @param sve Pointer to a pre-computed SVE result for the kernel
 * @param max_size Maximum number of basis functions to include. If -1, all
 * @param status Pointer to store the status code
 * @return Pointer to the newly created basis object, or NULL if creation fails
 *
 * @note Using a pre-computed SVE can significantly improve performance when
 *       creating multiple basis objects with the same kernel
 * @see spir_sve_result_new
 * @see spir_release_finite_temp_basis
 */
spir_basis *spir_basis_new(int statistics, double beta, double omega_max,
                           const spir_kernel *k, const spir_sve_result *sve,
                           int max_size,
                           int *status);

/**
 * @brief Gets the size (number of basis functions) of a finite temperature
 * basis.
 *
 * This function returns the number of basis functions in the specified finite
 * temperature basis object. This size determines the dimensionality of the
 * basis and is needed when allocating arrays for basis function evaluations.
 *
 * @param b Pointer to the finite temperature basis object
 * @param size Pointer to store the number of basis functions
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note For an IR basis, the size is determined automatically during basis
 * construction based on the specified parameters (β, ωmax, ε) and the kernel's
 * singular value expansion.
 * @note For a DLR basis, the size is the number of poles.
 */
int spir_basis_get_size(const spir_basis *b, int *size);

/**
 * @brief Gets the singular values of a finite temperature basis.
 *
 * This function returns the singular values of the specified finite temperature
 * basis object. The singular values are the square roots of the eigenvalues of
 * the covariance matrix of the basis functions.
 *
 * @param sve Pointer to the finite temperature basis object
 * @param svals Pointer to store the singular values
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The singular values are ordered in descending order
 * @note The number of singular values is equal to the basis size
 * @see spir_basis_get_size
 */
int spir_basis_get_svals(const spir_basis *b, double *svals);

/**
 * @brief Gets the statistics type (Fermionic or Bosonic) of a finite
 * temperature basis.
 *
 * This function returns the statistics type of the specified finite temperature
 * basis object. The statistics type determines whether the basis is for
 * fermionic or bosonic Green's functions.
 *
 * @param b Pointer to the finite temperature basis object
 * @param statistics Pointer to store the statistics type
 * (SPIR_STATISTICS_FERMIONIC or SPIR_STATISTICS_BOSONIC)
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The statistics type is determined during basis construction and cannot
 * be changed
 * @note The statistics type affects the form of the basis functions and the
 *       sampling points used for evaluation.
 */
int spir_basis_get_stats(const spir_basis *b, int *statistics);

/**
 * @brief Gets the singular values of a finite temperature basis.
 *
 * This function returns the singular values of the specified finite temperature
 * basis object. The singular values are the square roots of the eigenvalues of
 * the covariance matrix of the basis functions.
 */
int spir_basis_get_singular_values(const spir_basis *b, double *svals);

/**
 * @brief Gets the basis functions of a finite temperature basis.
 *
 * This function returns an object representing the basis functions
 * in the imaginary-time domain of the specified finite temperature basis.
 *
 * @param b Pointer to the finite temperature basis object
 * @param status Pointer to store the status code
 * @return Pointer to the basis functions object, or NULL if creation fails
 *
 * @note The returned object must be freed using spir_release_funcs
 *       when no longer needed
 * @see spir_release_funcs
 */
spir_funcs *spir_basis_get_u(const spir_basis *b, int *status);

/**
 * @brief Gets the basis functions of a finite temperature basis.
 *
 * This function returns an object representing the basis functions
 * in the real-frequency domain of the specified finite temperature basis.
 *
 * @param b Pointer to the finite temperature basis object
 * @param status Pointer to store the status code
 * @return Pointer to the basis functions object, or NULL if creation fails
 *
 * @note The returned object must be freed using spir_release_funcs
 *       when no longer needed
 * @see spir_release_funcs
 */
spir_funcs *spir_basis_get_v(const spir_basis *b, int *status);

/**
 * @brief Gets the basis functions in Matsubara frequency domain.
 *
 * This function returns an object representing the basis functions
 * in the Matsubara-frequency domain of the specified finite temperature basis.
 *
 * @param b Pointer to the finite temperature basis object
 * @param status Pointer to store the status code
 * @return Pointer to the basis functions object, or NULL if creation fails
 *
 * @note The returned object must be freed using spir_release_funcs
 *       when no longer needed
 * @see spir_release_funcs
 */
spir_funcs *spir_basis_get_uhat(const spir_basis *b, int *status);

/**
 * @brief Gets the number of default tau sampling points for an IR basis.
 *
 * This function returns the number of default sampling points in imaginary time
 * (τ) that are automatically chosen for optimal conditioning of the sampling
 * matrix. These points are the extrema of the highest-order basis function in
 * imaginary time.
 *
 * @param b Pointer to a finite temperature basis object (must be an IR basis)
 * @param num_points Pointer to store the number of sampling points
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note This function is only available for IR basis objects
 * @note The default sampling points are chosen to provide near-optimal
 *       conditioning for the given basis size
 * @see spir_basis_get_default_taus
 */
int spir_basis_get_n_default_taus(const spir_basis *b, int *num_points);

/**
 * @brief Gets the default tau sampling points for an IR basis.
 *
 * This function fills the provided array with the default sampling points in
 * imaginary time (τ) that are automatically chosen for optimal conditioning of
 * the sampling matrix. These points are the extrema of the highest-order basis
 * function in imaginary time.
 *
 * @param b Pointer to a finite temperature basis object (must be an IR basis)
 * @param points Pre-allocated array to store the τ sampling points
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note This function is only available for IR basis objects
 * @note The array must be pre-allocated with size >=
 *       spir_basis_get_n_default_taus(b)
 * @note The default sampling points are chosen to provide near-optimal
 *       conditioning for the given basis size
 * @see spir_basis_get_n_default_taus
 */
int spir_basis_get_default_taus(const spir_basis *b, double *points);

/**
 * @brief Gets the number of default omega sampling points for an IR basis.
 *
 * This function returns the number of default sampling points in real frequency
 * (ω) that are automatically chosen for optimal conditioning of the sampling
 * matrix. These points are the extrema of the highest-order basis function in
 * real frequency.
 *
 * @param b Pointer to a finite temperature basis object (must be an IR basis)
 * @param num_points Pointer to store the number of sampling points
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note This function is only available for IR basis objects
 * @note The default sampling points are chosen to provide near-optimal
 *       conditioning for the given basis size
 * @see spir_basis_get_default_ws
 */
int spir_basis_get_n_default_ws(const spir_basis *b, int *num_points);

/**
 * @brief Gets the default omega sampling points for an IR basis.
 *
 * This function fills the provided array with the default sampling points in
 * real frequency (ω) that are automatically chosen for optimal conditioning of
 * the sampling matrix. These points are the extrema of the highest-order basis
 * function in real frequency.
 *
 * @param b Pointer to a finite temperature basis object (must be an IR basis)
 * @param points Pre-allocated array to store the ω sampling points
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note This function is only available for IR basis objects
 * @note The array must be pre-allocated with size >=
 *       spir_basis_get_n_default_ws(b)
 * @note The default sampling points are chosen to provide near-optimal
 *       conditioning for the given basis size
 * @see spir_basis_get_n_default_ws
 */
int spir_basis_get_default_ws(const spir_basis *b, double *points);


/***
 * @brief Gets the default tau sampling points for ann IR basis.
 *
 * This function returns default tau sampling points for an IR basis object.
 *
 * @param b Pointer to the basis object
 * @param n_points Number of requested sampling points.
 * @param points Pre-allocated array to store the sampling points. The size of the array must be at least n_points.
 * @param n_points_returned Number of sampling points returned.
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 */
int spir_basis_get_default_taus_ext(const spir_basis *b, int n_points, double *points, int *n_points_returned);

/**
 * @brief Gets the number of default Matsubara sampling points for an IR basis.
 *
 * This function returns the number of default sampling points in Matsubara
 * frequencies (iωn) that are automatically chosen for optimal conditioning of
 * the sampling matrix. These points are the extrema of the highest-order basis
 * function in Matsubara frequencies.
 *
 * @param b Pointer to a finite temperature basis object (must be an IR basis)
 * @param positive_only If true, only positive frequencies are used
 * @param num_points Pointer to store the number of sampling points
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note This function is only available for IR basis objects
 * @note The default sampling points are chosen to provide near-optimal
 *       conditioning for the given basis size
 * @see spir_basis_get_default_matsus
 */
int spir_basis_get_n_default_matsus(const spir_basis *b, bool positive_only,
                                    int *num_points);

/**
 * @brief Gets the default Matsubara sampling points for an IR basis.
 *
 * This function fills the provided array with the default sampling points in
 * Matsubara frequencies (iωn) that are automatically chosen for optimal
 * conditioning of the sampling matrix. These points are the extrema of the
 * highest-order basis function in Matsubara frequencies.
 *
 * @param b Pointer to a finite temperature basis object (must be an IR basis)
 * @param positive_only If true, only positive frequencies are used
 * @param points Pre-allocated array to store the Matsubara frequency indices
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note This function is only available for IR basis objects
 * @note The array must be pre-allocated with size >=
 *       spir_basis_get_n_default_matsus(b)
 * @note The default sampling points are chosen to provide near-optimal
 *       conditioning for the given basis size
 * @note For fermionic case, the indices n give frequencies ωn = (2n + 1)π/β
 * @note For bosonic case, the indices n give frequencies ωn = 2nπ/β
 * @see spir_basis_get_n_default_matsus
 */
int spir_basis_get_default_matsus(const spir_basis *b, bool positive_only,
                                  int64_t *points);

/**
 * @brief Gets the number of default Matsubara sampling points for an IR basis.
 *
 * This function returns the number of default sampling points in Matsubara
 * frequencies (iωn) that are automatically chosen for optimal conditioning of
 * the sampling matrix. These points are the extrema of the highest-order basis
 * function in Matsubara frequencies.
 *
 * @param b Pointer to a finite temperature basis object (must be an IR basis)
 * @param positive_only If true, only positive frequencies are used
 * @param L Number of requested sampling points.
 * @param num_points_returned Pointer to store the number of sampling points returned.
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note This function is only available for IR basis objects
 * @note The default sampling points are chosen to provide near-optimal
 *       conditioning for the given basis size
 * @see spir_basis_get_default_matsus
 */
int spir_basis_get_n_default_matsus_ext(const spir_basis *b, bool positive_only, int L, int *num_points_returned);

/**
 * @brief Gets the default Matsubara sampling points for an IR basis.
 *
 * This function fills the provided array with the default sampling points in
 * Matsubara frequencies (iωn) that are automatically chosen for optimal
 * conditioning of the sampling matrix. These points are the extrema of the
 * highest-order basis function in Matsubara frequencies.
 *
 * @param b Pointer to a finite temperature basis object (must be an IR basis)
 * @param positive_only If true, only positive frequencies are used
 * @param n_points Number of requested sampling points.
 * @param points Pre-allocated array to store the sampling points. The size of the array must be at least n_points.
 * @param n_points_returned Number of sampling points returned.
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 */
int spir_basis_get_default_matsus_ext(const spir_basis *b, bool positive_only, int n_points, int64_t *points, int *n_points_returned);

/**
 * @brief Creates a new Discrete Lehmann Representation (DLR) basis.
 *
 * This function implements a variant of the discrete Lehmann representation
 * (DLR). Unlike the IR which uses truncated singular value expansion of the
 * analytic continuation kernel K, the DLR is based on a "sketching" of K. The
 * resulting basis is a linear combination of discrete set of poles on the
 * real-frequency axis, continued to the imaginary-frequency axis:
 *
 * G(iν) = ∑ a[i] * reg[i] / (iν - w[i]) for i = 1, 2, ..., L
 *
 * where:
 * - a[i] are the expansion coefficients
 * - w[i] are the poles on the real axis
 * - reg[i] are the regularization factors, which are 1 for fermionic
 * frequencies. For bosonic frequencies, we take reg[i] = tanh(βω[i]/2)
 * (logistic kernel), reg[i] = w[i] (regularized bosonic kernel). The DLR basis
 * functions are given by u[i](iν) = reg[i] / (iν - w[i]) in the
 * imaginary-frequency domain. In the imaginary-time domain, the basis functions
 * are given by u[i](τ) = reg[i] * exp(-w[i]τ) / (1 + exp(-w[i]β)) for fermionic
 * frequencies, u[i](τ) = reg[i] * exp(-w[i]τ) / (1 - exp(-w[i]β)) for bosonic
 * frequencies.
 * - iν are Matsubara frequencies
 *
 * @param b Pointer to a finite temperature basis object
 * @param status Pointer to store the status code
 * @return Pointer to the newly created DLR object, or NULL if creation fails
 */
spir_basis *spir_dlr_new(const spir_basis *b, int *status);

/**
 * @brief Creates a new Discrete Lehmann Representation (DLR) with
 * custom poles.
 *
 * This function creates a DLR basis with user-specified pole locations on the
 * real-frequency axis. This allows for more control over the pole selection
 * compared to the automatic pole selection in spir_dlr_new.
 *
 * @param b Pointer to a finite temperature basis object
 * @param npoles Number of poles to use in the representation
 * @param poles Array of pole locations on the real-frequency axis
 * @param status Pointer to store the status code
 * @return Pointer to the newly created DLR object, or NULL if creation fails
 */
spir_basis *spir_dlr_new_with_poles(const spir_basis *b, const int npoles,
                                    const double *poles, int *status);

/**
 * @brief Gets the number of poles in a DLR.
 *
 * This function returns the number of poles in the specified DLR object.
 *
 * @param dlr Pointer to the DLR object
 * @param num_poles Pointer to store the number of poles
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @see spir_dlr_get_poles
 */
int spir_dlr_get_npoles(const spir_basis *dlr, int *num_poles);

/**
 * @brief Gets the poles in a DLR.
 *
 * This function returns the poles in the specified DLR object.
 *
 * @param dlr Pointer to the DLR object
 * @param poles Pointer to store the poles
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @see spir_dlr_get_npoles
 */
int spir_dlr_get_poles(const spir_basis *dlr, double *poles);

/**
 * @brief Transforms a given input array from the Intermediate Representation
 * (IR) to the Discrete Lehmann Representation (DLR) using the specified DLR
 * object. This version handles real (double precision) input and output arrays.
 *
 * @param dlr Pointer to the DLR basis object
 * @param order Order type (C or Fortran)
 * @param ndim Number of dimensions of input/output arrays
 * @param input_dims Array of dimensions
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input coefficients array in IR
 * @param out Output array in DLR
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The input and output arrays must be allocated with sufficient memory.
 *       The size of the input and output arrays should match the dimensions
 *       specified. The order type determines the memory layout of the input and
 *       output arrays. The function assumes that the input array is in the
 *       specified order type. The output array will be in the specified order
 *       type.
 *
 * @see spir_ir2dlr
 * @see spir_dlr2ir_dd
 */
int spir_ir2dlr_dd(const spir_basis *dlr, int order, int ndim,
                   const int *input_dims, int target_dim, const double *input,
                   double *out);

int spir_ir2dlr_zz(const spir_basis *dlr, int order, int ndim,
                   const int *input_dims, int target_dim,
                   const c_complex *input, c_complex *out);

/**
 * @brief Transforms coefficients from DLR basis to IR representation.
 *
 * This function converts expansion coefficients from the Discrete Lehmann
 * Representation (DLR) basis to the Intermediate Representation (IR) basis.
 * The transformation is performed using the fitting matrix:
 *
 * g_IR = fitmat * g_DLR
 *
 * where:
 * - g_IR are the coefficients in the IR basis
 * - g_DLR are the coefficients in the DLR basis
 * - fitmat is the transformation matrix
 *
 * @param dlr Pointer to the DLR object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of DLR coefficients (double precision)
 * @param out Output array for the IR coefficients (double precision)
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note The transformation is a direct matrix multiplication, which is
 *       typically faster than the inverse transformation
 *
 * @see spir_ir2dlr
 */
int spir_dlr2ir_dd(const spir_basis *dlr, int order, int ndim,
                   const int *input_dims, int target_dim, const double *input,
                   double *out);

/**
 * @brief Transforms coefficients from DLR basis to IR representation.
 * This version handles complex input and output arrays.
 *
 * This function converts expansion coefficients from the Discrete Lehmann
 * Representation (DLR) basis to the Intermediate Representation (IR) basis.
 * The transformation is performed using the fitting matrix:
 *
 * g_IR = fitmat * g_DLR
 *
 * where:
 * - g_IR are the coefficients in the IR basis
 * - g_DLR are the coefficients in the DLR basis
 * - fitmat is the transformation matrix
 *
 * @param dlr Pointer to the DLR object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of DLR coefficients (complex)
 * @param out Output array for the IR coefficients (complex)
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note The transformation is a direct matrix multiplication, which is
 *       typically faster than the inverse transformation
 *
 * @see spir_ir2dlr_zz
 * @see spir_dlr2ir_dd
 */
int spir_dlr2ir_zz(const spir_basis *dlr, int order, int ndim,
                   const int *input_dims, int target_dim,
                   const c_complex *input, c_complex *out);

/**
 * @brief Creates a new tau sampling object for sparse sampling in
 * imaginary time with custom sampling points.
 *
 * Constructs a sampling object that allows transformation between the IR basis
 * and a user-specified set of sampling points in imaginary time (τ). The
 * sampling points are provided by the user, allowing for custom sampling
 * strategies.
 *
 * @param b Pointer to a finite temperature basis object
 * @param num_points Number of sampling points
 * @param points Array of sampling points in imaginary time (τ)
 * @param status Pointer to store the status code
 * @return Pointer to the newly created sampling object, or NULL if creation
 * fails
 *
 * @note The sampling points should be chosen to ensure numerical stability and
 *       accuracy for the given basis
 * @note The sampling matrix is automatically factorized using SVD for efficient
 *       transformations
 * @note The returned object must be freed using spir_release_sampling when no
 *       longer needed
 * @see spir_release_sampling
 */
spir_sampling *spir_tau_sampling_new(const spir_basis *b, int num_points,
                                     const double *points, int *status);

/**
 * @brief Creates a new tau sampling object for sparse sampling in
 * imaginary time with custom sampling points and a pre-computed matrix.
 *
 * This function creates a sampling object that allows transformation between
 * the IR basis and a user-specified set of sampling points in imaginary time
 * (τ). The sampling points are provided by the user, allowing for custom
 * sampling strategies.
 *
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param statistics Statistics type (SPIR_STATISTICS_FERMIONIC or
 * SPIR_STATISTICS_BOSONIC)
 * @param basis_size Basis size
 * @param num_points Number of sampling points
 * @param points Array of sampling points in imaginary time (τ)
 * @param matrix Pre-computed matrix for the sampling points (num_points x
 * basis_size). For Matsubara sampling, this should be a complex matrix.
 * @param status Pointer to store the status code
 * @return Pointer to the newly created sampling object, or NULL if creation
 * fails
 */
spir_sampling *spir_tau_sampling_new_with_matrix(int order, int statistics,
                                                 int basis_size, int num_points,
                                                 const double *points,
                                                 const double *matrix,
                                                 int *status);

/**
 * @brief Creates a new Matsubara sampling object for sparse sampling
 * in Matsubara frequencies with custom sampling points.
 *
 * Constructs a sampling object that allows transformation between the IR basis
 * and a user-specified set of sampling points in Matsubara frequencies (iωn).
 * The sampling points are provided by the user, allowing for custom sampling
 * strategies.
 *
 * @param b Pointer to a finite temperature basis object
 * @param positive_only If true, only positive frequencies are used
 * @param num_points Number of sampling points
 * @param points Array of Matsubara frequency indices (n) for the sampling
 * points
 * @param status Pointer to store the status code
 * @return Pointer to the newly created sampling object, or NULL if creation
 * fails
 */
spir_sampling *spir_matsu_sampling_new(const spir_basis *b, bool positive_only,
                                       int num_points, const int64_t *points,
                                       int *status);

/**
 * @brief Creates a new Matsubara sampling object for sparse sampling in
 * Matsubara frequencies with custom sampling points and a pre-computed
 * evaluation matrix.
 *
 * This function creates a sampling object that can be used to evaluate and fit
 * functions at specific Matsubara frequencies. The sampling points and
 * evaluation matrix are provided directly, allowing for custom sampling
 * configurations.
 *
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param statistics Statistics type (SPIR_STATISTICS_FERMIONIC or
 * SPIR_STATISTICS_BOSONIC)
 * @param basis_size Basis size
 * @param positive_only If true, only positive Matsubara frequencies are used
 * @param num_points Number of sampling points
 * @param points Array of Matsubara frequencies (integer indices)
 * @param matrix Pre-computed evaluation matrix of size (num_points ×
 * basis_size)
 * @param status Pointer to store the status code
 * @return Pointer to the new sampling object, or NULL if creation fails
 *
 * @see spir_matsu_sampling_new
 */
spir_sampling *
spir_matsu_sampling_new_with_matrix(int order, int statistics, int basis_size,
                                    bool positive_only, int num_points,
                                    const int64_t *points,
                                    const c_complex *matrix, int *status);

/**
 * @brief Gets the number of sampling points in a sampling object.
 *
 * This function returns the number of sampling points used in the specified
 * sampling object. This number is needed to allocate arrays of the correct size
 * when retrieving the actual sampling points.
 *
 * @param s Pointer to the sampling object
 * @param num_points Pointer to store the number of sampling points
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @see spir_sampling_get_taus
 * @see spir_sampling_get_matsus
 */
int spir_sampling_get_npoints(const spir_sampling *s, int *num_points);

/**
 * @brief Gets the imaginary time sampling points.
 *
 * This function fills the provided array with the imaginary time (τ) sampling
 * points used in the specified sampling object. The array must be pre-allocated
 * with sufficient size (use spir_sampling_get_npoints to determine the
 * required size).
 *
 * @param s Pointer to the sampling object
 * @param points Pre-allocated array to store the τ sampling points
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The array must be pre-allocated with size >=
 *       spir_sampling_get_npoints(s)
 * @see spir_sampling_get_npoints
 */
int spir_sampling_get_taus(const spir_sampling *s, double *points);

/**
 * @brief Gets the Matsubara frequency sampling points.
 *
 * This function fills the provided array with the Matsubara frequency indices
 * (n) used in the specified sampling object. The actual Matsubara frequencies
 * are ωn = (2n + 1)π/β for fermionic case and ωn = 2nπ/β for bosonic case. The
 * array must be pre-allocated with sufficient size (use
 * spir_sampling_get_npoints to determine the required size).
 *
 * @param s Pointer to the sampling object
 * @param points Pre-allocated array to store the Matsubara frequency indices
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The array must be pre-allocated with size >=
 *       spir_sampling_get_npoints(s)
 * @note For fermionic case, the indices n give frequencies ωn = (2n + 1)π/β
 * @note For bosonic case, the indices n give frequencies ωn = 2nπ/β
 * @see spir_sampling_get_npoints
 */
int spir_sampling_get_matsus(const spir_sampling *s, int64_t *points);

/**
 * @brief Gets the condition number of the sampling matrix.
 *
 * This function returns the condition number of the sampling matrix used in the
 * specified sampling object. The condition number is a measure of how well-
 * conditioned the sampling matrix is.
 *
 * @param s Pointer to the sampling object
 * @param cond_num Pointer to store the condition number
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note A large condition number indicates that the sampling matrix is ill-conditioned,
 *       which may lead to numerical instability in transformations
 * @note The condition number is the ratio of the largest to smallest singular value
 *       of the sampling matrix
 */
int spir_sampling_get_cond_num(const spir_sampling *s, double *cond_num);

/**
 * @brief Evaluates basis coefficients at sampling points (double to double
 * version).
 *
 * Transforms basis coefficients to values at sampling points, where both input
 * and output are real (double precision) values. The operation can be performed
 * along any dimension of a multidimensional array.
 *
 * @param s Pointer to the sampling object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of basis coefficients
 * @param out Output array for the evaluated values at sampling points
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note For optimal performance, the target dimension should be either the
 *       first (0) or the last (ndim-1) dimension to avoid large temporary array
 *       allocations
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note The transformation is performed using a pre-computed sampling matrix
 *       that is factorized using SVD for efficiency
 *
 * @see spir_sampling_eval_dz
 * @see spir_sampling_eval_zz
 */
int spir_sampling_eval_dd(const spir_sampling *s, int order, int ndim,
                          const int *input_dims, int target_dim,
                          const double *input, double *out);

/**
 * @brief Evaluates basis coefficients at sampling points (double to complex
 * version).
 *
 * For more details, see spir_sampling_eval_dd
 * @see spir_sampling_eval_dd
 */
int spir_sampling_eval_dz(const spir_sampling *s, int order, int ndim,
                          const int *input_dims, int target_dim,
                          const double *input, c_complex *out);

/**
 * @brief Evaluates basis coefficients at sampling points (complex to complex
 * version).
 *
 * For more details, see spir_sampling_eval_dd
 * @see spir_sampling_eval_dd
 */
int spir_sampling_eval_zz(const spir_sampling *s, int order, int ndim,
                          const int *input_dims, int target_dim,
                          const c_complex *input, c_complex *out);

/**
 * @brief Fits values at sampling points to basis coefficients (double to double
 * version).
 *
 * Transforms values at sampling points back to basis coefficients, where both
 * input and output are real (double precision) values. The operation can be
 * performed along any dimension of a multidimensional array.
 *
 * @param s Pointer to the sampling object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of values at sampling points
 * @param out Output array for the fitted basis coefficients
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The output array must be pre-allocated with the correct size
 * @note This function performs the inverse operation of
 *       spir_sampling_eval_dd
 * @note The transformation is performed using a pre-computed sampling matrix
 *       that is factorized using SVD for efficiency
 *
 * @see spir_sampling_eval_dd
 * @see spir_sampling_fit_zz
 */
int spir_sampling_fit_dd(const spir_sampling *s, int order, int ndim,
                         const int *input_dims, int target_dim,
                         const double *input, double *out);

/**
 * @brief Fits values at sampling points to basis coefficients (complex to
 * complex version).
 *
 * For more details, see spir_sampling_fit_dd
 * @see spir_sampling_fit_dd
 */
int spir_sampling_fit_zz(const spir_sampling *s, int order, int ndim,
                         const int *input_dims, int target_dim,
                         const c_complex *input, c_complex *out);

#ifdef __cplusplus
}
#endif
