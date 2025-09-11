#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <sparseir/sparseir.h>

int main() {

typedef double _Complex c_complex;

// Create a fermionic finite temperature basis
double beta = 10.0;        // Inverse temperature
double omega_max = 10.0;   // Ultraviolet cutoff
double epsilon = 1e-8;     // Accuracy target

int32_t status;

spir_kernel* kernel = spir_logistic_kernel_new(beta * omega_max, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(kernel != NULL);

// Create a pre-computed SVE result
double cutoff = -1.0;
int lmax = -1;
int n_gauss = -1;
int Twork = SPIR_TWORK_FLOAT64X2;
spir_sve_result* sve_logistic = spir_sve_result_new(kernel, epsilon, cutoff, lmax, n_gauss, Twork, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(sve_logistic != NULL);

// Create fermionic and bosonic finite temperature bases with pre-computed SVE result
int max_size = -1;
spir_basis* basis = spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, kernel, sve_logistic, max_size, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(basis != NULL);

// Get imaginary-time sampling points
int32_t n_tau;
status = spir_basis_get_n_default_taus(basis, &n_tau);
assert(status == SPIR_COMPUTATION_SUCCESS);

double* tau_points = (double*)malloc(n_tau * sizeof(double));
status = spir_basis_get_default_taus(basis, tau_points);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Create sampling object for imaginary-time domain
spir_sampling* tau_sampling = spir_tau_sampling_new(basis, n_tau, tau_points, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(tau_sampling != NULL);

// Get Matsubara frequency indices
bool positive_only = false;
int32_t n_matsubara;
status = spir_basis_get_n_default_matsus(basis, positive_only, &n_matsubara);
assert(status == SPIR_COMPUTATION_SUCCESS);
int64_t* matsubara_indices = (int64_t*)malloc(n_matsubara * sizeof(int64_t));
status = spir_basis_get_default_matsus(basis, positive_only, matsubara_indices);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Create sampling object for Matsubara domain
spir_sampling* matsubara_sampling = spir_matsu_sampling_new(basis, positive_only, n_matsubara, matsubara_indices, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(matsubara_sampling != NULL);

// Create Green's function with a pole at 0.5*omega_max
status = spir_sampling_get_npoints(matsubara_sampling, &n_matsubara);
assert(status == SPIR_COMPUTATION_SUCCESS);

c_complex* g_matsubara = (c_complex*)malloc(n_matsubara * sizeof(c_complex));


// Set pole position
const double pole_position = 0.0 * omega_max;

// Initialize Green's function in Matsubara frequencies
// G(iω_n) = 1/(iω_n - ε)
for (int i = 0; i < n_matsubara; ++i) {
    assert(llabs(matsubara_indices[i]) % 2 == 1); // fermionic Matsubara frequency
    g_matsubara[i] = 1.0 / (I * matsubara_indices[i] * M_PI / beta - pole_position);
}

int32_t target_dim = 0; // target dimension for evaluation and fit

// Matsubara sampling points to basis coefficients
int32_t n_basis;
status = spir_basis_get_size(basis, &n_basis);
assert(status == SPIR_COMPUTATION_SUCCESS);

c_complex* g_fit = (c_complex*)malloc(n_basis * sizeof(c_complex));
int32_t dims[1] = {n_matsubara};
status = spir_sampling_fit_zz(matsubara_sampling, SPIR_ORDER_COLUMN_MAJOR,
                             1, dims, target_dim, g_matsubara, g_fit);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Basis coefficients to imaginary-time sampling points
c_complex* g_tau = (c_complex*)malloc(n_tau * sizeof(c_complex));
dims[0] = n_basis;
status = spir_sampling_eval_zz(tau_sampling, SPIR_ORDER_COLUMN_MAJOR,
                                  1, dims, target_dim, g_fit, g_tau);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Compare with expected result:
//   G(tau) = -exp(-tau * pole_position) / (1 + exp(-beta * pole_position))
for (int i = 0; i < n_tau; ++i) {
    double tau = tau_points[i];
    double expected;
    if (tau >= 0.0) {
        expected = -exp(-tau * pole_position) / (1.0 + exp(-beta * pole_position));
    } else {
        expected = +exp(-(tau + beta) * pole_position) / (1.0 + exp(-beta * pole_position));
    }
    assert(fabs(creal(g_tau[i]) - expected) < epsilon);
    assert(fabs(cimag(g_tau[i])) < epsilon);
}

// Imaginary-time sampling points to basis coefficients
c_complex* g_fit2 = (c_complex*)malloc(n_basis * sizeof(c_complex));
dims[0] = n_tau;
status = spir_sampling_fit_zz(tau_sampling, SPIR_ORDER_COLUMN_MAJOR,
                              1, dims, target_dim, g_tau, g_fit2);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Basis coefficients to Matsubara Green's function
c_complex* g_matsubara_reconstructed = (c_complex*)malloc(n_matsubara * sizeof(c_complex));
dims[0] = n_basis;
status = spir_sampling_eval_zz(matsubara_sampling, SPIR_ORDER_COLUMN_MAJOR,
                                  1, dims, target_dim, g_fit2, g_matsubara_reconstructed);
assert(status == SPIR_COMPUTATION_SUCCESS);

for (int i = 0; i < n_matsubara; ++i) {
    assert(fabs(creal(g_matsubara_reconstructed[i]) - creal(g_matsubara[i])) < epsilon);
    assert(fabs(cimag(g_matsubara_reconstructed[i]) - cimag(g_matsubara[i])) < epsilon);
}

// Clean up (order is arbitrary)
free(matsubara_indices);
free(g_matsubara);
free(g_fit);
free(g_fit2);
free(g_tau);
free(tau_points);
free(g_matsubara_reconstructed);
spir_basis_release(basis);
spir_sampling_release(tau_sampling);
spir_sampling_release(matsubara_sampling);

    return 0;
}
