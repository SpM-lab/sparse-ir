#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <sparseir/sparseir.h>

int main() {

typedef double _Complex c_complex;

int32_t status;

// Create a fermionic finite temperature basis
double beta = 10.0;        // Inverse temperature
double omega_max = 10.0;   // Ultraviolet cutoff
double epsilon = 1e-8;     // Accuracy target

// Create a logistic kernel
spir_kernel* kernel = spir_logistic_kernel_new(beta * omega_max, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(kernel != NULL);

// Create a pre-computed SVE result
double cutoff = -1.0;
int lmax = -1;
int n_gauss = -1;
int Twork = SPIR_TWORK_FLOAT64X2;
spir_sve_result* sve = spir_sve_result_new(kernel, epsilon, cutoff, lmax, n_gauss, Twork, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(sve != NULL);

// Create a fermionic finite temperature basis with pre-computed SVE result
// Use SPIR_STATISTICS_BOSONIC for bosonic basis
int max_size = -1;
spir_basis* basis = spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, kernel, sve, max_size, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(basis != NULL);

int32_t n_basis;
status = spir_basis_get_size(basis, &n_basis);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Evaluate the basis functions at a given tau point
spir_funcs* u = spir_basis_get_u(basis, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(u != NULL);

double tau = 0.5 * beta;
double* uval = (double*)malloc(n_basis * sizeof(double));
status = spir_funcs_eval(u, tau, uval);
assert(status == SPIR_COMPUTATION_SUCCESS);
for (int i = 0; i < n_basis; ++i) {
    printf("u[%d] = %f\n", i, uval[i]);
}

// Evaluate the basis functions at a given omega point
spir_funcs* v = spir_basis_get_v(basis, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(v != NULL);

double omega = 0.5 * omega_max;
double* vval = (double*)malloc(n_basis * sizeof(double));
status = spir_funcs_eval(v, omega, vval);
assert(status == SPIR_COMPUTATION_SUCCESS);
for (int i = 0; i < n_basis; ++i) {
    printf("v[%d] = %f\n", i, vval[i]);
}

// Evaluate the basis functions at given Matsubara frequencies
int n_freqs = 10;
int64_t* matsubara_freq_indices = (int64_t*)malloc(n_freqs * sizeof(int64_t));
for (int i = 0; i < n_freqs; ++i) {
    matsubara_freq_indices[i] = 2 * i + 1; // fermionic Matsubara frequency
}

c_complex* uhat_val = (c_complex*)malloc(n_basis * n_freqs * sizeof(c_complex));
spir_funcs* uhat = spir_basis_get_uhat(basis, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(uhat != NULL);

status = spir_funcs_batch_eval_matsu(uhat, SPIR_ORDER_COLUMN_MAJOR, n_freqs, matsubara_freq_indices, uhat_val);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Clean up (in arbitrary order)
free(uval);
free(vval);
free(matsubara_freq_indices);
free(uhat_val);
spir_funcs_release(u);
spir_funcs_release(v);
spir_funcs_release(uhat);
spir_basis_release(basis);
spir_sve_result_release(sve);
spir_kernel_release(kernel);

    return 0;
}
