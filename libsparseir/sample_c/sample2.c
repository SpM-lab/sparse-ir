#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <sparseir/sparseir.h>

int main() {

typedef double _Complex c_complex;

int32_t status;

// Create a bosonic finite temperature basis
double beta = 10.0;        // Inverse temperature
double omega_max = 10.0;   // Ultraviolet cutoff
double epsilon = 1e-8;     // Accuracy target

// Create a logistic kernel
spir_kernel* logistic_kernel = spir_logistic_kernel_new(beta * omega_max, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(logistic_kernel != NULL);

// Create a pre-computed SVE result
double cutoff = -1.0;
int lmax = -1;
int n_gauss = -1;
int Twork = SPIR_TWORK_FLOAT64X2;
spir_sve_result* sve_logistic = spir_sve_result_new(logistic_kernel, epsilon, cutoff, lmax, n_gauss, Twork, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(sve_logistic != NULL);

// Create fermionic and bosonic finite temperature bases with pre-computed SVE result
int max_size = -1;
spir_basis* basis_fermionic = spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, logistic_kernel, sve_logistic, max_size, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(basis_fermionic != NULL);

spir_basis* basis_bosonic = spir_basis_new(SPIR_STATISTICS_BOSONIC, beta, omega_max, logistic_kernel, sve_logistic, max_size, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(basis_bosonic != NULL);

// Create a regularized bosonic basis
spir_kernel* regularized_kernel = spir_reg_bose_kernel_new(beta * omega_max, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(regularized_kernel != NULL);

spir_sve_result* sve_regularized = spir_sve_result_new(regularized_kernel, epsilon, cutoff, lmax, n_gauss, Twork, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(sve_regularized != NULL);

spir_basis* basis_regularized = spir_basis_new(SPIR_STATISTICS_BOSONIC, beta, omega_max, regularized_kernel, sve_regularized, max_size, &status);
assert(status == SPIR_COMPUTATION_SUCCESS);
assert(basis_regularized != NULL);

// Clean up (in arbitrary order)
spir_kernel_release(logistic_kernel);
spir_sve_result_release(sve_logistic);
spir_kernel_release(regularized_kernel);
spir_sve_result_release(sve_regularized);
spir_basis_release(basis_fermionic);
spir_basis_release(basis_bosonic);
spir_basis_release(basis_regularized);

    return 0;
}
