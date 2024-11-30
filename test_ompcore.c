#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "ompcore.h"


int main() {
    /* Define dimensions */
    size_t n = 5;   // Dimension of the signal space
    size_t m = 8;   // Number of atoms in the dictionary
    size_t L = 3;   // Number of signals

    /* Define parameters */
    int T = 3;            // Sparsity level
    double eps = 1e-6;    // Error tolerance (not used if erroromp == 0)
    int gamma_mode = 0;   // FULL_GAMMA
    int profile = 0;      // Disable profiling
    double msg_delta = 0.0; // Disable status messages
    int erroromp = 0;     // Disable error-based OMP

    /* Allocate and initialize input matrices and vectors */
    double *D = (double *)malloc(n * m * sizeof(double));    // Dictionary matrix
    double *x = (double *)malloc(n * L * sizeof(double));    // Signals matrix
    double *DtX = NULL;   // Not precomputed
    double *XtX = NULL;   // Not precomputed
    double *G = NULL;     // Not provided (standard OMP)

    /* Initialize D with random values */
    srand((unsigned int)time(NULL));  // Seed the random number generator
    for (size_t i = 0; i < n * m; ++i) {
        D[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // Random values in [-1, 1]
    }

    /* Normalize the columns of D */
    for (size_t j = 0; j < m; ++j) {
        double norm = 0.0;
        for (size_t i = 0; i < n; ++i) {
            norm += D[j * n + i] * D[j * n + i];
        }
        norm = sqrt(norm);
        for (size_t i = 0; i < n; ++i) {
            D[j * n + i] /= norm;
        }
    }

    /* Initialize x with random signals */
    for (size_t i = 0; i < n * L; ++i) {
        x[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // Random values in [-1, 1]
    }

    /* Print the dictionary and signals */
    print_matrix("Dictionary (D)", D, n, m);
    print_matrix("Signals (x)", x, n, L);

    /* Prepare the output GammaMatrix structure */
    GammaMatrix Gamma;
    Gamma.gamma_mode = gamma_mode;
    Gamma.m = m;
    Gamma.L = L;
    Gamma.Gamma_full = NULL;
    Gamma.nzmax = 0;
    Gamma.gammaPr = NULL;
    Gamma.gammaIr = NULL;
    Gamma.gammaJc = NULL;

    /* Call the ompcore function */
    ompcore(D, x, DtX, XtX, G, n, m, L, T, eps, gamma_mode, profile, msg_delta, erroromp, &Gamma);

    /* Process and print the output */
    if (Gamma.gamma_mode == 0 && Gamma.Gamma_full != NULL) {
        /* FULL_GAMMA mode: Print the full gamma matrix */
        print_matrix("Gamma (Full Matrix)", Gamma.Gamma_full, m, L);
    } else if (Gamma.gamma_mode != 0) {
        /* SPARSE_GAMMA mode: Print the sparse gamma representation */
        printf("Gamma (Sparse Matrix):\n");
        for (size_t l = 0; l < L; ++l) {
            size_t start = Gamma.gammaJc[l];
            size_t end = Gamma.gammaJc[l + 1];
            printf("Signal %zu coefficients:\n", l);
            for (size_t idx = start; idx < end; ++idx) {
                size_t row = Gamma.gammaIr[idx];
                double val = Gamma.gammaPr[idx];
                printf("Index %zu: %f\n", row, val);
            }
        }
    } else {
        printf("No coefficients computed.\n");
    }

    /* Free allocated memory */
    free(D);
    free(x);
    if (Gamma.gamma_mode == 0 && Gamma.Gamma_full != NULL) {
        free(Gamma.Gamma_full);
    } else {
        if (Gamma.gammaPr != NULL) free(Gamma.gammaPr);
        if (Gamma.gammaIr != NULL) free(Gamma.gammaIr);
        if (Gamma.gammaJc != NULL) free(Gamma.gammaJc);
    }

    return 0;
}
