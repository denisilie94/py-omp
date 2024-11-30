/**************************************************************************
 *
 * File name: ompcore.h
 *
 * Ron Rubinstein
 * Computer Science Department
 * Technion, Haifa 32000 Israel
 * ronrubin@cs
 *
 * Last Updated: 18.8.2009
 *
 * Contains the core implementation of Batch-OMP / OMP-Cholesky.
 *
 *************************************************************************/


#ifndef __OMP_CORE_H__
#define __OMP_CORE_H__


#include <stdlib.h>
#include <stdio.h>


/* Define data structures */
typedef struct {
    int gamma_mode;    /* Same as input gamma_mode */
    size_t m;          /* Number of rows */
    size_t L;          /* Number of columns */
    double *Gamma_full;      /* If gamma_mode == FULL_GAMMA, this is m x L array */
    size_t nzmax;            /* For sparse matrix */
    double *gammaPr;         /* Non-zero elements */
    size_t *gammaIr;         /* Row indices */
    size_t *gammaJc;         /* Column pointers */
} GammaMatrix;


/**************************************************************************
 * Perform Batch-OMP or OMP-Cholesky on a specified set of signals, using
 * either a fixed number of atoms or an error bound.
 *
 * Parameters (not all required):
 *
 *   D - the dictionary, of size n X m
 *   x - the signals, of size n X L
 *   DtX - D'*x, of size m X L
 *   XtX - squared norms of the signals in x, sum(x.*x), of length L
 *   G - D'*D, of size m X m
 *   T - target sparsity, or maximal number of atoms for error-based OMP
 *   eps - target residual norm for error-based OMP
 *   gamma_mode - one of the constants FULL_GAMMA or SPARSE_GAMMA
 *   profile - if non-zero, profiling info is printed
 *   msg_delta - positive: the # of seconds between status prints, otherwise: nothing is printed
 *   erroromp - if nonzero indicates error-based OMP, otherwise fixed sparsity OMP
 *
 * Usage:
 *
 *   The function can be called using different parameters, and will have
 *   different complexity depending on the parameters specified. Arrays which
 *   are not specified should be passed as null (0). When G is specified, 
 *   Batch-OMP is performed. Otherwise, OMP-Cholesky is performed.
 *
 *   Fixed-sparsity usage:
 *   ---------------------
 *   Either DtX, or D and x, must be specified. Specifying DtX is more efficient.
 *   XtX does not need to be specified.
 *   When D and x are specified, G is not required. However, not providing G
 *   will significantly degrade efficiency.
 *   The number of atoms must be specified in T. The value of eps is ignored.
 *   Finally, set erroromp to 0.
 *
 *   Error-OMP usage:
 *   ----------------
 *   Either DtX and Xtx, or D and x, must be specified. Specifying DtX and XtX
 *   is more efficient.
 *   When D and x are specified, G is not required. However, not providing G
 *   will significantly degrade efficiency.
 *   The target error must be specified in eps. A hard limit on the number
 *   of atoms can also be specified via the parameter T. Otherwise, T should 
 *   be negative. Finally, set erroromp to nonzero.
 *
 *
 * Returns: 
 *   An mxArray containing the sparse representations of the signals in x
 *   (allocated using the appropriate mxCreateXXX() function).
 *   The array is either full or sparse, depending on gamma_mode.
 *
 **************************************************************************/
void ompcore(double D[], double x[], double DtX[], double XtX[], double G[], size_t n, size_t m, size_t L,
             int T, double eps, int gamma_mode, int profile, double msg_delta, int erroromp,
             GammaMatrix *Gamma);


/**
 * @brief Frees a 2D matrix previously allocated by `allocate_matrix`.
 *
 * @param matrix The matrix to free.
 */
void free_matrix(double** matrix);


/* Helper function to print a matrix */
void print_matrix(const char* name, double* mat, size_t rows, size_t cols);


#endif
