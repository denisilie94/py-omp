/**************************************************************************
 *
 * File name: ompcore.c
 *
 * Ron Rubinstein
 * Computer Science Department
 * Technion, Haifa 32000 Israel
 * ronrubin@cs
 *
 * Last Updated: 25.8.2009
 *
 *************************************************************************/


#include "ompcore.h"
#include "omputils.h"
#include "ompprof.h"
#include "myblas.h"
#include <math.h>
#include <string.h>


/* Helper function to print a matrix */
void print_matrix(const char* name, double* mat, size_t rows, size_t cols) {
    printf("%s:\n", name);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            printf("%f ", mat[j * rows + i]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrix_to_file(const char* name, double* mat, size_t rows, size_t cols) {
    // Open the file in append mode to preserve existing content
    FILE *file = fopen("log.txt", "a");
    if (!file) {
        fprintf(stderr, "Error: Could not open log.txt for writing.\n");
        return;
    }

    // Print matrix name
    fprintf(file, "%s:\n", name);

    // Print matrix elements
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            fprintf(file, "%f ", mat[j * rows + i]);
        }
        fprintf(file, "\n");
    }

    // Add a newline after the matrix for better readability
    fprintf(file, "\n");

    // Close the file
    fclose(file);
}

/**
 * Free a 2D matrix allocated by allocate_matrix.
 */
void free_matrix(double** matrix) {
    if (matrix) {
        free(matrix[0]);  // Free the single block of data
        free(matrix);     // Free the row pointers
    }
}


void ompcore(double D[], double x[], double DtX[], double XtX[], double G[], size_t n, size_t m, size_t L,
             int T, double eps, int gamma_mode, int profile, double msg_delta, int erroromp,
             GammaMatrix *Gamma)
{
    /* Variable declarations */
    size_t i, j, signum, pos, gamma_count;
    size_t *ind;
    int *selected_atoms;
    size_t allocated_coefs, allocated_cols;
    double *gammaPr;
    size_t *gammaIr;
    size_t *gammaJc;
    int DtX_specified, XtX_specified, batchomp, standardomp;
    double *alpha, *r, *Lchol, *c, *Gsub, *Dsub, sum, *gamma_full, *tempvec1, *tempvec2;
    double eps2, resnorm, delta, deltaprev, secs_remain;
    int mins_remain, hrs_remain;
    clock_t lastprint_time, starttime;

    /* Print the dictionary and signals */
    // print_matrix_to_file("Dictionary (D)", D, n, m);
    // print_matrix_to_file("Signals (x)", x, n, L);

    /* Check flags */
    DtX_specified = (DtX != NULL);
    XtX_specified = (XtX != NULL);
    standardomp = (G == NULL);
    batchomp = !standardomp;

    /* Allocate output matrix */
    if (gamma_mode == 0) {
        printf("full");
        /* Allocate full matrix of size m X L */
        gamma_full = (double *)malloc(m * L * sizeof(double));
        memset(gamma_full, 0, m * L * sizeof(double));
        gammaPr = NULL;
        gammaIr = NULL;
        gammaJc = NULL;
    } else {
        /* Allocate sparse matrix with room for allocated_coefs nonzeros */
        allocated_coefs = erroromp ? (size_t)(ceil(L * sqrt((double)n) / 2.0) + 1.01) : L * T;
        gammaPr = (double *)malloc(allocated_coefs * sizeof(double));
        gammaIr = (size_t *)malloc(allocated_coefs * sizeof(size_t));
        gammaJc = (size_t *)malloc((L + 1) * sizeof(size_t));
        gamma_count = 0;
        gammaJc[0] = 0;
        gamma_full = NULL;
    }

    /* Assign to Gamma */
    Gamma->gamma_mode = gamma_mode;
    Gamma->m = m;
    Gamma->L = L;
    Gamma->Gamma_full = gamma_full;
    Gamma->gammaPr = gammaPr;
    Gamma->gammaIr = gammaIr;
    Gamma->gammaJc = gammaJc;
    Gamma->nzmax = allocated_coefs;

    /* Helper arrays */
    alpha = (double *)malloc(m * sizeof(double));
    ind = (size_t *)malloc(n * sizeof(size_t));
    selected_atoms = (int *)malloc(m * sizeof(int));
    c = (double *)malloc(n * sizeof(double));

    /* Current number of columns in Dsub / Gsub / Lchol */
    allocated_cols = erroromp ? (size_t)(ceil(sqrt((double)n) / 2.0) + 1.01) : T;

    /* Cholesky decomposition of D_I'*D_I */
    Lchol = (double *)malloc(n * allocated_cols * sizeof(double));

    /* Temporary vectors */
    tempvec1 = (double *)malloc(m * sizeof(double));
    tempvec2 = (double *)malloc(m * sizeof(double));

    if (batchomp) {
        /* Matrix containing G(:,ind) */
        Gsub = (double *)malloc(m * allocated_cols * sizeof(double));
        Dsub = NULL;
        r = NULL;
    } else {
        /* Matrix containing D(:,ind) */
        Dsub = (double *)malloc(n * allocated_cols * sizeof(double));
        r = (double *)malloc(n * sizeof(double));
        Gsub = NULL;
    }

    if (!DtX_specified) {
        /* Contains D'*x for the current signal */
        DtX = (double *)malloc(m * sizeof(double));
    }

    /* Initializations for error omp */
    if (erroromp) {
        eps2 = eps * eps;
        if (T < 0 || T > n) {
            T = n;
        }
    }

    /* Initialize timers */
    starttime = clock();
    lastprint_time = starttime;

    /* Perform omp for each signal */
    for (signum = 0; signum < L; ++signum) {
        /* Initialize residual norm and deltaprev for error-omp */
        if (erroromp) {
            if (XtX_specified) {
                resnorm = XtX[signum];
            } else {
                resnorm = dotprod(x + n * signum, x + n * signum, n);
            }
            deltaprev = 0;
        } else {
            /* Ignore residual norm stopping criterion */
            eps2 = 0;
            resnorm = 1;
        }

        if (resnorm > eps2 && T > 0) {
            /* Compute DtX */
            if (!DtX_specified) {
                matT_vec(1.0, D, x + n * signum, DtX, n, m);
            }

            /* Initialize alpha := DtX */
            memcpy(alpha, DtX + m * signum * DtX_specified, m * sizeof(double));

            /* Mark all atoms as unselected */
            for (i = 0; i < m; ++i) {
                selected_atoms[i] = 0;
            }
        }

        /* Main loop */
        i = 0;
        while (resnorm > eps2 && i < T) {
            /* Index of next atom */
            pos = maxabs(alpha, m);

            /* Stop criterion: selected same atom twice, or inner product too small */
            if (selected_atoms[pos] || alpha[pos] * alpha[pos] < 1e-14) {
                break;
            }

            /* Mark selected atom */
            ind[i] = pos;
            selected_atoms[pos] = 1;

            /* Matrix reallocation */
            if (erroromp && i >= allocated_cols) {
                allocated_cols = (size_t)(ceil(allocated_cols * MAT_INC_FACTOR) + 1.01);
                Lchol = (double *)realloc(Lchol, n * allocated_cols * sizeof(double));
                if (batchomp) {
                    Gsub = (double *)realloc(Gsub, m * allocated_cols * sizeof(double));
                } else {
                    Dsub = (double *)realloc(Dsub, n * allocated_cols * sizeof(double));
                }
            }

            /* Append column to Gsub or Dsub */
            if (batchomp) {
                for (j = 0; j < m; ++j) {
                    Gsub[i * m + j] = G[pos * m + j];
                }
            } else {
                for (j = 0; j < n; ++j) {
                    Dsub[i * n + j] = D[pos * n + j];
                }
            }

            /* Cholesky update */
            if (i == 0) {
                Lchol[0] = 1;
            } else {
                /* Incremental Cholesky decomposition */
                if (standardomp) {
                    matT_vec(1.0, Dsub, D + n * pos, tempvec1, n, i);
                } else {
                    for (j = 0; j < i; ++j) {
                        tempvec1[j] = Gsub[i * m + ind[j]];
                    }
                }
                backsubst('L', Lchol, tempvec1, tempvec2, n, i);
                for (j = 0; j < i; ++j) {
                    Lchol[j * n + i] = tempvec2[j];
                }

                /* Compute Lchol(i,i) */
                sum = 0;
                for (j = 0; j < i; ++j) {
                    sum += tempvec2[j] * tempvec2[j];
                }
                if ((1 - sum) <= 1e-14) {
                    break;
                }
                Lchol[i * n + i] = sqrt(1 - sum);
            }

            i++;

            /* Perform orthogonal projection and compute sparse coefficients */
            for (j = 0; j < i; ++j) {
                tempvec1[j] = DtX[m * signum * DtX_specified + ind[j]];
            }
            cholsolve('L', Lchol, tempvec1, c, n, i);

            /* Update alpha = D'*residual */
            if (standardomp) {
                mat_vec(-1.0, Dsub, c, r, n, i);
                vec_sum(1.0, x + n * signum, r, n);
                matT_vec(1.0, D, r, alpha, n, m);
                /* Update residual norm */
                if (erroromp) {
                    resnorm = dotprod(r, r, n);
                }
            } else {
                mat_vec(1.0, Gsub, c, tempvec1, m, i);
                memcpy(alpha, DtX + m * signum * DtX_specified, m * sizeof(double));
                vec_sum(-1.0, tempvec1, alpha, m);
                /* Update residual norm */
                if (erroromp) {
                    for (j = 0; j < i; ++j) {
                        tempvec2[j] = tempvec1[ind[j]];
                    }
                    delta = dotprod(c, tempvec2, i);
                    resnorm = resnorm - delta + deltaprev;
                    deltaprev = delta;
                }
            }
        }

        /* Generate output vector gamma */
        if (gamma_mode == 0) {
            /* Write the coefs in c to their correct positions in gamma */
            for (j = 0; j < i; ++j) {
                gamma_full[m * signum + ind[j]] = c[j];
            }
        } else {
            /* Sort the coefs by index before writing them to gamma */
            quicksort(ind, c, i);

            /* Gamma is full - reallocate */
            if (gamma_count + i >= Gamma->nzmax) {
                while (gamma_count + i >= Gamma->nzmax) {
                    Gamma->nzmax = (size_t)(ceil(GAMMA_INC_FACTOR * Gamma->nzmax) + 1.01);
                }
                gammaPr = (double *)realloc(gammaPr, Gamma->nzmax * sizeof(double));
                gammaIr = (size_t *)realloc(gammaIr, Gamma->nzmax * sizeof(size_t));
                Gamma->gammaPr = gammaPr;
                Gamma->gammaIr = gammaIr;
            }

            /* Append coefs to gamma and update the indices */
            for (j = 0; j < i; ++j) {
                gammaPr[gamma_count] = c[j];
                gammaIr[gamma_count] = ind[j];
                gamma_count++;
            }
            gammaJc[signum + 1] = gammaJc[signum] + i;
        }

        /* Display status messages */
        if (msg_delta > 0 && (clock() - lastprint_time) / (double)CLOCKS_PER_SEC >= msg_delta) {
            lastprint_time = clock();
            double time_elapsed = (lastprint_time - starttime) / (double)CLOCKS_PER_SEC;
            double time_per_signal = time_elapsed / (signum + 1);
            double time_remaining = (L - signum - 1) * time_per_signal;
            hrs_remain = (int)(time_remaining / 3600);
            mins_remain = (int)((time_remaining - hrs_remain * 3600) / 60);
            secs_remain = time_remaining - hrs_remain * 3600 - mins_remain * 60;
            printf("omp: signal %zu / %zu, estimated remaining time: %02d:%02d:%05.2f\n",
                   signum + 1, L, hrs_remain, mins_remain, secs_remain);
        }
    }

    /* Print final messages */
    if (msg_delta > 0) {
        printf("omp: signal %zu / %zu\n", signum, L);
    }

    /* Free memory */
    if (!DtX_specified) {
        free(DtX);
    }
    if (standardomp) {
        free(r);
        free(Dsub);
    } else {
        free(Gsub);
    }
    free(tempvec2);
    free(tempvec1);
    free(Lchol);
    free(c);
    free(selected_atoms);
    free(ind);
    free(alpha);
}