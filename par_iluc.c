#include "sparse_matrix.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define zero 0
#define DEBUG 1
#define MANUAL 1
#define THD 0
#define DEBUG_LEVEL 2

#define debug_print(T_ID, ...)                                                 \
  do {                                                                         \
    if (DEBUG && (DEBUG_LEVEL > 1))                                            \
      if ((T_ID) == THD)                                                       \
        fprintf(stderr, __VA_ARGS__);                                          \
      else                                                                     \
        ;                                                                      \
  } while (0)

#if ((DEBUG > 0) && (DEBUG_LEVEL > 0))
#define d_fprintf(fmt, args...)                                                \
  fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__,     \
          ##args)
#else
#define d_fprintf(fmt, args...)
#endif

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

struct par_matrix {
  int max_nnz_row;
  int *ms_i;
  int *ms_j;
  int *act_rows;
  int *ms_rows_freq;
  int *ms_rhs_idx;
  int *f_act_rows;
  int *level_idx;
  double *ms_data;
  double *ms_vdata;
};
void print_csr_mat(const SparseMatrixCSR *csr_mat);
void densify(const SparseMatrixCSR *csr_mat, float *dens_mat);

int main(int argc, char *argv[]) {

  size_t i, ii, jj, j, m;
  size_t t_id, size, rest, ne, ns, inc;
  SparseMatrixCSR csr_mat;

#if MANUAL
  const int n = 8 * 2;
  const int nnz = 23 * 4;
  csr_mat.nnz = nnz;
  csr_mat.nrows = n;

  csr_mat.row_pointers = (int[n + 1]){0,  6,  14, 18, 24, 28, 36, 40, 46,
                                      52, 60, 64, 70, 74, 82, 86, 92};

  csr_mat.column_indices =
      (int[nnz]){0, 2, 4, 5, 10, 12, 1, 2, 3, 6, 9, 10, 11, 14, 0, 2, 8, 10, 1,
                 3, 4, 9, 11, 15, 3, 4, 11, 12, 1, 2, 5, 6, 9, 10, 11, 14, 0, 1,
                 13, 14, 4, 5, 7, 9, 13, 15,
                 /**LOWER*/
                 0, 2, 4, 8, 10, 12, 1, 2, 3, 7, 9, 10, 11, 14, 0, 2, 8, 11, 1,
                 3, 4, 9, 10, 12, 3, 4, 11, 12, 1, 2, 5, 6, 9, 10, 11, 14, 5, 6,
                 13, 14, 4, 5, 7, 11, 12, 15};
  csr_mat.values = (double *)malloc(sizeof(double) * nnz);

  for (i = 0; i < n; i++) {
    for (jj = csr_mat.row_pointers[i]; jj < csr_mat.row_pointers[i + 1]; jj++) {
      ii = csr_mat.column_indices[jj];
      csr_mat.values[jj] = 100 * i + ii;
    }
  }

#if DEBUG
  for (i = 0; i < n; i++) {
    fprintf(stderr, "%4d  ", i);
  }
  fprintf(stderr, "\n");

  for (i = 0; i < n; i++) {
    fprintf(stderr, "%2d: ", i);
    for (j = 0; j < csr_mat.column_indices[csr_mat.row_pointers[i]]; j++) {
      fprintf(stderr, "%4d, ", 0);
    }
    for (jj = csr_mat.row_pointers[i]; jj < csr_mat.row_pointers[i + 1]; jj++) {
      ii = csr_mat.column_indices[jj];
      fprintf(stderr, "%4.0f, ", csr_mat.values[jj]);
      for (j = csr_mat.column_indices[jj] + 1;
           j < csr_mat.column_indices[jj + 1]; j++) {
        fprintf(stderr, "%4d, ", 0);
      }
    }
    for (j = csr_mat.column_indices[csr_mat.row_pointers[i + 1] - 1]; j < n - 1;
         j++) {
      fprintf(stderr, "%4d, ", 0);
    }
    fprintf(stderr, "\n");
  }
#endif

  fprintf(stderr, "\n");
  print_csr_mat(&csr_mat);

#else
  char *mtx_filepath;
  int n, nnz;
  // int *A_diag_i;
  // int *A_diag_j;
  // double *A_diag_data;

  /* Parse command line */
  int arg_index = 0;
  while (arg_index < argc) {
    if (strcmp(argv[arg_index], "-mtx") == 0) {
      arg_index++;
      mtx_filepath = argv[arg_index++];
    } else {
      arg_index++;
    }
  }

  // SparseMatrixCOO coo_mat;
  SparseMatrixCOO coo_mat, t_coo_mat;

  fast_load_from_mtx_file(mtx_filepath, &t_coo_mat);
  sort_coo_row(&t_coo_mat, &coo_mat);
  convert_coo_to_csr(&coo_mat, &csr_mat, 1);

  n = csr_mat.nrows;
  nnz = csr_mat.nnz;

  // A_diag_i = csr_mat.row_pointers;
  // A_diag_j = csr_mat.column_indices;
  // A_diag_data = csr_mat.values;

#endif

  double *u_data, *su_data;
  double *f_data;
  u_data = (double *)malloc(sizeof(double) * n);
  su_data = (double *)malloc(sizeof(double) * n);
  f_data = (double *)malloc(sizeof(double) * n);

  for (i = 0; i < n; i++) {
    // u_data[i] = 1.0;
    // su_data[i] = 1.0;
    // f_data[i] = 1.0;
    u_data[i] = i + 0.1;
    su_data[i] = i + 0.1;
    f_data[i] = i + 0.1;
  }

  const int num_threads = omp_get_max_threads();

  // struct par_mat *A_diag =
  //     (struct par_mat *)malloc(sizeof(struct par_mat));

// schedule levels (forward & backward sbustitutions)
#if (DEBUG && DEBUG_LEVEL > 3)
  fprintf(stderr, "u_data\n");
  for (i = 0; i < n; i++) {
    fprintf(stderr, "%0.3f, ", u_data[i]);
  }
  fprintf(stderr, "\n");
#endif
  // perform iluc operation

  // free result arrays
  free(u_data);
  free(f_data);
  free(su_data);
}

void ser_iluc(const SparseMatrixCSR *csr_mat, SparseMatrixCSR *U,
              SparseMatrixCSR *L) {}

/****************************************************************
 * Util functions
 ***************************************************************/

/**
 * @brief convert CSR matrix into dense matrix for result validation
 *
 * @param csr_mat
 * @param dens_mat
 */
void densify(const SparseMatrixCSR *csr_mat, float *dens_mat) {

  const size_t n = csr_mat->nrows;
  const size_t n2 = n * n;
  int i, j, jj, idx;

  dens_mat = (float *)malloc(sizeof(float) * n2);
  memset(dens_mat, 0, sizeof(float) * n2);

  for (i = 0; i < n; i++) {
    for (j = csr_mat->row_pointers[i]; j < csr_mat->row_pointers[i + 1]; j++) {
      jj = csr_mat->column_indices[j];
      idx = i * n + jj;
      dens_mat[idx] = csr_mat->values[j];
    }
  }
}

void print_csr_mat(const SparseMatrixCSR *csr_mat) {
  const size_t n = csr_mat->nrows;
  size_t i, ii, j, jj;

  for (i = 0; i < n; i++) {
    fprintf(stderr, "%4d  ", i);
  }
  fprintf(stderr, "\n");

  for (i = 0; i < n; i++) {
    fprintf(stderr, "%2d: ", i);
    for (j = 0; j < csr_mat->column_indices[csr_mat->row_pointers[i]]; j++) {
      fprintf(stderr, "%4d, ", 0);
    }
    for (jj = csr_mat->row_pointers[i]; jj < csr_mat->row_pointers[i + 1];
         jj++) {
      ii = csr_mat->column_indices[jj];
      fprintf(stderr, "%4.0f, ", csr_mat->values[jj]);
      for (j = csr_mat->column_indices[jj] + 1;
           j < csr_mat->column_indices[jj + 1]; j++) {
        fprintf(stderr, "%4d, ", 0);
      }
    }
    for (j = csr_mat->column_indices[csr_mat->row_pointers[i + 1] - 1];
         j < n - 1; j++) {
      fprintf(stderr, "%4d, ", 0);
    }
    fprintf(stderr, "\n");
  }
}