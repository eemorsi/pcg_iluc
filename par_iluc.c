#include "par_iluc.h"
#include <omp.h>

int main(int argc, char *argv[]) {

  size_t i, ii, jj, j, m;
  size_t t_id, size, rest, ne, ns, inc;
  SparseMatrixCSR csr_mat;

#if MANUAL
  const int n = 8 * 2;
  const int nnz = 23 * 4;
  csr_mat.nnz = nnz;
  csr_mat.nrows = n;
  csr_mat.ncolumns = n;

  csr_mat.row_pointers = (int[n + 1]){0,  6,  14, 18, 24, 28, 36, 40, 46,
                                      52, 60, 64, 70, 74, 82, 86, 92};

  csr_mat.column_indices =
      (int[nnz]){0, 2, 4, 5, 10, 12, 1, 2, 3, 6, 9, 10, 11, 14, 0, 2, 8, 10, 1,
                 3, 4, 9, 11, 15, 3, 4, 11, 12, 1, 2, 5, 6, 9, 10, 11, 14, 6, 7,
                 13, 14, 4, 5, 7, 9, 13, 15,
                 /**LOWER*/
                 0, 2, 4, 8, 10, 12, 1, 2, 3, 7, 9, 10, 11, 14, 0, 2, 8, 11, 1,
                 3, 4, 9, 10, 12, 3, 4, 11, 12, 1, 2, 5, 6, 9, 10, 11, 14, 5, 6,
                 13, 14, 4, 5, 7, 11, 12, 15};
  csr_mat.values = (elem_t *)malloc(sizeof(elem_t) * nnz);

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
  // elem_t *A_diag_data;

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

  elem_t *u_data, *su_data;
  elem_t *f_data;
  u_data = (elem_t *)malloc(sizeof(elem_t) * n);
  su_data = (elem_t *)malloc(sizeof(elem_t) * n);
  f_data = (elem_t *)malloc(sizeof(elem_t) * n);

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

#if (DEBUG && DEBUG_LEVEL > 3)
  fprintf(stderr, "u_data\n");
  for (i = 0; i < n; i++) {
    fprintf(stderr, "%0.3f, ", u_data[i]);
  }
  fprintf(stderr, "\n");
#endif
  // perform iluc operation

  elem_t *mat1, *mat2;
  densify(&csr_mat, &mat1);
  const size_t nbytes = sizeof(elem_t) * csr_mat.nrows * csr_mat.ncolumns;
  // mat2 = (elem_t *)malloc(nbytes);
  // memcpy(mat2, mat1, nbytes);

  // LU_fac(csr_mat.nrows, csr_mat.ncolumns, mat1);

  print_dens_mat(csr_mat.nrows, csr_mat.ncolumns, mat1);
  // free factorized matrices
  free(mat1);
  free(mat2);
  // free result arrays
  free(u_data);
  free(f_data);
  free(su_data);
}

void ser_iluc(const SparseMatrixCSR *csr_mat, SparseMatrixCSR *U,
              SparseMatrixCSR *L) {}

#ifdef __INTEL_COMPILER
/*---------------------------------------------------------------------------
/* Calculate ILU0 preconditioner.
/*                      !ATTENTION!
/* DCSRILU0 routine uses some IPAR, DPAR set by DFGMRES_INIT routine.
/* Important for DCSRILU0 default entries set by DFGMRES_INIT are
/* ipar[1] = 6 - output of error messages to the screen,
/* ipar[5] = 1 - allow output of errors,
/* ipar[30]= 0 - abort DCSRILU0 calculations if routine meets zero diagonal
element.
/*
/* If ILU0 is going to be used out of MKL FGMRES context, than the values
/* of ipar[1], ipar[5], ipar[30], dpar[30], and dpar[31] should be user
/* provided before the DCSRILU0 routine call.
/*
/* In this example, specific for DCSRILU0 entries are set in turn:
/* ipar[30]= 1 - change small diagonal value to that given by dpar[31],
/* dpar[30]= 1.E-20 instead of the default value set by DFGMRES_INIT.
/*                  It is a small value to compare a diagonal entry with it.
/* dpar[31]= 1.E-16 instead of the default value set by DFGMRES_INIT.
/*                  It is the target value of the diagonal value if it is
/*                  small as compared to dpar[30] and the routine should change
/*                  it rather than abort DCSRILU0 calculations.
/*---------------------------------------------------------------------------*/

void mkl_dcsrilu0(const SparseMatrixCSR *csr_mat) {

#define size 128

  const size_t N = csr_mat.nrows;
  MKL_INT ipar[size];
  elem_t dpar[size], tmp[N * (2 * N + 1) + (N * (N + 9)) / 2 + 1];
  elem_t trvec[N], bilu0[csr_mat.nnz];
  elem_t expected_solution[N];
  elem_t rhs[N], b[N];
  elem_t computed_solution[N];
  elem_t residual[N];

  MKL_INT matsize = 12, incx = 1, ref_nit = 2;
  double ref_norm2 = 7.772387E+0, nrm2;

  ipar[30] = 1;
  dpar[30] = 1.E-20;
  dpar[31] = 1.E-16;

  dcsrilu0(&ivar, A, ia, ja, bilu0, ipar, dpar, &ierr);
  nrm2 = dnrm2(&matsize, bilu0, &incx);

  if (ierr != 0) {
    printf("Preconditioner dcsrilu0 has returned the ERROR code %d", ierr);
    goto FAILED1;
  }
}
#endif
/****************************************************************
 * Util functions
 ***************************************************************/

/**
 * @brief convert CSR matrix into dense matrix for result validation
 *
 * @param csr_mat
 * @param dens_mat
 */
void densify(const SparseMatrixCSR *csr_mat, elem_t **dens_mat) {

  const size_t n = csr_mat->nrows;
  const size_t lda = csr_mat->ncolumns;
  const size_t n2 = n * lda;
  int i, j, jj, idx;

  *dens_mat = (elem_t *)malloc(sizeof(elem_t) * n2);
  memset((*dens_mat), 0, sizeof(elem_t) * n2);

  for (i = 0; i < n; i++) {
    for (j = csr_mat->row_pointers[i]; j < csr_mat->row_pointers[i + 1]; j++) {
      jj = csr_mat->column_indices[j];
      idx = i * lda + jj;
      (*dens_mat)[idx] = csr_mat->values[j];
    }
  }
}

void print_dens_mat(const int n, const int lda, const elem_t *mat) {
  int i, j;
  for (i = 0; i < lda; i++) {
    fprintf(stderr, "%4d  ", i);
  }
  fprintf(stderr, "\n");

  for (i = 0; i < n; i++) {
    fprintf(stderr, "%2d: ", i);
    for (j = 0; j < lda; j++) {
      fprintf(stderr, "%4.0f, ", mat[i * lda + j]);
    }
    fprintf(stderr, "\n");
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
// #ifdef __INTEL_COMPILER
void LU_fac(const int n, const int lda, elem_t *const mat) {
  int i, j, k, cnt;
  for (k = 0; k < n - 1; ++k) {
    i = k + 1;
    j = i;
    cnt = n - j;
    cblas_dscal((n - i), 1 / mat[k * lda + k], &mat[i * lda + k], lda);

#pragma omp parallel for shared(mat)
    for (; i < n; i++) {
      cblas_daxpy(cnt, -1 * mat[i * lda + k], &mat[k * lda + j], 1,
                  &mat[i * lda + j], 1);
    }
  }
}
void LU_fac_2(const int n, const int lda, elem_t *const mat) {
  int i, j, k;
  for (k = 0; k < n; k++) {
    i = k + 1;
    elem_t kk = mat[k * lda + k];
#pragma omp parallel for shared(mat) /*default(none) schedule(static, 8) */
    for (; i < n; i++) {
      mat[i * lda + k] = mat[i * lda + k] / kk;
      elem_t ik = mat[i * lda + k];
#pragma omp simd
#pragma ivdep
#pragma unroll(16)
      for (j = k + 1; j < n; j++) {
        mat[i * lda + j] -= ik * mat[k * lda + j];
      }
    }
  }
}
// #endif