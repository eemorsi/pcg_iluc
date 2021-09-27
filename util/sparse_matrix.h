/* Copyright 2020 Barcelona Supercomputing Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <stdlib.h>
#include <stdint.h>

typedef float elem_t;

enum MatrixFormat
{
    COO,
    CSR,
    CSC,
    CSRSIMD,
    ELLPACK,
    VBSF
};

struct SparseMatrixCSR_STRUCT
{
    char *name;
    float *values; // values of matrix entries
    int *column_indices;
    int *row_pointers;
    int nrows;
    int ncolumns;
    int nnz;
};
typedef struct SparseMatrixCSR_STRUCT SparseMatrixCSR;

struct SparseMatrixELLPACK_STRUCT
{
    char *name;
    elem_t *values;
    size_t *column_indices;
    size_t max_row_size;
    size_t nrows;
    size_t ncolumns;
    size_t nnz;
};
typedef struct SparseMatrixELLPACK_STRUCT SparseMatrixELLPACK;

struct SparseMatrixCOO_STRUCT
{
    char *name;
    float *values;  // values of matrix entries
    int *rows;    // row_index
    int *columns; // col_index
    int nrows;
    int ncolumns;
    int nnz;
};
typedef struct SparseMatrixCOO_STRUCT SparseMatrixCOO;

// void load_from_mtx_file(const char *mtx_filepath, SparseMatrixCOO *coo_matrix);
void fast_load_from_mtx_file(const char *mtx_filepath, SparseMatrixCOO *coo_matrix);
void convert_coo_to_csr(const SparseMatrixCOO *coo_matrix, SparseMatrixCSR *csr_matrix, int free_coo);
void sort_coo_row(const SparseMatrixCOO *coo_matrix,
                  SparseMatrixCOO *s_coo_matrix) ;
#endif // SPARSE_MATRIX_H
