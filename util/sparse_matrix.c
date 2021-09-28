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

#include "sparse_matrix.h"
#include "mmio.h"
#include <inttypes.h>
#include <libgen.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifdef __ve__
#include <asl.h>
#endif

#define align_size 128
/* 
    Loads a Sparse Matrix from a .MTX file format
    into a SparseMatrixCOO data structure.

    More info about the MTX format @ https://math.nist.gov/MatrixMarket/
    Find the biggest MTX repository @ https://sparse.tamu.edu/
*/
void fast_load_from_mtx_file(const char *mtx_filepath, SparseMatrixCOO *coo_matrix)
{

    int ret_code;
    unsigned int mtx_rows, mtx_cols, mtx_entries;
    FILE *f;
    MM_typecode matcode;

    if ((f = fopen(mtx_filepath, "r")) == NULL)
    {
        fprintf(stderr, "Could not open file: %s \n", mtx_filepath);
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        exit(1);
    }

    /* 
        This code, will only work with MTX containing: REAL number, Sparse, Matrices.
        Throws an error otherwise. See mmio.h for more information.
    */
    if ((!mm_is_real(matcode) && !mm_is_pattern(matcode)) || !mm_is_matrix(matcode) || !mm_is_sparse(matcode))
    {
        fprintf(stderr, "Market Market type: [%s] not supported\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* Get the number of matrix rows and columns */
    if ((ret_code = mm_read_mtx_crd_size(f, &mtx_rows, &mtx_cols, &mtx_entries)) != 0)
    {
        fprintf(stderr, "Error while reading matrix dimension sizes.\n");
        exit(1);
    }

    long current_stream_position = ftell(f);
    fseek(f, 0, SEEK_END);
    long nnz_string_size = ftell(f) - current_stream_position;
    fseek(f, current_stream_position, SEEK_SET); // Leave the pointer where it was before

    char *nnz_string = (char *) malloc(nnz_string_size + 1);
    fread(nnz_string, 1, nnz_string_size, f);
    fclose(f);

    /* Fill COO struct */
    coo_matrix->nrows = mtx_rows;
    coo_matrix->ncolumns = mtx_cols;
    coo_matrix->name = basename((char *)mtx_filepath);

    int nnz_count = 0, i;
    if (mm_is_symmetric(matcode))
    {
        int max_entries = 2 * mtx_entries; // 2 * mtx_entries is an upper bound
        coo_matrix->rows = (int *)malloc(max_entries * sizeof(int));
        // check_mem_alloc(coo_matrix->rows, "coo rows");

        coo_matrix->columns = (int *)malloc(max_entries * sizeof(int));
        // check_mem_alloc(coo_matrix->columns, "coo cols");

        coo_matrix->values = (elem_t *)malloc(max_entries * sizeof(elem_t));
        // check_mem_alloc(coo_matrix->values, "coo values");

        // Load Symmetric MTX, note that COO might be unordered.
        if (!mm_is_pattern(matcode))
        {
            char *line_ptr = nnz_string;
            char *next_token;

            for (i = 0; i < mtx_entries; i++)
            {
                coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->values[nnz_count] = strtod(line_ptr, &next_token);
                line_ptr = next_token;

                if (coo_matrix->rows[nnz_count] == coo_matrix->columns[nnz_count])
                {
                    nnz_count++;
                }
                else
                {
                    coo_matrix->rows[nnz_count + 1] = coo_matrix->columns[nnz_count];
                    coo_matrix->columns[nnz_count + 1] = coo_matrix->rows[nnz_count];
                    coo_matrix->values[nnz_count + 1] = coo_matrix->values[nnz_count];
                    nnz_count = nnz_count + 2;
                }
            }
        }
        else
        {
            char *line_ptr = nnz_string;
            char *next_token;

            for (i = 0; i < mtx_entries; i++)
            {
                coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->values[nnz_count] = 1.0f;
                // fprintf(stderr, " %lu %lu %lf\n", coo_matrix->rows[nnz_count], coo_matrix->columns[nnz_count], coo_matrix->values[nnz_count]);
                if (coo_matrix->rows[nnz_count] == coo_matrix->columns[nnz_count])
                {
                    nnz_count++;
                }
                else
                {
                    coo_matrix->rows[nnz_count + 1] = coo_matrix->columns[nnz_count];
                    coo_matrix->columns[nnz_count + 1] = coo_matrix->rows[nnz_count];
                    coo_matrix->values[nnz_count + 1] = 1.0f;
                    nnz_count = nnz_count + 2;
                }
            }
        }
    }
    else
    {
        coo_matrix->rows = (int *)malloc(mtx_entries * sizeof(int));
        // check_mem_alloc(coo_matrix->rows, "coo rows");

        coo_matrix->columns = (int *)malloc(mtx_entries * sizeof(int));
        // check_mem_alloc(coo_matrix->columns, "coo cols");

        coo_matrix->values = (elem_t *)malloc(mtx_entries * sizeof(elem_t));
        // check_mem_alloc(coo_matrix->values, "coo values");

        if (!mm_is_pattern(matcode))
        {
            char *line_ptr = nnz_string;
            char *next_token;

            for (i = 0; i < mtx_entries; i++)
            {
                coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->values[nnz_count] = strtod(line_ptr, &next_token);
                line_ptr = next_token;
                nnz_count++;
            }
        }
        else
        {
            char *line_ptr = nnz_string;
            char *next_token;

            for (i = 0; i < mtx_entries; i++)
            {
                coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->values[nnz_count] = 1.0f;
                nnz_count++;
            }
        }
    }
    // TODO: REMOVE EXPLICIT 0's. apparently some matrices have few (~0.3%). it does not affect the GFLOPS per se.
    coo_matrix->nnz = nnz_count;
    free(nnz_string);

}

void sort_coo_row(const SparseMatrixCOO *coo_matrix,
                  SparseMatrixCOO *s_coo_matrix) {
  int i;

#ifdef __ve__

  asl_sort_t sort;
  asl_library_initialize();
  asl_sort_create_i32(&sort, ASL_SORTORDER_ASCENDING,
                      ASL_SORTALGORITHM_AUTO_STABLE);
  asl_sort_preallocate(sort, coo_matrix->nnz);

  int *idx = (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);
  int *vals_idx =
      (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);

  s_coo_matrix->values =
      (elem_t *)aligned_alloc(align_size, sizeof(elem_t) * coo_matrix->nnz);
  s_coo_matrix->rows =
      (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);

  s_coo_matrix->columns =
      (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);
  //   int *tcolumns =
  //       (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);
  s_coo_matrix->nnz = coo_matrix->nnz;
  s_coo_matrix->nrows = coo_matrix->nrows;
  s_coo_matrix->ncolumns = coo_matrix->ncolumns;

  for (i = 0; i < coo_matrix->nnz; i++) {
    vals_idx[i] = i;
  }
  //   asl_sort_execute_i32(sort, coo_matrix->nnz, s_coo_matrix->rows,
  //                        ASL_NULL, s_coo_matrix->rows,
  //                        idx);

  asl_sort_execute_i32(sort, coo_matrix->nnz, coo_matrix->rows, vals_idx,
                       s_coo_matrix->rows, idx);

  for (i = 0; i < coo_matrix->nnz; i++) {
    s_coo_matrix->values[i] = coo_matrix->values[idx[i]];
    s_coo_matrix->columns[i] = coo_matrix->columns[idx[i]];
  }
  free(idx);
  free(vals_idx);

  /* Sorting Finalization */
  asl_sort_destroy(sort);
  /* Library Finalization */
  asl_library_finalize();

#else
  int *freq, *inc, *jmp;

  freq = (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nrows);
  inc = (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nrows);
  jmp = (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nrows);

  // printf("the number of rows: %d\n", coo_matrix->nrows);

  // memset(inc, 0, coo_matrix->nrows);
  // memset(freq, 0, coo_matrix->nrows);
  for (i = 0; i < coo_matrix->nrows; i++) {
    inc[i] = 0;
    freq[i] = 0;
  }
  for (i = 0; i < coo_matrix->nnz; i++) {
    freq[coo_matrix->rows[i]]++;
  }

  jmp[0] = 0;
  for (i = 1; i < coo_matrix->nrows; i++) {
    jmp[i] = jmp[i - 1] + freq[i - 1];
    // freq[i] += freq[i - 1];
    // printf("%d -> %d\n", i, freq[i]);
  }
  s_coo_matrix->values =
      (elem_t *)aligned_alloc(align_size, sizeof(elem_t) * coo_matrix->nnz);
  s_coo_matrix->rows =
      (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);
  s_coo_matrix->columns =
      (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);

  s_coo_matrix->nnz = coo_matrix->nnz;
  s_coo_matrix->nrows = coo_matrix->nrows;
  s_coo_matrix->ncolumns = coo_matrix->ncolumns;

  for (i = 0; i < coo_matrix->nnz; i++) {
    int r = coo_matrix->rows[i];
    int p = jmp[r] + inc[r];
    s_coo_matrix->values[p] = coo_matrix->values[i];
    s_coo_matrix->rows[p] = coo_matrix->rows[i];
    s_coo_matrix->columns[p] = coo_matrix->columns[i];
    inc[r]++;
  }

  //   printf("compare rows %d\n", coo_matrix->nnz);
  //   for (i = 0; i < coo_matrix->nnz; i++) {
  //     if (s_coo_matrix->columns[i] != ts_coo_matrix.columns[i])
  //       printf("col: %d\t%d\n", s_coo_matrix->columns[i],
  //       ts_coo_matrix.columns[i]);
  //     if (s_coo_matrix->rows[i] != ts_coo_matrix.rows[i])
  //       printf("row: %d\t%d\n", s_coo_matrix->rows[i],
  //       ts_coo_matrix.rows[i]);
  //     if (s_coo_matrix->values[i] != ts_coo_matrix.values[i])
  //       printf("val: %f\t%f\n", s_coo_matrix->values[i],
  //       ts_coo_matrix.values[i]);
  //   }
  //   exit(0);
#endif
}

void convert_coo_to_csr(const SparseMatrixCOO *coo_matrix, SparseMatrixCSR *csr_matrix, int free_coo)
{

    /* Allocate CSR Matrix data structure in memory */
    csr_matrix->row_pointers = (int *)malloc((coo_matrix->nrows + 1) * sizeof(int));
    // check_mem_alloc(csr_matrix->row_pointers, "SparseMatrixCSR.row_pointers");
    memset(csr_matrix->row_pointers, 0, (coo_matrix->nrows + 1) * sizeof(int));

    csr_matrix->column_indices = (int *)malloc(coo_matrix->nnz * sizeof(int));
    // check_mem_alloc(csr_matrix->column_indices, "SparseMatrixCSR.column_indices");

    csr_matrix->values = (elem_t *)malloc(coo_matrix->nnz * sizeof(elem_t));
    // check_mem_alloc(csr_matrix->values, "SparseMatrixCSR.values");

    // Store the number of Non-Zero elements in each Row
    int i;
    for (i = 0; i < coo_matrix->nnz; i++)
        csr_matrix->row_pointers[coo_matrix->rows[i]]++;

    // Update Row Pointers so they consider the previous pointer offset
    // (using accumulative sum).
    int cum_sum = 0;
    for (i = 0; i < coo_matrix->nrows; i++)
    {
        int row_nnz = csr_matrix->row_pointers[i];
        csr_matrix->row_pointers[i] = cum_sum;
        cum_sum += row_nnz;
    }

    /*  Adds COO values to CSR

        Note: Next block of code reuses csr->row_pointers[] to keep track of the values added from
        the COO matrix.
        This way is able to create the CSR matrix even if the COO matrix is not ordered by row.
        In the process, it 'trashes' the row pointers by shifting them one position up.
        At the end, each csr->row_pointers[i+1] should be in csr->row_pointers[i] */

    for (i = 0; i < coo_matrix->nnz; i++)
    {
        int row_index = coo_matrix->rows[i];
        int column_index = coo_matrix->columns[i];
        elem_t value = coo_matrix->values[i];

        int j = csr_matrix->row_pointers[row_index];
        csr_matrix->column_indices[j] = column_index;
        csr_matrix->values[j] = value;
        csr_matrix->row_pointers[row_index]++;
    }

    // Restore the correct row_pointers
    for (i = coo_matrix->nrows - 1; i > 0; i--)
    {
        csr_matrix->row_pointers[i] = csr_matrix->row_pointers[i - 1];
    }
    csr_matrix->row_pointers[0] = 0;
    csr_matrix->row_pointers[coo_matrix->nrows] = coo_matrix->nnz;

    csr_matrix->nnz = coo_matrix->nnz;
    csr_matrix->nrows = coo_matrix->nrows;
    csr_matrix->ncolumns = coo_matrix->ncolumns;
    csr_matrix->name = coo_matrix->name;

    /*  For each row, sort the corresponding arrasy csr.column_indices and csr.values

        TODO: We should check if this step makes sense or can be optimized
        1) If the .mtx format by definition is ordered 
        2) If we force the COO Matrix to be ordered first, we can avoid this
        3) Test speed of standard library sorting vs current sorting approach. */

    if (free_coo)
    {
        free(coo_matrix->values);
        free(coo_matrix->rows);
        free(coo_matrix->columns);
    }
}
