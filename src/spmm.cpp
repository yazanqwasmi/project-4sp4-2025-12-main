// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmm.h"
#include "immintrin.h"
#include <omp.h>

namespace swiftware::hpp {

  void spmmCSR(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp) {
    // Sparse Matrix-Matrix multiplication: C = A * B + C
    // A is sparse (m x k) in CSR format, B is dense (k x n), C is dense (m x n)
    // A has row pointers Ap, column indices Ai, and values Ax
    // B and C are stored in row-major order
    
    // Parallelize over rows of A (and C)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
      const int row_start = Ap[i];
      const int row_end = Ap[i + 1];
      
      // For each non-zero in row i of A
      for (int j_ptr = row_start; j_ptr < row_end; ++j_ptr) {
        int j = Ai[j_ptr];         // column index in A (row index in B)
        float a_val = Ax[j_ptr];   // value of A[i,j]
        
        const float* B_row = B + j * n;   // pointer to row j of B
        float* C_row = C + i * n;         // pointer to row i of C
        
        // C[i,:] += a_val * B[j,:]
        // Use SIMD for the vector update
        #pragma omp simd
        for (int col = 0; col < n; ++col) {
          C_row[col] += a_val * B_row[col];
        }
      }
    }
  }

}