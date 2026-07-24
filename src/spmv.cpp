// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmv.h"
#include <omp.h>

namespace swiftware::hpp {
  void spmvCSR(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp){
    // Sparse Matrix-Vector multiplication: c = A * b + c
    // A is stored in CSR format with row pointers Ap, column indices Ai, and values Ax
    // Parallelization over rows is efficient for CSR format
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
      float sum = 0.0f;
      const int row_start = Ap[i];
      const int row_end = Ap[i + 1];
      
      // Traverse non-zero elements in this row
      // SIMD reduction can help with the accumulation
      #pragma omp simd reduction(+:sum)
      for (int j = row_start; j < row_end; ++j) {
        int col = Ai[j];
        sum += Ax[j] * b[col];
      }
      
      // Accumulate into c
      c[i] += sum;
    }
  }


}