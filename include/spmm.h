// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef PROJECT_SPARSE_MATMUL_SPMM_H
#define PROJECT_SPARSE_MATMUL_SPMM_H

// please do not change this file

#include "def.h"

namespace swiftware::hpp {

  // please do not change below
  /// \brief Sparse Matrix-matrix multiplication
  /// \param m Number of rows of A
  /// \param n Number of columns of B
  /// \param k Number of columns of A
  /// \param Ap Pointer to the start of the row pointer array
  /// \param Ai Pointer to the column index array
  /// \param Ax Pointer to the value array
  /// \param B Matrix B
  /// \param C Matrix C
  /// \param Sp Schedule parameters
  void spmmCSR(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp);

}

#endif //PROJECT_DENSE_MATMUL_SPMM_H
