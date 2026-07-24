// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef PROJECT_SPMV_H
#define PROJECT_SPMV_H
#include "def.h"

namespace swiftware::hpp {

  // please do not change below
  /// \brief Sparse Matrix-vector multiplication
  /// \param m Number of rows of A
  /// \param n Number of columns of A
  /// \param Ap Pointer to the start of the row pointer array
  /// \param Ai Pointer to the column index array
  /// \param Ax Pointer to the value array
  /// \param b Vector b
  /// \param c Vector c
  /// \param Sp Schedule parameters
  void spmvCSR(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp);


}

#endif //PROJECT_SPMV_H
