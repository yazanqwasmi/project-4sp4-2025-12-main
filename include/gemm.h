// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#ifndef PROJECT_DENSE_MATMUL_GEMM_H
#define PROJECT_DENSE_MATMUL_GEMM_H

// please do not change this file

#include "def.h"

namespace swiftware::hpp {

 // please do not change below
/// \brief Matrix-matrix multiplication
/// \param m Number of rows of A and C
/// \param n Number of columns of B and C
/// \param k Number of columns of A and rows of B
/// \param A Matrix A
/// \param B Matrix B
/// \param C Matrix C
 void gemm(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp);


#ifdef USE_MKL
 void gemmMKL(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp);
#endif
}

#endif //PROJECT_DENSE_MATMUL_GEMM_H
