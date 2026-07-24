// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#ifndef PROJECT_DENSE_MATMUL_GEMV_H
#define PROJECT_DENSE_MATMUL_GEMV_H
// please do not change this file

#include "def.h"
namespace swiftware::hpp {

 // please do not change below
 /// \brief Matrix-vector multiplication
 /// \param m Number of rows of A
 /// \param n Number of columns of A
 /// \param A Matrix A
 /// \param x Vector x
 /// \param y Vector y
 void gemv(int m, int n, const float *A, const float *x, float *y, ScheduleParams Sp);

#ifdef USE_MKL
 void gemvMKL(int m, int n, const float *A, const float *x, float *y, ScheduleParams Sp);
#endif
}

#endif //PROJECT_DENSE_MATMUL_GEMV_H
