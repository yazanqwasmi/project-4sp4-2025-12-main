// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef PROJECT_DENSE_MATMUL_DENSE_NN_H
#define PROJECT_DENSE_MATMUL_DENSE_NN_H
#include "def.h"
#include "gemm.h"
#include "gemv.h"

namespace swiftware::hpp {

 // please do not change below
 DenseMatrix *dense_nn_gemm(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);

 DenseMatrix *dense_nn_gemv(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);

 #ifdef USE_MKL
    DenseMatrix *dense_nn_mkl_gemm(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);
    DenseMatrix *dense_nn_mkl_gemv(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);
 #endif
}
#endif //PROJECT_DENSE_MATMUL_DENSE_NN_H
