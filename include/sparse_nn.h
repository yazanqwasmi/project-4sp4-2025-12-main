// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#ifndef PROJECT_SPAARSE_MATMUL_SPARSE_NN_H
#define PROJECT_SPAARSE_MATMUL_SPARSE_NN_H
#include "def.h"
#include "spmm.h"
#include "spmv.h"


namespace swiftware::hpp {

  // please do not change below
  DenseMatrix *sparseNNSpmm(DenseMatrix *InData, CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);

  DenseMatrix *sparseNNSpmv(DenseMatrix *InData, CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);


}
#endif //PROJECT_DENSE_MATMUL_SPARSE_NN_H
