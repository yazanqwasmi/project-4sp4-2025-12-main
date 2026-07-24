// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#ifndef LAB01_GPU_SPARSE_NN_CUH
#define LAB01_GPU_SPARSE_NN_CUH

#include "def.h"

namespace swiftware::hpp
{
    DenseMatrix* gpu_sparse_nn_spmm(DenseMatrix* InData, 
                                    CSR* W1, CSR* W2, 
                                    DenseMatrix* B1, DenseMatrix* B2, 
                                    ScheduleParams Sp);

    DenseMatrix* gpu_sparse_nn_spmv(DenseMatrix* InData, 
                                    CSR* W1, CSR* W2, 
                                    DenseMatrix* B1, DenseMatrix* B2, 
                                    ScheduleParams Sp);

    DenseMatrix* gpu_sparse_nn_cusparse(DenseMatrix* InData, 
                                        CSR* W1, CSR* W2, 
                                        DenseMatrix* B1, DenseMatrix* B2, 
                                        ScheduleParams Sp);

}  // namespace swiftware::hpp

#endif //LAB01_GPU_SPARSE_NN_CUH
