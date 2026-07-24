// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmv.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace swiftware::hpp {

// Helper function to create CSR from dense matrix
void denseToCsr(const float* dense, int m, int n, 
                std::vector<int>& row_ptr, std::vector<int>& col_idx, std::vector<float>& values) {
    row_ptr.clear();
    col_idx.clear();
    values.clear();
    row_ptr.push_back(0);
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float val = dense[i * n + j];
            if (std::abs(val) > 1e-10) {
                col_idx.push_back(j);
                values.push_back(val);
            }
        }
        row_ptr.push_back(col_idx.size());
    }
}

// Test SpMV with identity matrix
TEST(SpmvTest, IdentityMatrix) {
    int m = 4, n = 4;
    // Identity matrix in dense format
    float A_dense[16] = {1,0,0,0,
                         0,1,0,0,
                         0,0,1,0,
                         0,0,0,1};
    
    std::vector<int> row_ptr, col_idx;
    std::vector<float> values;
    denseToCsr(A_dense, m, n, row_ptr, col_idx, values);
    
    float x[4] = {1, 2, 3, 4};
    float y[4] = {0, 0, 0, 0};
    
    ScheduleParams sp(2, 2);
    spmvCSR(m, n, row_ptr.data(), col_idx.data(), values.data(), x, y, sp);
    
    EXPECT_NEAR(y[0], 1.0f, 1e-5);
    EXPECT_NEAR(y[1], 2.0f, 1e-5);
    EXPECT_NEAR(y[2], 3.0f, 1e-5);
    EXPECT_NEAR(y[3], 4.0f, 1e-5);
}

// Test SpMV with sparse matrix
TEST(SpmvTest, SparseMatrix) {
    int m = 3, n = 4;
    // Sparse matrix:
    // [1, 0, 2, 0]
    // [0, 3, 0, 4]
    // [5, 0, 6, 0]
    float A_dense[12] = {1, 0, 2, 0,
                         0, 3, 0, 4,
                         5, 0, 6, 0};
    
    std::vector<int> row_ptr, col_idx;
    std::vector<float> values;
    denseToCsr(A_dense, m, n, row_ptr, col_idx, values);
    
    float x[4] = {1, 1, 1, 1};
    float y[3] = {0, 0, 0};
    
    ScheduleParams sp(-1, -1);
    spmvCSR(m, n, row_ptr.data(), col_idx.data(), values.data(), x, y, sp);
    
    EXPECT_NEAR(y[0], 3.0f, 1e-5);
    EXPECT_NEAR(y[1], 7.0f, 1e-5);
    EXPECT_NEAR(y[2], 11.0f, 1e-5);
}

// Test SpMV accumulation (y = Ax + y_initial)
TEST(SpmvTest, Accumulation) {
    int m = 2, n = 3;
    // Matrix: [1, 2, 3]
    //         [4, 5, 6]
    float A_dense[6] = {1, 2, 3,
                        4, 5, 6};
    
    std::vector<int> row_ptr, col_idx;
    std::vector<float> values;
    denseToCsr(A_dense, m, n, row_ptr, col_idx, values);
    
    float x[3] = {1, 1, 1};
    float y[2] = {10, 20}; 
    
    ScheduleParams sp(-1, -1);
    spmvCSR(m, n, row_ptr.data(), col_idx.data(), values.data(), x, y, sp);
 
    EXPECT_NEAR(y[0], 16.0f, 1e-5);
    EXPECT_NEAR(y[1], 35.0f, 1e-5);
}

// Test SpMV with all zeros row
TEST(SpmvTest, ZeroRow) {
    int m = 3, n = 3;
    // Matrix with zero middle row:
    // [1, 2, 3]
    // [0, 0, 0]
    // [7, 8, 9]
    float A_dense[9] = {1, 2, 3,
                        0, 0, 0,
                        7, 8, 9};
    
    std::vector<int> row_ptr, col_idx;
    std::vector<float> values;
    denseToCsr(A_dense, m, n, row_ptr, col_idx, values);
    
    float x[3] = {1, 1, 1};
    float y[3] = {0, 0, 0};
    
    ScheduleParams sp(-1, -1);
    spmvCSR(m, n, row_ptr.data(), col_idx.data(), values.data(), x, y, sp);
    
    EXPECT_NEAR(y[0], 6.0f, 1e-5);
    EXPECT_NEAR(y[1], 0.0f, 1e-5);  // Zero row
    EXPECT_NEAR(y[2], 24.0f, 1e-5);
}

// Test SpMV with single non-zero element
TEST(SpmvTest, SingleElement) {
    int m = 3, n = 3;
    float A_dense[9] = {0, 0, 0,
                        0, 5, 0,
                        0, 0, 0};
    
    std::vector<int> row_ptr, col_idx;
    std::vector<float> values;
    denseToCsr(A_dense, m, n, row_ptr, col_idx, values);
    
    float x[3] = {1, 2, 3};
    float y[3] = {0, 0, 0};
    
    ScheduleParams sp(-1, -1);
    spmvCSR(m, n, row_ptr.data(), col_idx.data(), values.data(), x, y, sp);
    
    EXPECT_NEAR(y[0], 0.0f, 1e-5);
    EXPECT_NEAR(y[1], 10.0f, 1e-5); 
    EXPECT_NEAR(y[2], 0.0f, 1e-5);
}

} // namespace swiftware::hpp
