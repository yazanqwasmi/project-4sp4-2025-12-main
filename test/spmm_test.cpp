// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmm.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace swiftware::hpp {

// Helper function to create CSR from dense matrix
void denseToCsrForSpMM(const float* dense, int m, int n, 
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

// Test SpMM with identity sparse matrix
TEST(SpmmTest, IdentityMatrix) {
    int m = 3, k = 3, n = 2;
    // Identity sparse matrix A (3x3)
    float A_dense[9] = {1,0,0,
                        0,1,0,
                        0,0,1};
    
    std::vector<int> row_ptr, col_idx;
    std::vector<float> values;
    denseToCsrForSpMM(A_dense, m, k, row_ptr, col_idx, values);
    
    // Dense matrix B (3x2)
    float B[6] = {1, 2,
                  3, 4,
                  5, 6};
    
    float C[6] = {0, 0,
                  0, 0,
                  0, 0};
    
    ScheduleParams sp(2, 2);
    spmmCSR(m, n, k, row_ptr.data(), col_idx.data(), values.data(), B, C, sp);
    
    // C = I * B = B
    EXPECT_NEAR(C[0], 1.0f, 1e-5);
    EXPECT_NEAR(C[1], 2.0f, 1e-5);
    EXPECT_NEAR(C[2], 3.0f, 1e-5);
    EXPECT_NEAR(C[3], 4.0f, 1e-5);
    EXPECT_NEAR(C[4], 5.0f, 1e-5);
    EXPECT_NEAR(C[5], 6.0f, 1e-5);
}

// Test SpMM with sparse matrix
TEST(SpmmTest, SparseMatrix) {
    int m = 2, k = 3, n = 2;
    // Sparse matrix A (2x3):
    // [1, 0, 2]
    // [0, 3, 0]
    float A_dense[6] = {1, 0, 2,
                        0, 3, 0};
    
    std::vector<int> row_ptr, col_idx;
    std::vector<float> values;
    denseToCsrForSpMM(A_dense, m, k, row_ptr, col_idx, values);
    
    // Dense matrix B (3x2)
    float B[6] = {1, 2,
                  3, 4,
                  5, 6};
    
    float C[4] = {0, 0,
                  0, 0};
    
    ScheduleParams sp(-1, -1);
    spmmCSR(m, n, k, row_ptr.data(), col_idx.data(), values.data(), B, C, sp);
    
    // C[0,0] = 1*1 + 2*5 = 11
    // C[0,1] = 1*2 + 2*6 = 14
    // C[1,0] = 3*3 = 9
    // C[1,1] = 3*4 = 12
    EXPECT_NEAR(C[0], 11.0f, 1e-5);
    EXPECT_NEAR(C[1], 14.0f, 1e-5);
    EXPECT_NEAR(C[2], 9.0f, 1e-5);
    EXPECT_NEAR(C[3], 12.0f, 1e-5);
}

// Test SpMM accumulation
TEST(SpmmTest, Accumulation) {
    int m = 2, k = 2, n = 2;
    // A = [[1, 2], [3, 4]] fully dense
    float A_dense[4] = {1, 2,
                        3, 4};
    
    std::vector<int> row_ptr, col_idx;
    std::vector<float> values;
    denseToCsrForSpMM(A_dense, m, k, row_ptr, col_idx, values);
    
    // B = [[1, 0], [0, 1]] identity
    float B[4] = {1, 0,
                  0, 1};
    
    // C with initial values
    float C[4] = {10, 10,
                  10, 10};
    
    ScheduleParams sp(-1, -1);
    spmmCSR(m, n, k, row_ptr.data(), col_idx.data(), values.data(), B, C, sp);
    
    // A*B = A = [[1,2],[3,4]]
    // C_new = [[11, 12], [13, 14]]
    EXPECT_NEAR(C[0], 11.0f, 1e-5);
    EXPECT_NEAR(C[1], 12.0f, 1e-5);
    EXPECT_NEAR(C[2], 13.0f, 1e-5);
    EXPECT_NEAR(C[3], 14.0f, 1e-5);
}

// Test SpMM with zero row in sparse matrix
TEST(SpmmTest, ZeroRow) {
    int m = 3, k = 2, n = 2;
    // A with zero middle row
    float A_dense[6] = {1, 2,
                        0, 0,
                        3, 4};
    
    std::vector<int> row_ptr, col_idx;
    std::vector<float> values;
    denseToCsrForSpMM(A_dense, m, k, row_ptr, col_idx, values);
    
    float B[4] = {1, 1,
                  1, 1};
    
    float C[6] = {0, 0,
                  0, 0,
                  0, 0};
    
    ScheduleParams sp(-1, -1);
    spmmCSR(m, n, k, row_ptr.data(), col_idx.data(), values.data(), B, C, sp);
    
    EXPECT_NEAR(C[0], 3.0f, 1e-5);  // 1+2
    EXPECT_NEAR(C[1], 3.0f, 1e-5);
    EXPECT_NEAR(C[2], 0.0f, 1e-5);  // Zero row
    EXPECT_NEAR(C[3], 0.0f, 1e-5);
    EXPECT_NEAR(C[4], 7.0f, 1e-5);  // 3+4
    EXPECT_NEAR(C[5], 7.0f, 1e-5);
}

// Test SpMM with high sparsity
TEST(SpmmTest, HighSparsity) {
    int m = 4, k = 4, n = 2;
    // Very sparse matrix - only diagonal elements
    float A_dense[16] = {1, 0, 0, 0,
                         0, 2, 0, 0,
                         0, 0, 3, 0,
                         0, 0, 0, 4};
    
    std::vector<int> row_ptr, col_idx;
    std::vector<float> values;
    denseToCsrForSpMM(A_dense, m, k, row_ptr, col_idx, values);
    
    float B[8] = {1, 2,
                  3, 4,
                  5, 6,
                  7, 8};
    
    float C[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    ScheduleParams sp(-1, -1);
    spmmCSR(m, n, k, row_ptr.data(), col_idx.data(), values.data(), B, C, sp);
    
    // C = diag(1,2,3,4) * B
    EXPECT_NEAR(C[0], 1.0f, 1e-5);   // 1*1
    EXPECT_NEAR(C[1], 2.0f, 1e-5);   // 1*2
    EXPECT_NEAR(C[2], 6.0f, 1e-5);   // 2*3
    EXPECT_NEAR(C[3], 8.0f, 1e-5);   // 2*4
    EXPECT_NEAR(C[4], 15.0f, 1e-5);  // 3*5
    EXPECT_NEAR(C[5], 18.0f, 1e-5);  // 3*6
    EXPECT_NEAR(C[6], 28.0f, 1e-5);  // 4*7
    EXPECT_NEAR(C[7], 32.0f, 1e-5);  // 4*8
}

} // namespace swiftware::hpp
