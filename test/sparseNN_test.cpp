// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "sparse_nn.h"
#include "dense_nn.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

namespace swiftware::hpp {

// Helper to convert dense matrix to CSR
CSR denseToCSR(const DenseMatrix& dense) {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
    row_ptr.push_back(0);
    
    for (int i = 0; i < dense.m; ++i) {
        for (int j = 0; j < dense.n; ++j) {
            float val = dense.data[i * dense.n + j];
            if (std::abs(val) > 1e-10f) {
                col_idx.push_back(j);
                values.push_back(val);
            }
        }
        row_ptr.push_back(col_idx.size());
    }
    
    CSR csr(dense.m, dense.n, values.size());
    csr.row_ptr = row_ptr;
    csr.col_idx = col_idx;
    csr.values = values;
    return csr;
}

// Test Sparse NN with fully dense weights (should match Dense NN)
TEST(SparseNNTest, DenseWeightsMatchDenseNN) {
    int batchSize = 2;
    int inputDim = 4;
    int hiddenDim = 3;
    int outputDim = 2;
    
    DenseMatrix input(batchSize, inputDim);
    input.data = {0.5f, -0.3f, 0.8f, -0.1f,
                  0.2f, 0.7f, -0.4f, 0.9f};
    
    // Fully dense W1
    DenseMatrix W1_dense(hiddenDim, inputDim);
    W1_dense.data = {0.1f, 0.2f, -0.1f, 0.3f,
                     -0.2f, 0.1f, 0.4f, -0.3f,
                     0.3f, -0.4f, 0.2f, 0.1f};
    
    DenseMatrix B1(hiddenDim, 1);
    B1.data = {0.1f, -0.1f, 0.05f};
    
    // Fully dense W2
    DenseMatrix W2_dense(outputDim, hiddenDim);
    W2_dense.data = {0.2f, -0.3f, 0.1f,
                     -0.1f, 0.4f, -0.2f};
    
    DenseMatrix B2(outputDim, 1);
    B2.data = {0.05f, -0.05f};
    
    // Convert to CSR
    CSR W1_csr = denseToCSR(W1_dense);
    CSR W2_csr = denseToCSR(W2_dense);
    
    ScheduleParams sp(32, 32);
    
    // Run both
    DenseMatrix* dense_output = dense_nn_gemm(&input, &W1_dense, &W2_dense, &B1, &B2, sp);
    DenseMatrix* sparse_output = sparseNNSpmm(&input, &W1_csr, &W2_csr, &B1, &B2, sp);
    
    ASSERT_NE(dense_output, nullptr);
    ASSERT_NE(sparse_output, nullptr);
    ASSERT_EQ(dense_output->m, sparse_output->m);
    ASSERT_EQ(dense_output->n, sparse_output->n);
    
    // Should produce same results
    for (int i = 0; i < batchSize * outputDim; ++i) {
        EXPECT_NEAR(dense_output->data[i], sparse_output->data[i], 1e-4f)
            << "Mismatch at index " << i;
    }
    
    delete dense_output;
    delete sparse_output;
}

// Test Sparse NN SpMM and SpMV produce same results
TEST(SparseNNTest, SpmmSpmvConsistency) {
    int batchSize = 2;
    int inputDim = 4;
    int hiddenDim = 3;
    int outputDim = 2;
    
    DenseMatrix input(batchSize, inputDim);
    input.data = {0.5f, -0.3f, 0.8f, -0.1f,
                  0.2f, 0.7f, -0.4f, 0.9f};
    
    // 50% sparse W1
    DenseMatrix W1_dense(hiddenDim, inputDim);
    W1_dense.data = {0.1f, 0.0f, -0.1f, 0.0f,
                     0.0f, 0.1f, 0.0f, -0.3f,
                     0.3f, 0.0f, 0.2f, 0.0f};
    
    DenseMatrix B1(hiddenDim, 1);
    B1.data = {0.1f, -0.1f, 0.05f};
    
    // 50% sparse W2
    DenseMatrix W2_dense(outputDim, hiddenDim);
    W2_dense.data = {0.2f, 0.0f, 0.1f,
                     0.0f, 0.4f, 0.0f};
    
    DenseMatrix B2(outputDim, 1);
    B2.data = {0.05f, -0.05f};
    
    CSR W1_csr = denseToCSR(W1_dense);
    CSR W2_csr = denseToCSR(W2_dense);
    
    ScheduleParams sp(32, 32);
    
    DenseMatrix* output_spmm = sparseNNSpmm(&input, &W1_csr, &W2_csr, &B1, &B2, sp);
    DenseMatrix* output_spmv = sparseNNSpmv(&input, &W1_csr, &W2_csr, &B1, &B2, sp);
    
    ASSERT_NE(output_spmm, nullptr);
    ASSERT_NE(output_spmv, nullptr);
    ASSERT_EQ(output_spmm->m, output_spmv->m);
    ASSERT_EQ(output_spmm->n, output_spmv->n);
    
    for (int i = 0; i < batchSize * outputDim; ++i) {
        EXPECT_NEAR(output_spmm->data[i], output_spmv->data[i], 1e-4f)
            << "Mismatch at index " << i;
    }
    
    delete output_spmm;
    delete output_spmv;
}

// Test Sparse NN with high sparsity (90%)
TEST(SparseNNTest, HighSparsityWeights) {
    int batchSize = 1;
    int inputDim = 10;
    int hiddenDim = 10;
    int outputDim = 2;
    
    DenseMatrix input(batchSize, inputDim);
    for (int i = 0; i < inputDim; ++i) {
        input.data.push_back(0.1f * (i + 1));
    }
    
    // 90% sparse W1 (only diagonal)
    DenseMatrix W1_dense(hiddenDim, inputDim);
    W1_dense.data.resize(hiddenDim * inputDim, 0.0f);
    for (int i = 0; i < std::min(hiddenDim, inputDim); ++i) {
        W1_dense.data[i * inputDim + i] = 0.5f;
    }
    
    DenseMatrix B1(hiddenDim, 1);
    B1.data.resize(hiddenDim, 0.0f);
    
    // 90% sparse W2 (only first column)
    DenseMatrix W2_dense(outputDim, hiddenDim);
    W2_dense.data.resize(outputDim * hiddenDim, 0.0f);
    W2_dense.data[0] = 1.0f;  // W2[0,0]
    W2_dense.data[hiddenDim] = 1.0f;  // W2[1,0]
    
    DenseMatrix B2(outputDim, 1);
    B2.data = {0.0f, 0.0f};
    
    CSR W1_csr = denseToCSR(W1_dense);
    CSR W2_csr = denseToCSR(W2_dense);
    
    ScheduleParams sp(32, 32);
    DenseMatrix* output = sparseNNSpmm(&input, &W1_csr, &W2_csr, &B1, &B2, sp);
    
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->m, batchSize);
    ASSERT_EQ(output->n, outputDim);
    
    // Output should be valid sigmoid values
    for (int i = 0; i < batchSize * outputDim; ++i) {
        EXPECT_GE(output->data[i], 0.0f);
        EXPECT_LE(output->data[i], 1.0f);
    }
    
    delete output;
}

// Test Sparse NN with zero weights (edge case)
TEST(SparseNNTest, AllZeroWeights) {
    int batchSize = 1;
    int inputDim = 2;
    int hiddenDim = 2;
    int outputDim = 2;
    
    DenseMatrix input(batchSize, inputDim);
    input.data = {1.0f, 1.0f};
    
    // All zero W1 (after pruning extreme case)
    DenseMatrix W1_dense(hiddenDim, inputDim);
    W1_dense.data = {0.0f, 0.0f,
                     0.0f, 0.0f};
    
    DenseMatrix B1(hiddenDim, 1);
    B1.data = {0.5f, -0.5f};  // Non-zero bias
    
    DenseMatrix W2_dense(outputDim, hiddenDim);
    W2_dense.data = {0.0f, 0.0f,
                     0.0f, 0.0f};
    
    DenseMatrix B2(outputDim, 1);
    B2.data = {0.0f, 0.0f};
    
    CSR W1_csr = denseToCSR(W1_dense);
    CSR W2_csr = denseToCSR(W2_dense);
    
    ScheduleParams sp(32, 32);
    DenseMatrix* output = sparseNNSpmm(&input, &W1_csr, &W2_csr, &B1, &B2, sp);
    
    ASSERT_NE(output, nullptr);
    
    // With zero weights, hidden = tanh(0 + bias) = tanh([0.5, -0.5])
    // output = sigmoid(0 + 0) = sigmoid(0) = 0.5
    EXPECT_NEAR(output->data[0], 0.5f, 0.01f);
    EXPECT_NEAR(output->data[1], 0.5f, 0.01f);
    
    delete output;
}

// Test batch processing with sparse NN
TEST(SparseNNTest, BatchProcessing) {
    int batchSize = 4;
    int inputDim = 3;
    int hiddenDim = 2;
    int outputDim = 2;
    
    DenseMatrix input(batchSize, inputDim);
    input.data = {1.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f,
                  0.0f, 0.0f, 1.0f,
                  1.0f, 1.0f, 1.0f};
    
    // 33% sparse W1
    DenseMatrix W1_dense(hiddenDim, inputDim);
    W1_dense.data = {1.0f, 0.0f, 1.0f,
                     0.0f, 1.0f, 0.0f};
    
    DenseMatrix B1(hiddenDim, 1);
    B1.data = {0.0f, 0.0f};
    
    // 50% sparse W2
    DenseMatrix W2_dense(outputDim, hiddenDim);
    W2_dense.data = {1.0f, 0.0f,
                     0.0f, 1.0f};
    
    DenseMatrix B2(outputDim, 1);
    B2.data = {0.0f, 0.0f};
    
    CSR W1_csr = denseToCSR(W1_dense);
    CSR W2_csr = denseToCSR(W2_dense);
    
    ScheduleParams sp(32, 32);
    DenseMatrix* output = sparseNNSpmm(&input, &W1_csr, &W2_csr, &B1, &B2, sp);
    
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->m, batchSize);
    ASSERT_EQ(output->n, outputDim);
    
    // Verify output is valid sigmoid values
    for (int i = 0; i < batchSize * outputDim; ++i) {
        EXPECT_GE(output->data[i], 0.0f);
        EXPECT_LE(output->data[i], 1.0f);
    }
    
    delete output;
}

} // namespace swiftware::hpp
