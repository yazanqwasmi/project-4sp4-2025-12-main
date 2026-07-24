// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "dense_nn.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

namespace swiftware::hpp {

// Helper to find argmax
int argmax(const float* data, int size) {
    return std::distance(data, std::max_element(data, data + size));
}

// Test Dense NN GEMM with simple known weights
TEST(DenseNNTest, GemmBasicForwardPass) {
    // Simple 2-layer NN: input(2) -> hidden(2) -> output(2)
    // Using small weights for easy verification
    
    int batchSize = 1;
    int inputDim = 2;
    int hiddenDim = 2;
    int outputDim = 2;
    
    // Input: [1, 1]
    DenseMatrix input(batchSize, inputDim);
    input.data = {1.0f, 1.0f};
    
    // W1 (hidden x input): identity matrix
    DenseMatrix W1(hiddenDim, inputDim);
    W1.data = {1.0f, 0.0f,
               0.0f, 1.0f};
    
    // B1: zeros
    DenseMatrix B1(hiddenDim, 1);
    B1.data = {0.0f, 0.0f};
    
    // W2 (output x hidden): identity matrix
    DenseMatrix W2(outputDim, hiddenDim);
    W2.data = {1.0f, 0.0f,
               0.0f, 1.0f};
    
    // B2: zeros
    DenseMatrix B2(outputDim, 1);
    B2.data = {0.0f, 0.0f};
    
    ScheduleParams sp(32, 32);
    DenseMatrix* output = dense_nn_gemm(&input, &W1, &W2, &B1, &B2, sp);
    
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->m, batchSize);
    ASSERT_EQ(output->n, outputDim);
    
    // With identity weights and zero biases:
    // hidden = tanh(input * I + 0) = tanh([1,1]) ≈ [0.7616, 0.7616]
    // output = sigmoid(hidden * I + 0) = sigmoid([0.7616, 0.7616]) ≈ [0.6818, 0.6818]
    float expected = 1.0f / (1.0f + std::exp(-std::tanh(1.0f)));
    EXPECT_NEAR(output->data[0], expected, 0.01f);
    EXPECT_NEAR(output->data[1], expected, 0.01f);
    
    delete output;
}

// Test Dense NN GEMV with simple known weights
TEST(DenseNNTest, GemvBasicForwardPass) {
    int batchSize = 1;
    int inputDim = 2;
    int hiddenDim = 2;
    int outputDim = 2;
    
    DenseMatrix input(batchSize, inputDim);
    input.data = {1.0f, 1.0f};
    
    DenseMatrix W1(hiddenDim, inputDim);
    W1.data = {1.0f, 0.0f,
               0.0f, 1.0f};
    
    DenseMatrix B1(hiddenDim, 1);
    B1.data = {0.0f, 0.0f};
    
    DenseMatrix W2(outputDim, hiddenDim);
    W2.data = {1.0f, 0.0f,
               0.0f, 1.0f};
    
    DenseMatrix B2(outputDim, 1);
    B2.data = {0.0f, 0.0f};
    
    ScheduleParams sp(32, 32);
    DenseMatrix* output = dense_nn_gemv(&input, &W1, &W2, &B1, &B2, sp);
    
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->m, batchSize);
    ASSERT_EQ(output->n, outputDim);
    
    float expected = 1.0f / (1.0f + std::exp(-std::tanh(1.0f)));
    EXPECT_NEAR(output->data[0], expected, 0.01f);
    EXPECT_NEAR(output->data[1], expected, 0.01f);
    
    delete output;
}

// Test GEMM and GEMV produce same results
TEST(DenseNNTest, GemmGemvConsistency) {
    int batchSize = 2;
    int inputDim = 4;
    int hiddenDim = 3;
    int outputDim = 2;
    
    // Random-ish input
    DenseMatrix input(batchSize, inputDim);
    input.data = {0.5f, -0.3f, 0.8f, -0.1f,
                  0.2f, 0.7f, -0.4f, 0.9f};
    
    // Random-ish weights
    DenseMatrix W1(hiddenDim, inputDim);
    W1.data = {0.1f, 0.2f, -0.1f, 0.3f,
               -0.2f, 0.1f, 0.4f, -0.3f,
               0.3f, -0.4f, 0.2f, 0.1f};
    
    DenseMatrix B1(hiddenDim, 1);
    B1.data = {0.1f, -0.1f, 0.05f};
    
    DenseMatrix W2(outputDim, hiddenDim);
    W2.data = {0.2f, -0.3f, 0.1f,
               -0.1f, 0.4f, -0.2f};
    
    DenseMatrix B2(outputDim, 1);
    B2.data = {0.05f, -0.05f};
    
    ScheduleParams sp(32, 32);
    
    DenseMatrix* output_gemm = dense_nn_gemm(&input, &W1, &W2, &B1, &B2, sp);
    DenseMatrix* output_gemv = dense_nn_gemv(&input, &W1, &W2, &B1, &B2, sp);
    
    ASSERT_NE(output_gemm, nullptr);
    ASSERT_NE(output_gemv, nullptr);
    ASSERT_EQ(output_gemm->m, output_gemv->m);
    ASSERT_EQ(output_gemm->n, output_gemv->n);
    
    // Both should produce the same results
    for (int i = 0; i < batchSize * outputDim; ++i) {
        EXPECT_NEAR(output_gemm->data[i], output_gemv->data[i], 1e-4f) 
            << "Mismatch at index " << i;
    }
    
    delete output_gemm;
    delete output_gemv;
}

// Test with batch processing
TEST(DenseNNTest, BatchProcessing) {
    int batchSize = 4;
    int inputDim = 3;
    int hiddenDim = 2;
    int outputDim = 2;
    
    DenseMatrix input(batchSize, inputDim);
    input.data = {1.0f, 0.0f, 0.0f,   // Sample 1
                  0.0f, 1.0f, 0.0f,   // Sample 2
                  0.0f, 0.0f, 1.0f,   // Sample 3
                  1.0f, 1.0f, 1.0f};  // Sample 4
    
    DenseMatrix W1(hiddenDim, inputDim);
    W1.data = {1.0f, 1.0f, 1.0f,
               -1.0f, -1.0f, -1.0f};
    
    DenseMatrix B1(hiddenDim, 1);
    B1.data = {0.0f, 0.0f};
    
    DenseMatrix W2(outputDim, hiddenDim);
    W2.data = {1.0f, 0.0f,
               0.0f, 1.0f};
    
    DenseMatrix B2(outputDim, 1);
    B2.data = {0.0f, 0.0f};
    
    ScheduleParams sp(32, 32);
    DenseMatrix* output = dense_nn_gemm(&input, &W1, &W2, &B1, &B2, sp);
    
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->m, batchSize);
    ASSERT_EQ(output->n, outputDim);
    
    // Verify output is valid (between 0 and 1 for sigmoid)
    for (int i = 0; i < batchSize * outputDim; ++i) {
        EXPECT_GE(output->data[i], 0.0f);
        EXPECT_LE(output->data[i], 1.0f);
    }
    
    delete output;
}

// Test activation functions are applied correctly
TEST(DenseNNTest, ActivationFunctions) {
    int batchSize = 1;
    int inputDim = 1;
    int hiddenDim = 1;
    int outputDim = 1;
    
    // Large positive input to test saturation
    DenseMatrix input(batchSize, inputDim);
    input.data = {10.0f};
    
    DenseMatrix W1(hiddenDim, inputDim);
    W1.data = {1.0f};
    
    DenseMatrix B1(hiddenDim, 1);
    B1.data = {0.0f};
    
    DenseMatrix W2(outputDim, hiddenDim);
    W2.data = {1.0f};
    
    DenseMatrix B2(outputDim, 1);
    B2.data = {0.0f};
    
    ScheduleParams sp(32, 32);
    DenseMatrix* output = dense_nn_gemm(&input, &W1, &W2, &B1, &B2, sp);
    
    ASSERT_NE(output, nullptr);
    
    // tanh(10) ≈ 1, sigmoid(1) ≈ 0.731
    float hidden = std::tanh(10.0f);  // ≈ 0.99999
    float expected = 1.0f / (1.0f + std::exp(-hidden));  // ≈ 0.731
    
    EXPECT_NEAR(output->data[0], expected, 0.01f);
    
    delete output;
}

} // namespace swiftware::hpp
