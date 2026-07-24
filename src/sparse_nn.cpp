// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "sparse_nn.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace swiftware::hpp {

    /* Helper Functions + Activation Functions for Sparse NN */
    
    static inline void tanh_activation(float* data, int size) {
        #pragma omp simd
        for (int i = 0; i < size; ++i) {
            data[i] = std::tanh(data[i]);
        }
    }

    static inline void sigmoid_activation(float* data, int size) {
        #pragma omp simd
        for (int i = 0; i < size; ++i) {
            data[i] = 1.0f / (1.0f + std::exp(-data[i]));
        }
    }


    DenseMatrix *sparseNNSpmm(DenseMatrix *InData,
                            CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
        // Sparse Neural Network using SpMM for batch processing
        // W1: (hiddenDim x inputDim) sparse CSR
        // W2: (outputDim x hiddenDim) sparse CSR
        // InData: (batchSize x inputDim)
        // B1: (hiddenDim x 1) biases
        // B2: (outputDim x 1) biases
        
        const int batchSize = InData->m;
        const int inputDim = InData->n;
        const int hiddenDim = W1->m;
        const int outputDim = W2->m;
        
        // Allocate matrices for intermediate and output results
        auto* hidden = new DenseMatrix(batchSize, hiddenDim);
        auto* pred = new DenseMatrix(batchSize, outputDim);
        
        // ---- Broadcast B1 into all rows of hidden ----
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < hiddenDim; ++j) {
                hidden->data[i * hiddenDim + j] = B1->data[j];
            }
        }
        
        // ---- Broadcast B2 into all rows of pred ----
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < outputDim; ++j) {
                pred->data[i * outputDim + j] = B2->data[j];
            }
        }
        
        // Hidden layer: hidden = W1 * InData^T + B1
        // But InData is (batchSize x inputDim) and we need (batchSize x hiddenDim)
        // So we compute: hidden^T = W1 * InData^T
        // Or equivalently: hidden = InData * W1^T (but W1 is sparse, so we use SpMM differently)
        // Actually: hidden = (W1 * InData^T)^T = InData * W1^T
        // Since W1 is CSR format for (hiddenDim x inputDim), we need to handle this carefully
        
        // The standard way: treat each row of InData as a column vector and use SpMM
        // For batch processing, we want: H = X * W1^T where X is (batch x input), W1 is (hidden x input)
        // This is equivalent to H^T = W1 * X^T
        // We'll transpose InData conceptually and use SpMM
        
        // Actually, let's use the SpMM as: C = A * B where A is sparse CSR
        // We need: hidden (batch x hidden) = InData (batch x input) * W1^T (input x hidden)
        // But W1 is (hidden x input) in CSR, so W1^T would be (input x hidden)
        // This requires transpose of CSR which is complex, so let's compute row by row instead
        
        // Alternative: Compute H^T = W1 * X^T, then transpose result
        // H^T is (hidden x batch), W1 is (hidden x input), X^T is (input x batch)
        // Let's create X^T:
        std::vector<float> InDataT(inputDim * batchSize);
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < inputDim; ++j) {
                InDataT[j * batchSize + i] = InData->data[i * inputDim + j];
            }
        }
        
        // Compute H^T = W1 * X^T using SpMM
        // H^T will be (hiddenDim x batchSize)
        std::vector<float> hiddenT(hiddenDim * batchSize, 0.0f);
        spmmCSR(hiddenDim, batchSize, inputDim,
                W1->row_ptr.data(), W1->col_idx.data(), W1->values.data(),
                InDataT.data(), hiddenT.data(), Sp);
        
        // Transpose H^T back to hidden and add bias
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < hiddenDim; ++j) {
                hidden->data[i * hiddenDim + j] += hiddenT[j * batchSize + i];
            }
        }
        
        // Apply tanh activation
        tanh_activation(hidden->data.data(), hidden->data.size());
        
        // Output layer: pred = hidden * W2^T + B2
        // Similar process: compute pred^T = W2 * hidden^T
        std::vector<float> hiddenTForW2(hiddenDim * batchSize);
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < hiddenDim; ++j) {
                hiddenTForW2[j * batchSize + i] = hidden->data[i * hiddenDim + j];
            }
        }
        
        // Compute pred^T = W2 * hidden^T
        std::vector<float> predT(outputDim * batchSize, 0.0f);
        spmmCSR(outputDim, batchSize, hiddenDim,
                W2->row_ptr.data(), W2->col_idx.data(), W2->values.data(),
                hiddenTForW2.data(), predT.data(), Sp);
        
        // Transpose pred^T back to pred and add bias
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < outputDim; ++j) {
                pred->data[i * outputDim + j] += predT[j * batchSize + i];
            }
        }
        
        // Apply sigmoid activation
        sigmoid_activation(pred->data.data(), pred->data.size());
        
        delete hidden;
        return pred;
    }

    DenseMatrix *sparseNNSpmv(DenseMatrix *InData,
                              CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp){
        // Sparse Neural Network using SpMV (process one sample at a time)
        const int batchSize = InData->m;
        const int inputDim = InData->n;
        const int hiddenDim = W1->m;
        const int outputDim = W2->m;
        
        // Return full logits/probabilities: batchSize x outputDim
        auto* pred = new DenseMatrix(batchSize, outputDim);
        
        // Temporary buffers for one sample
        std::vector<float> h(hiddenDim);
        std::vector<float> z(outputDim);
        
        for (int i = 0; i < batchSize; ++i) {
            // Pointer to i-th input sample
            const float* x = InData->data.data() + i * inputDim;
            
            // ---- Hidden layer: h = W1 * x + B1 ----
            // Initialize with bias
            for (int j = 0; j < hiddenDim; ++j) {
                h[j] = B1->data[j];
            }
            
            // SpMV: h += W1 * x
            spmvCSR(hiddenDim, inputDim,
                    W1->row_ptr.data(), W1->col_idx.data(), W1->values.data(),
                    x, h.data(), Sp);
            
            // Apply tanh
            tanh_activation(h.data(), hiddenDim);
            
            // ---- Output layer: z = W2 * h + B2 ----
            for (int j = 0; j < outputDim; ++j) {
                z[j] = B2->data[j];
            }
            
            // SpMV: z += W2 * h
            spmvCSR(outputDim, hiddenDim,
                    W2->row_ptr.data(), W2->col_idx.data(), W2->values.data(),
                    h.data(), z.data(), Sp);
            
            // Apply sigmoid
            sigmoid_activation(z.data(), outputDim);
            
            // ---- Store this sample's output vector into row i of pred ----
            for (int j = 0; j < outputDim; ++j) {
                pred->data[i * outputDim + j] = z[j];
            }
        }
        
        return pred;
    }


}