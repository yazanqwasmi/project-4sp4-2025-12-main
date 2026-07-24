// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "dense_nn.h"
#include <cmath>
#include <algorithm>
#include <chrono>


namespace swiftware::hpp {

    /* Helper Functions + Activations Functions for Running Dense NN */

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

    // Helper: transpose row-major A(m x n) into B(n x m)
    static std::vector<float> transpose(const float* A, int m, int n) {
        std::vector<float> B(n * m);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                B[j * m + i] = A[i * n + j];
            }
        }
        return B;
    }


    DenseMatrix* dense_nn_gemm(DenseMatrix* InData,
                           DenseMatrix* W1, DenseMatrix* W2,
                           DenseMatrix* B1, DenseMatrix* B2,
                           ScheduleParams Sp)
    {
        const int batchSize = InData->m;
        const int inputDim  = InData->n;
        const int hiddenDim = W1->m;
        const int outputDim = W2->m;

        auto* hidden = new DenseMatrix(batchSize, hiddenDim);
        auto* pred   = new DenseMatrix(batchSize, outputDim);

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

        // ---- Transpose W1: W1 is hiddenDim × inputDim ----
        std::vector<float> W1T = transpose(W1->data.data(), hiddenDim, inputDim);
        // Now W1T is inputDim × hiddenDim

        // Hidden layer: hidden = InData · W1^T + B1
        gemm(batchSize, hiddenDim, inputDim,
            InData->data.data(), W1T.data(), hidden->data.data(), Sp);

        // Apply tanh
        tanh_activation(hidden->data.data(), hidden->data.size());

        // ---- Transpose W2: W2 is outputDim × hiddenDim ----
        std::vector<float> W2T = transpose(W2->data.data(), outputDim, hiddenDim);
        // Now W2T is hiddenDim × outputDim

        // Output layer: pred = hidden · W2^T + B2
        gemm(batchSize, outputDim, hiddenDim,
            hidden->data.data(), W2T.data(), pred->data.data(), Sp);

        // Apply sigmoid 
        sigmoid_activation(pred->data.data(), pred->data.size());

        delete hidden;
        return pred;
    }

    DenseMatrix* dense_nn_gemv(DenseMatrix* InData,
                            DenseMatrix* W1, DenseMatrix* W2,
                            DenseMatrix* B1, DenseMatrix* B2,
                            ScheduleParams Sp)
    {
        const int batchSize  = InData->m;
        const int inputDim   = InData->n;
        const int hiddenDim  = W1->m;   // W1: hiddenDim x inputDim
        const int outputDim  = W2->m;   // W2: outputDim x hiddenDim

        // We return full logits/probabilities: batchSize x outputDim
        auto* pred = new DenseMatrix(batchSize, outputDim);

        // Temporary buffers for one sample
        std::vector<float> h(hiddenDim);
        std::vector<float> z(outputDim);

        const float* A1 = W1->data.data();  // (hiddenDim x inputDim), row-major
        const float* A2 = W2->data.data();  // (outputDim x hiddenDim), row-major

        for (int i = 0; i < batchSize; ++i)
        {
            // Pointer to i-th input sample (length = inputDim)
            const float* x = InData->data.data() + i * inputDim;

            // ---- Hidden layer: h = W1 * x + B1 ----
            // Initialize with bias
            for (int j = 0; j < hiddenDim; ++j) {
                h[j] = B1->data[j];
            }

            // GEMV: h += W1 * x
            gemv(hiddenDim, inputDim, A1, x, h.data(), Sp);

            // Apply tanh
            tanh_activation(h.data(), hiddenDim);

            // ---- Output layer: z = W2 * h + B2 ----
            for (int j = 0; j < outputDim; ++j) {
                z[j] = B2->data[j];
            }

            // GEMV: z += W2 * h
            gemv(outputDim, hiddenDim, A2, h.data(), z.data(), Sp);

            // Apply sigmoid
            sigmoid_activation(z.data(), outputDim);

            // ---- Store this sample's output vector into row i of pred ----
            for (int j = 0; j < outputDim; ++j) {
                pred->data[i * outputDim + j] = z[j];
            }
        }

        return pred;
    }


#ifdef USE_MKL
    void gemmMKL(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp);
    void gemvMKL(int m, int n, const float *A, const float *x, float *y, ScheduleParams Sp);
    
    DenseMatrix *dense_nn_mkl_gemm(DenseMatrix *InData,
                              DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
        int batchSize  = InData->m;
        int inputDim   = InData->n;
        int hiddenDim  = W1->m;
        int outputDim  = W2->m;

        DenseMatrix *pred = new DenseMatrix(batchSize, 1);

        DenseMatrix H(batchSize, hiddenDim);
        for (int i = 0; i < batchSize; ++i)
            for (int j = 0; j < hiddenDim; ++j)
                H.data[i * hiddenDim + j] = B1->data[j];

        std::vector<float> W1T = transpose(W1->data.data(), hiddenDim, inputDim);
        gemmMKL(batchSize, hiddenDim, inputDim,
                InData->data.data(), W1T.data(), H.data.data(), Sp);
        tanh_activation(H.data.data(), batchSize * hiddenDim);

        DenseMatrix Z(batchSize, outputDim);
        for (int i = 0; i < batchSize; ++i)
            for (int j = 0; j < outputDim; ++j)
                Z.data[i * outputDim + j] = B2->data[j];

        std::vector<float> W2T = transpose(W2->data.data(), outputDim, hiddenDim);
        gemmMKL(batchSize, outputDim, hiddenDim,
                H.data.data(), W2T.data(), Z.data.data(), Sp);
        sigmoid_activation(Z.data.data(), batchSize * outputDim);

        for (int i = 0; i < batchSize; ++i) {
            int bestIdx = 0;
            float bestVal = Z.data[i * outputDim + 0];
            for (int j = 1; j < outputDim; ++j) {
                float v = Z.data[i * outputDim + j];
                if (v > bestVal) {
                    bestVal = v;
                    bestIdx = j;
                }
            }
            pred->data[i] = static_cast<float>(bestIdx);
        }
        return pred;
    }
    
    DenseMatrix *dense_nn_mkl_gemv(DenseMatrix *InData,
                                   DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
        int batchSize  = InData->m;
        int inputDim   = InData->n;
        int hiddenDim  = W1->m;
        int outputDim  = W2->m;

        DenseMatrix *pred = new DenseMatrix(batchSize, 1);

        std::vector<float> h(hiddenDim);
        std::vector<float> z(outputDim);

        const float* A1 = W1->data.data();
        const float* A2 = W2->data.data();

        for (int i = 0; i < batchSize; ++i) {
            const float* x = InData->data.data() + i * inputDim;

            for (int j = 0; j < hiddenDim; ++j)
                h[j] = B1->data[j];
            gemvMKL(hiddenDim, inputDim, A1, x, h.data(), Sp);
            tanh_activation(h.data(), hiddenDim);

            for (int j = 0; j < outputDim; ++j)
                z[j] = B2->data[j];
            gemvMKL(outputDim, hiddenDim, A2, h.data(), z.data(), Sp);
            sigmoid_activation(z.data(), outputDim);

            int bestIdx = 0;
            float bestVal = z[0];
            for (int j = 1; j < outputDim; ++j) {
                if (z[j] > bestVal) {
                    bestVal = z[j];
                    bestIdx = j;
                }
            }
            pred->data[i] = static_cast<float>(bestIdx);
        }
        
        return pred;
    }
#endif

}