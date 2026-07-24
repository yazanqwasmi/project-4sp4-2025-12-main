// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gpu_dense_nn.cuh"
#include "kernels.cuh"
#include "gpu_utils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(x) swiftware::hpp::cuda_check((x), __FILE__, __LINE__)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        } \
    } while(0)

namespace swiftware::hpp {

float* gpu_alloc_and_copy(const float* hostData, size_t size) {
    float* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePtr, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(devicePtr, hostData, size * sizeof(float), cudaMemcpyHostToDevice));
    return devicePtr;
}

float* gpu_alloc(size_t size) {
    float* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePtr, size * sizeof(float)));
    return devicePtr;
}

void gpu_copy_to_host(float* hostDst, const float* deviceSrc, size_t size) {
    CUDA_CHECK(cudaMemcpy(hostDst, deviceSrc, size * sizeof(float), cudaMemcpyDeviceToHost));
}

void gpu_free(float* devicePtr) {
    if (devicePtr != nullptr) {
        CUDA_CHECK(cudaFree(devicePtr));
    }
}

static void gpu_transpose(const float* d_src, float* d_dst, int m, int n) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    Transpose<<<gridDim, blockDim>>>(d_src, d_dst, m, n);
}

static void gpu_broadcast_bias(float* d_C, const float* d_bias, int m, int n) {
    int total = m * n;
    int blockSize = BLOCK_SIZE_1D;
    int gridSize = (total + blockSize - 1) / blockSize;
    BroadcastBias<<<gridSize, blockSize>>>(d_C, d_bias, m, n);
}

static void gpu_tanh(float* d_data, int size) {
    int blockSize = BLOCK_SIZE_1D;
    int gridSize = (size + blockSize - 1) / blockSize;
    TanhActivation<<<gridSize, blockSize>>>(d_data, size);
}

static void gpu_sigmoid(float* d_data, int size) {
    int blockSize = BLOCK_SIZE_1D;
    int gridSize = (size + blockSize - 1) / blockSize;
    SigmoidActivation<<<gridSize, blockSize>>>(d_data, size);
}

static void gpu_gemm_advanced(const float* d_A, const float* d_B, float* d_C, int m, int n, int k) {
    if (n <= 16) {
        int warpsPerBlock = 8;
        int blockSize = warpsPerBlock * 32;
        int numBlocks = (m + warpsPerBlock - 1) / warpsPerBlock;
        MM_SmallN<<<numBlocks, blockSize>>>(d_A, d_B, d_C, m, n, k);
    }

    else {
        dim3 blockDim(256);
        dim3 gridDim((n + BN - 1) / BN, (m + BM - 1) / BM);
        MM<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
    }
}

DenseMatrix* gpu_dense_nn_gemm(DenseMatrix* InData, 
                               DenseMatrix* W1, DenseMatrix* W2, 
                               DenseMatrix* B1, DenseMatrix* B2, 
                               ScheduleParams Sp) 
{
    const int batchSize = InData->m;
    const int inputDim  = InData->n;
    const int hiddenDim = W1->m;
    const int outputDim = W2->m;

    auto* pred = new DenseMatrix(batchSize, outputDim);

    float* d_Input = gpu_alloc_and_copy(InData->data.data(), batchSize * inputDim);
    
    float* d_W1 = gpu_alloc_and_copy(W1->data.data(), hiddenDim * inputDim);
    float* d_W1T = gpu_alloc(inputDim * hiddenDim);
    
    float* d_W2 = gpu_alloc_and_copy(W2->data.data(), outputDim * hiddenDim);
    float* d_W2T = gpu_alloc(hiddenDim * outputDim);
    
    float* d_B1 = gpu_alloc_and_copy(B1->data.data(), hiddenDim);
    float* d_B2 = gpu_alloc_and_copy(B2->data.data(), outputDim);
    
    float* d_Hidden = gpu_alloc(batchSize * hiddenDim);
    
    float* d_Output = gpu_alloc(batchSize * outputDim);

    gpu_transpose(d_W1, d_W1T, hiddenDim, inputDim);
    gpu_transpose(d_W2, d_W2T, outputDim, hiddenDim);

    gpu_broadcast_bias(d_Hidden, d_B1, batchSize, hiddenDim);
    
    gpu_gemm_advanced(d_Input, d_W1T, d_Hidden, batchSize, hiddenDim, inputDim);
    
    gpu_tanh(d_Hidden, batchSize * hiddenDim);

    gpu_broadcast_bias(d_Output, d_B2, batchSize, outputDim);
    
    gpu_gemm_advanced(d_Hidden, d_W2T, d_Output, batchSize, outputDim, hiddenDim);
    
    gpu_sigmoid(d_Output, batchSize * outputDim);

    CUDA_CHECK(cudaDeviceSynchronize());
    
    gpu_copy_to_host(pred->data.data(), d_Output, batchSize * outputDim);

    gpu_free(d_Input);
    gpu_free(d_W1);
    gpu_free(d_W1T);
    gpu_free(d_W2);
    gpu_free(d_W2T);
    gpu_free(d_B1);
    gpu_free(d_B2);
    gpu_free(d_Hidden);
    gpu_free(d_Output);

    return pred;
}

static void gpu_gemv_batched(const float* d_W, const float* d_X, float* d_Y, 
                              int batchSize, int m, int n) {
    dim3 gridDim(m, batchSize);
    int blockSize = 128;  // threads per block for reduction
    size_t sharedMem = blockSize * sizeof(float);
    MV_Batched<<<gridDim, blockSize, sharedMem>>>(d_W, d_X, d_Y, batchSize, m, n);
}

static void gpu_copy_bias_batched(float* d_Y, const float* d_bias, int batchSize, int m) {
    int total = batchSize * m;
    int blockSize = BLOCK_SIZE_1D;
    int gridSize = (total + blockSize - 1) / blockSize;
    CopyBiasBatched<<<gridSize, blockSize>>>(d_Y, d_bias, batchSize, m);
}

DenseMatrix* gpu_dense_nn_gemv(DenseMatrix* InData, 
                               DenseMatrix* W1, DenseMatrix* W2, 
                               DenseMatrix* B1, DenseMatrix* B2, 
                               ScheduleParams Sp) 
{
    const int batchSize = InData->m;
    const int inputDim  = InData->n;
    const int hiddenDim = W1->m;
    const int outputDim = W2->m;

    auto* pred = new DenseMatrix(batchSize, outputDim);

    float* d_Input = gpu_alloc_and_copy(InData->data.data(), batchSize * inputDim);
    
    float* d_W1 = gpu_alloc_and_copy(W1->data.data(), hiddenDim * inputDim);
    
    float* d_W2 = gpu_alloc_and_copy(W2->data.data(), outputDim * hiddenDim);
    
    float* d_B1 = gpu_alloc_and_copy(B1->data.data(), hiddenDim);
    float* d_B2 = gpu_alloc_and_copy(B2->data.data(), outputDim);
    
    float* d_Hidden = gpu_alloc(batchSize * hiddenDim);
    
    float* d_Output = gpu_alloc(batchSize * outputDim);

    gpu_copy_bias_batched(d_Hidden, d_B1, batchSize, hiddenDim);
    
    gpu_gemv_batched(d_W1, d_Input, d_Hidden, batchSize, hiddenDim, inputDim);
    
    gpu_tanh(d_Hidden, batchSize * hiddenDim);

    gpu_copy_bias_batched(d_Output, d_B2, batchSize, outputDim);
    
    gpu_gemv_batched(d_W2, d_Hidden, d_Output, batchSize, outputDim, hiddenDim);
    
    gpu_sigmoid(d_Output, batchSize * outputDim);

    CUDA_CHECK(cudaDeviceSynchronize());
    gpu_copy_to_host(pred->data.data(), d_Output, batchSize * outputDim);

    gpu_free(d_Input);
    gpu_free(d_W1);
    gpu_free(d_W2);
    gpu_free(d_B1);
    gpu_free(d_B2);
    gpu_free(d_Hidden);
    gpu_free(d_Output);

    return pred;
}

DenseMatrix* gpu_dense_nn_cublas(DenseMatrix* InData, 
                                 DenseMatrix* W1, DenseMatrix* W2, 
                                 DenseMatrix* B1, DenseMatrix* B2, 
                                 ScheduleParams Sp) 
{
    const int batchSize = InData->m;
    const int inputDim  = InData->n;
    const int hiddenDim = W1->m;
    const int outputDim = W2->m;

    auto* pred = new DenseMatrix(batchSize, outputDim);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 1.0f;  

    float* d_Input = gpu_alloc_and_copy(InData->data.data(), batchSize * inputDim);
    
    float* d_W1 = gpu_alloc_and_copy(W1->data.data(), hiddenDim * inputDim);
    
    float* d_W2 = gpu_alloc_and_copy(W2->data.data(), outputDim * hiddenDim);
    
    float* d_B1 = gpu_alloc_and_copy(B1->data.data(), hiddenDim);
    float* d_B2 = gpu_alloc_and_copy(B2->data.data(), outputDim);
    
    float* d_Hidden = gpu_alloc(batchSize * hiddenDim);
    
    float* d_Output = gpu_alloc(batchSize * outputDim);

    gpu_broadcast_bias(d_Hidden, d_B1, batchSize, hiddenDim);
    
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_T,      
                             CUBLAS_OP_N,      
                             hiddenDim,        
                             batchSize,        
                             inputDim,         
                             &alpha,
                             d_W1, inputDim,   
                             d_Input, inputDim,
                             &beta,
                             d_Hidden, hiddenDim)); 
    
    gpu_tanh(d_Hidden, batchSize * hiddenDim);

    gpu_broadcast_bias(d_Output, d_B2, batchSize, outputDim);
    
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_T,       
                             CUBLAS_OP_N,       
                             outputDim,         
                             batchSize,         
                             hiddenDim,         
                             &alpha,
                             d_W2, hiddenDim,   
                             d_Hidden, hiddenDim,
                             &beta,
                             d_Output, outputDim)); 
    
    gpu_sigmoid(d_Output, batchSize * outputDim);

    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy output to host
    gpu_copy_to_host(pred->data.data(), d_Output, batchSize * outputDim);

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    gpu_free(d_Input);
    gpu_free(d_W1);
    gpu_free(d_W2);
    gpu_free(d_B1);
    gpu_free(d_B2);
    gpu_free(d_Hidden);
    gpu_free(d_Output);

    return pred;
}

}  // namespace swiftware::hpp
