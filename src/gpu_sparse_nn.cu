// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gpu_sparse_nn.cuh"
#include "gpu_utils.h"
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(x) swiftware::hpp::cuda_check((x), __FILE__, __LINE__)

namespace swiftware::hpp {


constexpr int SPARSE_BLOCK_SIZE = 256;

// Tanh activation
static __global__ void SparseTanh(float* __restrict__ data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanhf(data[idx]);
    }
}

// Sigmoid activation
static __global__ void SparseSigmoid(float* __restrict__ data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

// Add bias
static __global__ void SparseAddBias(float* __restrict__ C, const float* __restrict__ bias, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        int col = idx % n;
        C[idx] += bias[col];
    }
}

// Copy bias
static __global__ void SparseCopyBias(float* __restrict__ y, const float* __restrict__ bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = bias[idx];
    }
}

// Fused DenseSpMM_T with bias and tanh: C = tanh(A * B^T + bias)
// OPTIMIZED: Warp-cooperative processing for better load balancing
static __global__ void DenseSpMM_T_Bias_Tanh_Optimized(const float* __restrict__ A,
                                                         const int* __restrict__ B_row_ptr,
                                                         const int* __restrict__ B_col_idx,
                                                         const float* __restrict__ B_values,
                                                         const float* __restrict__ bias,
                                                         float* __restrict__ C,
                                                         int m, int n, int k)
{
    extern __shared__ float shared_input[];
    
    const int sample = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = num_threads / 32;
    
    if (sample >= m) return;
    
    const float* A_row = A + sample * k;
    float* C_row = C + sample * n;
    
    // Cooperative load of input row
    for (int i = tid; i < k; i += num_threads) {
        shared_input[i] = A_row[i];
    }
    __syncthreads();
    
    // Each warp processes one output neuron cooperatively
    for (int j = warp_id; j < n; j += num_warps) {
        const float b = bias[j];
        const int start = B_row_ptr[j];
        const int end = B_row_ptr[j + 1];
        
        float partial_sum = 0.0f;
        
        // Warp-parallel accumulation over sparse elements
        for (int idx = start + lane_id; idx < end; idx += 32) {
            partial_sum += shared_input[B_col_idx[idx]] * B_values[idx];
        }
        
        // Warp reduction using shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        
        // Lane 0 writes final result
        if (lane_id == 0) {
            C_row[j] = tanhf(partial_sum + b);
        }
    }
}

// Fused DenseSpMM_T with bias and sigmoid: C = sigmoid(A * B^T + bias)
// OPTIMIZED: Warp-cooperative processing for better load balancing
static __global__ void DenseSpMM_T_Bias_Sigmoid_Optimized(const float* __restrict__ A,
                                                          const int* __restrict__ B_row_ptr,
                                                          const int* __restrict__ B_col_idx,
                                                          const float* __restrict__ B_values,
                                                          const float* __restrict__ bias,
                                                          float* __restrict__ C,
                                                          int m, int n, int k)
{
    extern __shared__ float shared_input[];
    
    const int sample = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = num_threads / 32;
    
    if (sample >= m) return;
    
    const float* A_row = A + sample * k;
    float* C_row = C + sample * n;
    
    // Cooperative load of input row
    for (int i = tid; i < k; i += num_threads) {
        shared_input[i] = A_row[i];
    }
    __syncthreads();
    
    // Each warp processes one output neuron cooperatively
    for (int j = warp_id; j < n; j += num_warps) {
        const float b = bias[j];
        const int start = B_row_ptr[j];
        const int end = B_row_ptr[j + 1];
        
        float partial_sum = 0.0f;
        
        // Warp-parallel accumulation over sparse elements
        for (int idx = start + lane_id; idx < end; idx += 32) {
            partial_sum += shared_input[B_col_idx[idx]] * B_values[idx];
        }
        
        // Warp reduction using shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        
        // Lane 0 writes final result
        if (lane_id == 0) {
            C_row[j] = 1.0f / (1.0f + expf(-(partial_sum + b)));
        }
    }
}

static __global__ void SparseSpMV_Warp(const int* __restrict__ row_ptr, 
                                        const int* __restrict__ col_idx, 
                                        const float* __restrict__ values, 
                                        const float* __restrict__ x, 
                                        float* __restrict__ y,
                                        int m, int n) 
{
    extern __shared__ float sdata[];
    
    int row = blockIdx.x;
    if (row >= m) return;
    
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    int nnz_row = end - start;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < nnz_row; i += blockDim.x) {
        int idx = start + i;
        sum += values[idx] * x[col_idx[idx]];
    }
    
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        y[row] += sdata[0];
    }
}

static float* gpu_alloc_and_copy_float(const float* hostData, size_t size) {
    float* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePtr, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(devicePtr, hostData, size * sizeof(float), cudaMemcpyHostToDevice));
    return devicePtr;
}

static int* gpu_alloc_and_copy_int(const int* hostData, size_t size) {
    int* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePtr, size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(devicePtr, hostData, size * sizeof(int), cudaMemcpyHostToDevice));
    return devicePtr;
}

static float* gpu_alloc_float(size_t size) {
    float* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePtr, size * sizeof(float)));
    return devicePtr;
}

static void gpu_copy_to_host_float(float* hostDst, const float* deviceSrc, size_t size) {
    CUDA_CHECK(cudaMemcpy(hostDst, deviceSrc, size * sizeof(float), cudaMemcpyDeviceToHost));
}

static void gpu_free_float(float* devicePtr) {
    if (devicePtr) CUDA_CHECK(cudaFree(devicePtr));
}

static void gpu_free_int(int* devicePtr) {
    if (devicePtr) CUDA_CHECK(cudaFree(devicePtr));
}


DenseMatrix* gpu_sparse_nn_spmm(DenseMatrix* InData, 
                                CSR* W1, CSR* W2, 
                                DenseMatrix* B1, DenseMatrix* B2, 
                                ScheduleParams Sp) 
{
    const int batchSize = InData->m;
    const int inputDim  = InData->n;      
    const int hiddenDim = W1->m;          
    const int outputDim = W2->m;          

    // Allocate output matrix on host
    auto* pred = new DenseMatrix(batchSize, outputDim);

    float* d_Input = gpu_alloc_and_copy_float(InData->data.data(), batchSize * inputDim);
    
    int* d_W1_row_ptr = gpu_alloc_and_copy_int(W1->row_ptr.data(), W1->m + 1);
    int* d_W1_col_idx = gpu_alloc_and_copy_int(W1->col_idx.data(), W1->nnz);
    float* d_W1_values = gpu_alloc_and_copy_float(W1->values.data(), W1->nnz);
    
    int* d_W2_row_ptr = gpu_alloc_and_copy_int(W2->row_ptr.data(), W2->m + 1);
    int* d_W2_col_idx = gpu_alloc_and_copy_int(W2->col_idx.data(), W2->nnz);
    float* d_W2_values = gpu_alloc_and_copy_float(W2->values.data(), W2->nnz);
    
    // Biases
    float* d_B1 = gpu_alloc_and_copy_float(B1->data.data(), hiddenDim);
    float* d_B2 = gpu_alloc_and_copy_float(B2->data.data(), outputDim);
    
    // Hidden layer: batchSize x hiddenDim
    float* d_Hidden = gpu_alloc_float(batchSize * hiddenDim);

    // Output layer: batchSize x outputDim
    float* d_Output = gpu_alloc_float(batchSize * outputDim);

    // OPTIMIZED: Warp-cooperative approach - each warp handles one neuron
    // Hidden layer: 128 neurons -> use 128*32 = 4096 threads ideally, but cap at 1024
    // With 512 threads (16 warps), each warp processes 128/16 = 8 neurons in loop
    dim3 blockDim1(512);
    dim3 gridDim1(batchSize);
    size_t sharedMemSize1 = inputDim * sizeof(float);
    DenseSpMM_T_Bias_Tanh_Optimized<<<gridDim1, blockDim1, sharedMemSize1>>>(d_Input,
                                                                               d_W1_row_ptr, d_W1_col_idx, d_W1_values,
                                                                               d_B1, d_Hidden,
                                                                               batchSize, hiddenDim, inputDim);

    // Output layer: 10 neurons -> use 10*32 = 320 threads (10 warps, one per neuron)
    dim3 blockDim2(320);
    dim3 gridDim2(batchSize);
    size_t sharedMemSize2 = hiddenDim * sizeof(float);
    DenseSpMM_T_Bias_Sigmoid_Optimized<<<gridDim2, blockDim2, sharedMemSize2>>>(d_Hidden,
                                                                                  d_W2_row_ptr, d_W2_col_idx, d_W2_values,
                                                                                  d_B2, d_Output,
                                                                                  batchSize, outputDim, hiddenDim);

    
    CUDA_CHECK(cudaDeviceSynchronize());
    gpu_copy_to_host_float(pred->data.data(), d_Output, batchSize * outputDim);

    // Free GPU memory
    gpu_free_float(d_Input);
    gpu_free_int(d_W1_row_ptr);
    gpu_free_int(d_W1_col_idx);
    gpu_free_float(d_W1_values);
    gpu_free_int(d_W2_row_ptr);
    gpu_free_int(d_W2_col_idx);
    gpu_free_float(d_W2_values);
    gpu_free_float(d_B1);
    gpu_free_float(d_B2);
    gpu_free_float(d_Hidden);
    gpu_free_float(d_Output);

    return pred;
}

DenseMatrix* gpu_sparse_nn_spmv(DenseMatrix* InData, 
                                CSR* W1, CSR* W2, 
                                DenseMatrix* B1, DenseMatrix* B2, 
                                ScheduleParams Sp) 
{
    const int batchSize = InData->m;
    const int inputDim  = InData->n;
    const int hiddenDim = W1->m;
    const int outputDim = W2->m;

    auto* pred = new DenseMatrix(batchSize, outputDim);

    float* d_Input = gpu_alloc_and_copy_float(InData->data.data(), batchSize * inputDim);
    
    int* d_W1_row_ptr = gpu_alloc_and_copy_int(W1->row_ptr.data(), W1->m + 1);
    int* d_W1_col_idx = gpu_alloc_and_copy_int(W1->col_idx.data(), W1->nnz);
    float* d_W1_values = gpu_alloc_and_copy_float(W1->values.data(), W1->nnz);
    
    int* d_W2_row_ptr = gpu_alloc_and_copy_int(W2->row_ptr.data(), W2->m + 1);
    int* d_W2_col_idx = gpu_alloc_and_copy_int(W2->col_idx.data(), W2->nnz);
    float* d_W2_values = gpu_alloc_and_copy_float(W2->values.data(), W2->nnz);
    
    float* d_B1 = gpu_alloc_and_copy_float(B1->data.data(), hiddenDim);
    float* d_B2 = gpu_alloc_and_copy_float(B2->data.data(), outputDim);
    
    float* d_Hidden = gpu_alloc_float(batchSize * hiddenDim);
    float* d_Output = gpu_alloc_float(batchSize * outputDim);

    for (int i = 0; i < batchSize; ++i) {
        float* d_x = d_Input + i * inputDim;
        float* d_h = d_Hidden + i * hiddenDim;
        
        int gridSize = (hiddenDim + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE;
        SparseCopyBias<<<gridSize, SPARSE_BLOCK_SIZE>>>(d_h, d_B1, hiddenDim);
        
        size_t sharedMem = 256 * sizeof(float);
        SparseSpMV_Warp<<<hiddenDim, 256, sharedMem>>>(d_W1_row_ptr, d_W1_col_idx, d_W1_values, 
                                                       d_x, d_h, hiddenDim, inputDim);
    }
    
    {
        int total = batchSize * hiddenDim;
        int gridSize = (total + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE;
        SparseTanh<<<gridSize, SPARSE_BLOCK_SIZE>>>(d_Hidden, total);
    }
    
    for (int i = 0; i < batchSize; ++i) {
        float* d_h = d_Hidden + i * hiddenDim;
        float* d_o = d_Output + i * outputDim;
        
        int gridSize = (outputDim + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE;
        SparseCopyBias<<<gridSize, SPARSE_BLOCK_SIZE>>>(d_o, d_B2, outputDim);
        
        size_t sharedMem = 256 * sizeof(float);
        SparseSpMV_Warp<<<outputDim, 256, sharedMem>>>(d_W2_row_ptr, d_W2_col_idx, d_W2_values, 
                                                       d_h, d_o, outputDim, hiddenDim);
    }
    
    {
        int total = batchSize * outputDim;
        int gridSize = (total + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE;
        SparseSigmoid<<<gridSize, SPARSE_BLOCK_SIZE>>>(d_Output, total);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    gpu_copy_to_host_float(pred->data.data(), d_Output, batchSize * outputDim);

    // Free GPU memory
    gpu_free_float(d_Input);
    gpu_free_int(d_W1_row_ptr);
    gpu_free_int(d_W1_col_idx);
    gpu_free_float(d_W1_values);
    gpu_free_int(d_W2_row_ptr);
    gpu_free_int(d_W2_col_idx);
    gpu_free_float(d_W2_values);
    gpu_free_float(d_B1);
    gpu_free_float(d_B2);
    gpu_free_float(d_Hidden);
    gpu_free_float(d_Output);

    return pred;
}

}  // namespace swiftware::hpp

#include <cusparse.h>

#define CUSPARSE_CHECK(x)                                                      \
    do {                                                                       \
        cusparseStatus_t status = (x);                                         \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace swiftware::hpp {

// cuSPARSE-based Sparse Neural Network
DenseMatrix* gpu_sparse_nn_cusparse(DenseMatrix* InData, 
                                    CSR* W1, CSR* W2, 
                                    DenseMatrix* B1, DenseMatrix* B2, 
                                    ScheduleParams Sp) 
{
    const int batchSize = InData->m;
    const int inputDim  = InData->n;      
    const int hiddenDim = W1->m;          
    const int outputDim = W2->m;          

    auto* pred = new DenseMatrix(batchSize, outputDim);

    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    float* d_Input;
    CUDA_CHECK(cudaMalloc(&d_Input, batchSize * inputDim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Input, InData->data.data(), batchSize * inputDim * sizeof(float), cudaMemcpyHostToDevice));
    
    // W1 CSR: hiddenDim x inputDim
    int* d_W1_row_ptr;
    int* d_W1_col_idx;
    float* d_W1_values;
    CUDA_CHECK(cudaMalloc(&d_W1_row_ptr, (W1->m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_W1_col_idx, W1->nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_W1_values, W1->nnz * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W1_row_ptr, W1->row_ptr.data(), (W1->m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1_col_idx, W1->col_idx.data(), W1->nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1_values, W1->values.data(), W1->nnz * sizeof(float), cudaMemcpyHostToDevice));
    
    // W2 CSR: outputDim x hiddenDim
    int* d_W2_row_ptr;
    int* d_W2_col_idx;
    float* d_W2_values;
    CUDA_CHECK(cudaMalloc(&d_W2_row_ptr, (W2->m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_W2_col_idx, W2->nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_W2_values, W2->nnz * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W2_row_ptr, W2->row_ptr.data(), (W2->m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2_col_idx, W2->col_idx.data(), W2->nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2_values, W2->values.data(), W2->nnz * sizeof(float), cudaMemcpyHostToDevice));
    
    // Biases
    float* d_B1;
    float* d_B2;
    CUDA_CHECK(cudaMalloc(&d_B1, hiddenDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B2, outputDim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_B1, B1->data.data(), hiddenDim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B2, B2->data.data(), outputDim * sizeof(float), cudaMemcpyHostToDevice));
    
    // HiddenT: hiddenDim x batchSize - result of W1 * InputT
    float* d_HiddenT;
    CUDA_CHECK(cudaMalloc(&d_HiddenT, hiddenDim * batchSize * sizeof(float)));
    
    // Hidden: batchSize x hiddenDim - after transpose
    float* d_Hidden;
    CUDA_CHECK(cudaMalloc(&d_Hidden, batchSize * hiddenDim * sizeof(float)));
    
    // OutputT: outputDim x batchSize 
    float* d_OutputT;
    CUDA_CHECK(cudaMalloc(&d_OutputT, outputDim * batchSize * sizeof(float)));
    
    // Output: batchSize x outputDim 
    float* d_Output;
    CUDA_CHECK(cudaMalloc(&d_Output, batchSize * outputDim * sizeof(float)));
    
    cusparseSpMatDescr_t matW1, matW2;
    cusparseDnMatDescr_t matInputT, matHiddenT, matHiddenT2, matOutputT;
    
    CUSPARSE_CHECK(cusparseCreateCsr(&matW1, hiddenDim, inputDim, W1->nnz,
                                     d_W1_row_ptr, d_W1_col_idx, d_W1_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    CUSPARSE_CHECK(cusparseCreateCsr(&matW2, outputDim, hiddenDim, W2->nnz,
                                     d_W2_row_ptr, d_W2_col_idx, d_W2_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CUSPARSE_CHECK(cusparseCreateDnMat(&matInputT, inputDim, batchSize, inputDim,
                                       d_Input, CUDA_R_32F, CUSPARSE_ORDER_COL));
    
    // HiddenT: hiddenDim x batchSize
    CUSPARSE_CHECK(cusparseCreateDnMat(&matHiddenT, hiddenDim, batchSize, hiddenDim,
                                       d_HiddenT, CUDA_R_32F, CUSPARSE_ORDER_COL));

    float alpha = 1.0f;
    float beta = 0.0f;
    
    size_t bufferSize1 = 0;
    void* dBuffer1 = nullptr;
    
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matW1, matInputT, &beta, matHiddenT,
                                           CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                           &bufferSize1));
    CUDA_CHECK(cudaMalloc(&dBuffer1, bufferSize1));
    
    CUSPARSE_CHECK(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matW1, matInputT, &beta, matHiddenT,
                                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                dBuffer1));
    
    CUDA_CHECK(cudaMemcpy(d_Hidden, d_HiddenT, batchSize * hiddenDim * sizeof(float), cudaMemcpyDeviceToDevice));
    
    {
        int total = batchSize * hiddenDim;
        int gridSize = (total + 256 - 1) / 256;
        SparseAddBias<<<gridSize, 256>>>(d_Hidden, d_B1, batchSize, hiddenDim);
        SparseTanh<<<gridSize, 256>>>(d_Hidden, total);
    }

    CUSPARSE_CHECK(cusparseDestroyDnMat(matHiddenT));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matHiddenT2, hiddenDim, batchSize, hiddenDim,
                                       d_Hidden, CUDA_R_32F, CUSPARSE_ORDER_COL));
    
    // OutputT: outputDim x batchSize
    CUSPARSE_CHECK(cusparseCreateDnMat(&matOutputT, outputDim, batchSize, outputDim,
                                       d_OutputT, CUDA_R_32F, CUSPARSE_ORDER_COL));
    
    size_t bufferSize2 = 0;
    void* dBuffer2 = nullptr;
    
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matW2, matHiddenT2, &beta, matOutputT,
                                           CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                           &bufferSize2));
    CUDA_CHECK(cudaMalloc(&dBuffer2, bufferSize2));
    
    CUSPARSE_CHECK(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matW2, matHiddenT2, &beta, matOutputT,
                                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                dBuffer2));
    
    CUDA_CHECK(cudaMemcpy(d_Output, d_OutputT, batchSize * outputDim * sizeof(float), cudaMemcpyDeviceToDevice));
    
    {
        int total = batchSize * outputDim;
        int gridSize = (total + 256 - 1) / 256;
        SparseAddBias<<<gridSize, 256>>>(d_Output, d_B2, batchSize, outputDim);
        SparseSigmoid<<<gridSize, 256>>>(d_Output, total);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(pred->data.data(), d_Output, batchSize * outputDim * sizeof(float), cudaMemcpyDeviceToHost));

    CUSPARSE_CHECK(cusparseDestroySpMat(matW1));
    CUSPARSE_CHECK(cusparseDestroySpMat(matW2));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matInputT));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matHiddenT2));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matOutputT));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    
    CUDA_CHECK(cudaFree(d_Input));
    CUDA_CHECK(cudaFree(d_W1_row_ptr));
    CUDA_CHECK(cudaFree(d_W1_col_idx));
    CUDA_CHECK(cudaFree(d_W1_values));
    CUDA_CHECK(cudaFree(d_W2_row_ptr));
    CUDA_CHECK(cudaFree(d_W2_col_idx));
    CUDA_CHECK(cudaFree(d_W2_values));
    CUDA_CHECK(cudaFree(d_B1));
    CUDA_CHECK(cudaFree(d_B2));
    CUDA_CHECK(cudaFree(d_Hidden));
    CUDA_CHECK(cudaFree(d_HiddenT));
    CUDA_CHECK(cudaFree(d_Output));
    CUDA_CHECK(cudaFree(d_OutputT));
    CUDA_CHECK(cudaFree(dBuffer1));
    CUDA_CHECK(cudaFree(dBuffer2));

    return pred;
}

}  // namespace swiftware::hpp
