// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include <nvbench/nvbench.cuh>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernels.cuh"
#include "gpu_utils.h"

#define CUDA_CHECK(x) swiftware::hpp::cuda_check((x), __FILE__, __LINE__)

namespace hpp = swiftware::hpp;


void report_summary(nvbench::state& state)
{
    state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
    state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
    state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
    //state.get_summary("nv/cold/time/gpu/mean").set_string("hide", "");
    state.get_summary("nv/cold/time/cpu/mean").set_string("hide", "");
    state.get_summary("nv/cold/time/cpu/min").set_string("hide", "");
    state.get_summary("nv/cold/time/cpu/max").set_string("hide", "");
    state.get_summary("nv/cold/time/cpu/stdev/relative").set_string("hide", "");
    state.get_summary("nv/cold/sm_clock_rate/mean").remove_value("hide");
    state.get_summary("nv/cold/sm_clock_rate/scaling/percent").remove_value("hide");

}

// cuBLAS error checking
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while(0)

void nvbench_gemm_ours(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = static_cast<int>(state.get_int64("n"));
    const int k = static_cast<int>(state.get_int64("k"));

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    std::vector<float> h_C(m * n, 0.0f);

    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < k * n; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(256);
    dim3 gridDim((n + hpp::BN - 1) / hpp::BN, 
                 (m + hpp::BM - 1) / hpp::BM);

    state.add_element_count(2 * static_cast<int64_t>(m) * n * k, "FLOPs");
    state.add_global_memory_reads<float>(m * k + k * n, "Bytes Read");
    state.add_global_memory_writes<float>(m * n, "Bytes Written");

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        CUDA_CHECK(cudaMemset(d_C, 0, m * n * sizeof(float)));
        
        timer.start();
        hpp::MM<<<gridDim, blockDim, 0, launch.get_stream()>>>(d_A, d_B, d_C, m, n, k);
        cudaStreamSynchronize(launch.get_stream());
        timer.stop();
    });

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void nvbench_gemm_cublas(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = static_cast<int>(state.get_int64("n"));
    const int k = static_cast<int>(state.get_int64("k"));

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    std::vector<float> h_C(m * n, 0.0f);

    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < k * n; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    state.add_element_count(2 * static_cast<int64_t>(m) * n * k, "FLOPs");
    state.add_global_memory_reads<float>(m * k + k * n, "Bytes Read");
    state.add_global_memory_writes<float>(m * n, "Bytes Written");

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        CUBLAS_CHECK(cublasSetStream(handle, launch.get_stream()));
        
        timer.start();
        CUBLAS_CHECK(cublasSgemm(handle, 
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, m, k,  
                                 &alpha,
                                 d_B, n,   
                                 d_A, k,   
                                 &beta,
                                 d_C, n)); 
        cudaStreamSynchronize(launch.get_stream());
        timer.stop();
    });

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void nvbench_gemv_ours(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = static_cast<int>(state.get_int64("n"));

    std::vector<float> h_A(m * n);
    std::vector<float> h_x(n);
    std::vector<float> h_y(m, 0.0f);

    for (int i = 0; i < m * n; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < n; ++i) h_x[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, m * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int warpsPerBlock = 8;
    int blockSize = warpsPerBlock * 32;  
    int gridSize = (m + warpsPerBlock - 1) / warpsPerBlock;
    size_t sharedMem = blockSize * sizeof(float); 

    state.add_element_count(2 * static_cast<int64_t>(m) * n, "FLOPs");
    state.add_global_memory_reads<float>(m * n + n, "Bytes Read");
    state.add_global_memory_writes<float>(m, "Bytes Written");

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(float)));
        
        timer.start();
        hpp::MV<<<gridSize, blockSize, sharedMem, launch.get_stream()>>>(d_A, d_x, d_y, m, n);
        cudaStreamSynchronize(launch.get_stream());  // Ensure kernel completes
        timer.stop();
    });

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

void nvbench_gemv_cublas(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = static_cast<int>(state.get_int64("n"));

    std::vector<float> h_A(m * n);
    std::vector<float> h_x(n);
    std::vector<float> h_y(m, 0.0f);

    for (int i = 0; i < m * n; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < n; ++i) h_x[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, m * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    state.add_element_count(2 * static_cast<int64_t>(m) * n, "FLOPs");
    state.add_global_memory_reads<float>(m * n + n, "Bytes Read");
    state.add_global_memory_writes<float>(m, "Bytes Written");

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        CUBLAS_CHECK(cublasSetStream(handle, launch.get_stream()));
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(float)));
        
        timer.start();
        CUBLAS_CHECK(cublasSgemv(handle,
                                 CUBLAS_OP_T, 
                                 n, m,         
                                 &alpha,
                                 d_A, n,       
                                 d_x, 1,       
                                 &beta,
                                 d_y, 1));     
        cudaStreamSynchronize(launch.get_stream());  
        timer.stop();
    });

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

NVBENCH_BENCH(nvbench_gemm_ours)
    .set_name("GEMM_Custom")
    .add_int64_axis("m", {256, 512, 1024, 2048})
    .add_int64_axis("n", {256, 512, 1024, 2048})
    .add_int64_axis("k", {256, 512, 1024});

NVBENCH_BENCH(nvbench_gemm_cublas)
    .set_name("GEMM_cuBLAS")
    .add_int64_axis("m", {256, 512, 1024, 2048})
    .add_int64_axis("n", {256, 512, 1024, 2048})
    .add_int64_axis("k", {256, 512, 1024});

NVBENCH_BENCH(nvbench_gemv_ours)
    .set_name("GEMV_Custom")
    .add_int64_axis("m", {256, 512, 1024, 2048})
    .add_int64_axis("n", {256, 512, 784, 1024});

NVBENCH_BENCH(nvbench_gemv_cublas)
    .set_name("GEMV_cuBLAS")
    .add_int64_axis("m", {256, 512, 1024, 2048})
    .add_int64_axis("n", {256, 512, 784, 1024});
