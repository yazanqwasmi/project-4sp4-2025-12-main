// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "benchmark/benchmark.h"
#include "utils.h"
#include "dense_nn.h"
#include "spmm.h"
#include "spmv.h"
#include <iostream>

static void BM_GEMM(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);
    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    for (int i = 0; i < m * k; ++i) {
        A->data[i] = 1.0;
    }
    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0;
    }

    for (auto _: state) {
        // Reset C to zero before each iteration to avoid accumulation affecting results
        for (int i = 0; i < m * n; ++i) {
            C->data[i] = 0.0;
        }
        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }
    delete A;
    delete B;
    delete C;

}

static void BM_GEMV(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int t1 = state.range(2);
    int t2 = state.range(3);

    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *x = new swiftware::hpp::DenseMatrix(1, n);
    auto *y = new swiftware::hpp::DenseMatrix(1, m);

    for (int i = 0; i < m * n; ++i) A->data[i] = 1.0;
    for (int i = 0; i < n; ++i)    x->data[i] = 1.0;

    for (auto _ : state) {
        // Reset y to zero before each iteration since gemv accumulates
        for (int i = 0; i < m; ++i) {
            y->data[i] = 0.0;
        }
        swiftware::hpp::gemv(
            m, n,
            A->data.data(),
            x->data.data(),
            y->data.data(),
            swiftware::hpp::ScheduleParams(t1, t2)
        );
    }

    delete A;
    delete x;
    delete y;
}

static void BM_SPMM(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);

    // Create synthetic sparse matrix A (m x k) with ~10% sparsity
    int nnz = (m * k) / 10;  // Approximate 10% non-zero elements
    if (nnz < m) nnz = m;    // Ensure at least one element per row

    swiftware::hpp::CSR A(m, k, nnz);
    std::vector<float> B_data(k * n, 1.0f);
    std::vector<float> C_data(m * n, 0.0f);

    // Create a simple sparse pattern: every 10th element
    int idx = 0;
    A.row_ptr[0] = 0;
    for (int i = 0; i < m; ++i) {
        int row_nnz = (nnz - idx) / (m - i);  // Distribute remaining elements
        if (row_nnz < 1) row_nnz = 1;
        if (idx + row_nnz > nnz) row_nnz = nnz - idx;

        for (int j = 0; j < row_nnz && idx < nnz; ++j) {
            int col = (i * k / m + j * k / row_nnz) % k;  // Distribute across columns
            A.col_idx[idx] = col;
            A.values[idx] = 1.0f;
            idx++;
        }
        A.row_ptr[i + 1] = idx;
    }

    for (auto _: state) {
        // Reset C to zero before each iteration
        std::fill(C_data.begin(), C_data.end(), 0.0f);
        swiftware::hpp::spmmCSR(m, n, k, A.row_ptr.data(), A.col_idx.data(),
                                A.values.data(), B_data.data(), C_data.data(),
                                swiftware::hpp::ScheduleParams(t1, t2));
    }
}

static void BM_SPMV(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int t1 = state.range(2);
    int t2 = state.range(3);

    // Create synthetic sparse matrix A (m x n) with ~10% sparsity
    int nnz = (m * n) / 10;  // Approximate 10% non-zero elements
    if (nnz < m) nnz = m;    // Ensure at least one element per row

    swiftware::hpp::CSR A(m, n, nnz);
    std::vector<float> x_data(n, 1.0f);
    std::vector<float> y_data(m, 0.0f);


    int idx = 0;
    A.row_ptr[0] = 0;
    for (int i = 0; i < m; ++i) {
        int row_nnz = (nnz - idx) / (m - i);  // Distribute remaining elements
        if (row_nnz < 1) row_nnz = 1;
        if (idx + row_nnz > nnz) row_nnz = nnz - idx;

        for (int j = 0; j < row_nnz && idx < nnz; ++j) {
            int col = (i * n / m + j * n / row_nnz) % n;  // Distribute across columns
            A.col_idx[idx] = col;
            A.values[idx] = 1.0f;
            idx++;
        }
        A.row_ptr[i + 1] = idx;
    }

    for (auto _: state) {
        // Reset y to zero before each iteration
        std::fill(y_data.begin(), y_data.end(), 0.0f);
        swiftware::hpp::spmvCSR(m, n, A.row_ptr.data(), A.col_idx.data(),
                                A.values.data(), x_data.data(), y_data.data(),
                                swiftware::hpp::ScheduleParams(t1, t2));
    }
}


BENCHMARK(BM_GEMM)
    ->Args({32, 32, 32, 32, 32})
    ->Args({64, 64, 64, 32, 32})
    ->Args({64, 64, 64, 64, 64})
    ->Args({128, 128, 128, 32, 32})
    ->Args({128, 128, 128, 64, 64})
    ->Args({128, 128, 128, 128, 128})
    ->Args({256, 256, 256, 32, 32})
    ->Args({256, 256, 256, 64, 64})
    ->Args({256, 256, 256, 128, 128})
    ->Args({512, 512, 512, 32, 32})
    ->Args({512, 512, 512, 64, 64})
    ->Args({512, 512, 512, 128, 128})
    ->Args({1024, 1024, 1024, 32, 32})
    ->Args({1024, 1024, 1024, 64, 64})
    ->Args({1024, 1024, 1024, 128, 128})
    ->Args({2048, 2048, 2048, 32, 32})
    ->Args({2048, 2048, 2048, 64, 64})
    ->Args({2048, 2048, 2048, 128, 128})
    ->Args({4096, 4096, 4096, 32, 32})
    ->Args({4096, 4096, 4096, 64, 64})
    ->Args({4096, 4096, 4096, 128, 128})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);


BENCHMARK(BM_GEMV)
    ->Args({32, 32, 32, 32})
    ->Args({64, 64, 32, 32})
    ->Args({64, 64, 64, 64})
    ->Args({128, 128, 32, 32})
    ->Args({128, 128, 64, 64})
    ->Args({128, 128, 128, 128})
    ->Args({256, 256, 32, 32})
    ->Args({256, 256, 64, 64})
    ->Args({256, 256, 128, 128})
    ->Args({512, 512, 32, 32})
    ->Args({512, 512, 64, 64})
    ->Args({512, 512, 128, 128})
    ->Args({1024, 1024, 32, 32})
    ->Args({1024, 1024, 64, 64})
    ->Args({1024, 1024, 128, 128})
    ->Args({2048, 2048, 32, 32})
    ->Args({2048, 2048, 64, 64})
    ->Args({2048, 2048, 128, 128})
    ->Args({4096, 4096, 32, 32})
    ->Args({4096, 4096, 64, 64})
    ->Args({4096, 4096, 128, 128})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);


BENCHMARK(BM_SPMM)
    ->Args({32, 32, 32, 32, 32})
    ->Args({64, 64, 64, 32, 32})
    ->Args({64, 64, 64, 64, 64})
    ->Args({128, 128, 128, 32, 32})
    ->Args({128, 128, 128, 64, 64})
    ->Args({128, 128, 128, 128, 128})
    ->Args({256, 256, 256, 32, 32})
    ->Args({256, 256, 256, 64, 64})
    ->Args({256, 256, 256, 128, 128})
    ->Args({512, 512, 512, 32, 32})
    ->Args({512, 512, 512, 64, 64})
    ->Args({512, 512, 512, 128, 128})
    ->Args({1024, 1024, 1024, 32, 32})
    ->Args({1024, 1024, 1024, 64, 64})
    ->Args({1024, 1024, 1024, 128, 128})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);


BENCHMARK(BM_SPMV)
    ->Args({32, 32, 32, 32})
    ->Args({64, 64, 32, 32})
    ->Args({64, 64, 64, 64})
    ->Args({128, 128, 32, 32})
    ->Args({128, 128, 64, 64})
    ->Args({128, 128, 128, 128})
    ->Args({256, 256, 32, 32})
    ->Args({256, 256, 64, 64})
    ->Args({256, 256, 128, 128})
    ->Args({512, 512, 32, 32})
    ->Args({512, 512, 64, 64})
    ->Args({512, 512, 128, 128})
    ->Args({1024, 1024, 32, 32})
    ->Args({1024, 1024, 64, 64})
    ->Args({1024, 1024, 128, 128})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK_MAIN();
