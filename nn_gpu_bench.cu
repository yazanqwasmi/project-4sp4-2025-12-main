// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include <nvbench/nvbench.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

#include "gpu_dense_nn.cuh"
#include "gpu_sparse_nn.cuh"
#include "gpu_utils.h"
#include "utils.h"

#define CUDA_CHECK(x) swiftware::hpp::cuda_check((x), __FILE__, __LINE__)

namespace hpp = swiftware::hpp;

static void load_mnist_dataset(hpp::DenseMatrix*& labels,
                               hpp::DenseMatrix*& features)
{
    hpp::DenseMatrix* mnist = hpp::readCSV("./data/mnist_train.csv", /*hasHeader=*/true);

    labels   = new hpp::DenseMatrix(mnist->m, 1);
    features = new hpp::DenseMatrix(mnist->m, mnist->n - 1);

    for (int i = 0; i < mnist->m; ++i)
    {
        labels->data[i] = mnist->data[i * mnist->n];
        for (int j = 0; j < features->n; ++j)
        {
            float pixel = mnist->data[i * mnist->n + (j + 1)];
            features->data[i * features->n + j] = pixel;
        }
    }

    delete mnist;
}

static void load_model(hpp::DenseMatrix*& W1,
                       hpp::DenseMatrix*& W2,
                       hpp::DenseMatrix*& B1,
                       hpp::DenseMatrix*& B2)
{
    W1 = hpp::readCSV("./data/model/weights_hidden.csv");
    W2 = hpp::readCSV("./data/model/weights_output.csv");
    B1 = hpp::readCSV("./data/model/biases_hidden.csv");
    B2 = hpp::readCSV("./data/model/biases_output.csv");
}

static hpp::CSR* dense_to_csr_pruned(hpp::DenseMatrix* dense, float prune_threshold = 0.0f) {
    int m = dense->m;
    int n = dense->n;
    
    int nnz = 0;
    for (int i = 0; i < m * n; ++i) {
        if (std::abs(dense->data[i]) > prune_threshold) {
            ++nnz;
        }
    }
    
    auto* csr = new hpp::CSR(m, n, nnz);
    
    int idx = 0;
    for (int i = 0; i < m; ++i) {
        csr->row_ptr[i] = idx;
        for (int j = 0; j < n; ++j) {
            float val = dense->data[i * n + j];
            if (std::abs(val) > prune_threshold) {
                csr->col_idx[idx] = j;
                csr->values[idx] = val;
                ++idx;
            }
        }
    }
    csr->row_ptr[m] = nnz;
    
    return csr;
}

static hpp::CSR* dense_to_csr_target_sparsity(hpp::DenseMatrix* dense, float target_sparsity_pct) {
    int m = dense->m;
    int n = dense->n;
    int total = m * n;
    
    std::vector<float> abs_vals(total);
    for (int i = 0; i < total; ++i) {
        abs_vals[i] = std::abs(dense->data[i]);
    }
    std::sort(abs_vals.begin(), abs_vals.end());
    
    int target_zeros = static_cast<int>(total * target_sparsity_pct / 100.0f);
    float threshold = (target_zeros > 0 && target_zeros < total) ? abs_vals[target_zeros] : 0.0f;
    
    int nnz = 0;
    for (int i = 0; i < total; ++i) {
        if (std::abs(dense->data[i]) > threshold) {
            ++nnz;
        }
    }
    
    auto* csr = new hpp::CSR(m, n, nnz);
    
    int idx = 0;
    for (int i = 0; i < m; ++i) {
        csr->row_ptr[i] = idx;
        for (int j = 0; j < n; ++j) {
            float val = dense->data[i * n + j];
            if (std::abs(val) > threshold) {
                csr->col_idx[idx] = j;
                csr->values[idx] = val;
                ++idx;
            }
        }
    }
    csr->row_ptr[m] = nnz;
    
    return csr;
}

static void load_sparse_model(hpp::CSR*& W1, hpp::CSR*& W2,
                              hpp::DenseMatrix*& B1, hpp::DenseMatrix*& B2,
                              float prune_threshold = 0.01f)
{
    hpp::DenseMatrix* W1_dense = hpp::readCSV("./data/model/weights_hidden.csv");
    hpp::DenseMatrix* W2_dense = hpp::readCSV("./data/model/weights_output.csv");
    B1 = hpp::readCSV("./data/model/biases_hidden.csv");
    B2 = hpp::readCSV("./data/model/biases_output.csv");
    
    W1 = dense_to_csr_pruned(W1_dense, prune_threshold);
    W2 = dense_to_csr_pruned(W2_dense, prune_threshold);
    
    float sparsity1 = 1.0f - (float)W1->nnz / (W1->m * W1->n);
    float sparsity2 = 1.0f - (float)W2->nnz / (W2->m * W2->n);
    printf("W1 sparsity: %.2f%% (%d/%d non-zeros)\n", sparsity1 * 100, W1->nnz, W1->m * W1->n);
    printf("W2 sparsity: %.2f%% (%d/%d non-zeros)\n", sparsity2 * 100, W2->nnz, W2->m * W2->n);
    
    delete W1_dense;
    delete W2_dense;
}

static void load_sparse_model_target_sparsity(hpp::CSR*& W1, hpp::CSR*& W2,
                                              hpp::DenseMatrix*& B1, hpp::DenseMatrix*& B2,
                                              float target_sparsity_pct)
{
    hpp::DenseMatrix* W1_dense = hpp::readCSV("./data/model/weights_hidden.csv");
    hpp::DenseMatrix* W2_dense = hpp::readCSV("./data/model/weights_output.csv");
    B1 = hpp::readCSV("./data/model/biases_hidden.csv");
    B2 = hpp::readCSV("./data/model/biases_output.csv");
    
    W1 = dense_to_csr_target_sparsity(W1_dense, target_sparsity_pct);
    W2 = dense_to_csr_target_sparsity(W2_dense, target_sparsity_pct);
    
    float sparsity1 = 1.0f - (float)W1->nnz / (W1->m * W1->n);
    float sparsity2 = 1.0f - (float)W2->nnz / (W2->m * W2->n);
    printf("Target: %.0f%% | W1 actual: %.2f%% | W2 actual: %.2f%%\n", 
           target_sparsity_pct, sparsity1 * 100, sparsity2 * 100);
    
    delete W1_dense;
    delete W2_dense;
}

void nvbench_gpu_dense_nn_gemm(nvbench::state& state)
{
    const int batchSize = static_cast<int>(state.get_int64("batch_size"));

    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);

    hpp::DenseMatrix *W1 = nullptr, *W2 = nullptr, *B1 = nullptr, *B2 = nullptr;
    load_model(W1, W2, B1, B2);

    const int totalSamples = features->m;
    const int effectiveBatch = std::min(batchSize, totalSamples);

    hpp::DenseMatrix batchX(effectiveBatch, features->n);
    hpp::DenseMatrix batchY(effectiveBatch, 1);

    for (int i = 0; i < effectiveBatch; ++i) {
        batchY.data[i] = labels->data[i];
        for (int j = 0; j < features->n; ++j) {
            batchX.data[i * features->n + j] = features->data[i * features->n + j];
        }
    }

    hpp::ScheduleParams Sp(32, 32);

    int totalCorrect = 0;
    int totalSeen = 0;

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        timer.start();

        hpp::DenseMatrix* pred = hpp::gpu_dense_nn_gemm(&batchX, W1, W2, B1, B2, Sp);

        timer.stop();

        for (int i = 0; i < effectiveBatch; ++i) {
            const int outputDim = pred->n;
            const float* rowPtr = pred->data.data() + i * outputDim;
            auto it = std::max_element(rowPtr, rowPtr + outputDim);
            int argmaxIndex = static_cast<int>(std::distance(rowPtr, it));
            
            if (argmaxIndex == static_cast<int>(batchY.data[i])) {
                ++totalCorrect;
            }
            ++totalSeen;
        }

        delete pred;
    });

    double accuracy = totalSeen > 0 ? static_cast<double>(totalCorrect) / totalSeen : 0.0;
    state.add_element_count(effectiveBatch, "Samples");
    state.add_global_memory_reads<float>(effectiveBatch * features->n + 
                                         W1->m * W1->n + W2->m * W2->n + 
                                         B1->n + B2->n, "Data Read");
    state.add_global_memory_writes<float>(effectiveBatch * W2->m, "Data Written");  
    state.add_buffer_size(static_cast<int64_t>(accuracy * 10000), "Accuracy_x10000");

    printf("GPU GEMM Accuracy: %.4f%% (%d/%d)\n", accuracy * 100.0, totalCorrect, totalSeen);

    delete labels;
    delete features;
    delete W1;
    delete W2;
    delete B1;
    delete B2;
}

void nvbench_gpu_dense_nn_gemv(nvbench::state& state)
{
    const int batchSize = static_cast<int>(state.get_int64("batch_size"));

    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);

    hpp::DenseMatrix *W1 = nullptr, *W2 = nullptr, *B1 = nullptr, *B2 = nullptr;
    load_model(W1, W2, B1, B2);

    const int totalSamples = features->m;
    const int effectiveBatch = std::min(batchSize, totalSamples);

    hpp::DenseMatrix batchX(effectiveBatch, features->n);
    hpp::DenseMatrix batchY(effectiveBatch, 1);

    for (int i = 0; i < effectiveBatch; ++i) {
        batchY.data[i] = labels->data[i];
        for (int j = 0; j < features->n; ++j) {
            batchX.data[i * features->n + j] = features->data[i * features->n + j];
        }
    }

    hpp::ScheduleParams Sp(32, 32);

    int totalCorrect = 0;
    int totalSeen = 0;

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        timer.start();

        hpp::DenseMatrix* pred = hpp::gpu_dense_nn_gemv(&batchX, W1, W2, B1, B2, Sp);

        timer.stop();

        for (int i = 0; i < effectiveBatch; ++i) {
            const int outputDim = pred->n;
            const float* rowPtr = pred->data.data() + i * outputDim;
            auto it = std::max_element(rowPtr, rowPtr + outputDim);
            int argmaxIndex = static_cast<int>(std::distance(rowPtr, it));
            
            if (argmaxIndex == static_cast<int>(batchY.data[i])) {
                ++totalCorrect;
            }
            ++totalSeen;
        }

        delete pred;
    });

    double accuracy = totalSeen > 0 ? static_cast<double>(totalCorrect) / totalSeen : 0.0;
    state.add_element_count(effectiveBatch, "Samples");
    state.add_buffer_size(static_cast<int64_t>(accuracy * 10000), "Accuracy_x10000");

    printf("GPU GEMV Accuracy: %.4f%% (%d/%d)\n", accuracy * 100.0, totalCorrect, totalSeen);

    delete labels;
    delete features;
    delete W1;
    delete W2;
    delete B1;
    delete B2;
}

void nvbench_gpu_dense_nn_cublas(nvbench::state& state)
{
    const int batchSize = static_cast<int>(state.get_int64("batch_size"));

    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);

    hpp::DenseMatrix *W1 = nullptr, *W2 = nullptr, *B1 = nullptr, *B2 = nullptr;
    load_model(W1, W2, B1, B2);

    const int totalSamples = features->m;
    const int effectiveBatch = std::min(batchSize, totalSamples);

    hpp::DenseMatrix batchX(effectiveBatch, features->n);
    hpp::DenseMatrix batchY(effectiveBatch, 1);

    for (int i = 0; i < effectiveBatch; ++i) {
        batchY.data[i] = labels->data[i];
        for (int j = 0; j < features->n; ++j) {
            batchX.data[i * features->n + j] = features->data[i * features->n + j];
        }
    }

    hpp::ScheduleParams Sp(32, 32);

    int totalCorrect = 0;
    int totalSeen = 0;

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        timer.start();

        hpp::DenseMatrix* pred = hpp::gpu_dense_nn_cublas(&batchX, W1, W2, B1, B2, Sp);

        timer.stop();

        for (int i = 0; i < effectiveBatch; ++i) {
            const int outputDim = pred->n;
            const float* rowPtr = pred->data.data() + i * outputDim;
            auto it = std::max_element(rowPtr, rowPtr + outputDim);
            int argmaxIndex = static_cast<int>(std::distance(rowPtr, it));
            
            if (argmaxIndex == static_cast<int>(batchY.data[i])) {
                ++totalCorrect;
            }
            ++totalSeen;
        }

        delete pred;
    });

    double accuracy = totalSeen > 0 ? static_cast<double>(totalCorrect) / totalSeen : 0.0;
    state.add_element_count(effectiveBatch, "Samples");
    state.add_global_memory_reads<float>(effectiveBatch * features->n + 
                                         W1->m * W1->n + W2->m * W2->n + 
                                         B1->n + B2->n, "Data Read");
    state.add_global_memory_writes<float>(effectiveBatch * W2->m, "Data Written");
    state.add_buffer_size(static_cast<int64_t>(accuracy * 10000), "Accuracy_x10000");

    printf("GPU cuBLAS Accuracy: %.4f%% (%d/%d)\n", accuracy * 100.0, totalCorrect, totalSeen);

    delete labels;
    delete features;
    delete W1;
    delete W2;
    delete B1;
    delete B2;
}

void nvbench_gpu_sparse_nn_spmm(nvbench::state& state)
{
    const int batchSize = static_cast<int>(state.get_int64("batch_size"));
    const float prune_threshold = 0.1f;  // Prune weights with magnitude <= 0.1

    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);

    hpp::CSR *W1 = nullptr, *W2 = nullptr;
    hpp::DenseMatrix *B1 = nullptr, *B2 = nullptr;
    load_sparse_model(W1, W2, B1, B2, prune_threshold);

    const int totalSamples = features->m;
    const int effectiveBatch = std::min(batchSize, totalSamples);

    hpp::DenseMatrix batchX(effectiveBatch, features->n);
    hpp::DenseMatrix batchY(effectiveBatch, 1);

    for (int i = 0; i < effectiveBatch; ++i) {
        batchY.data[i] = labels->data[i];
        for (int j = 0; j < features->n; ++j) {
            batchX.data[i * features->n + j] = features->data[i * features->n + j];
        }
    }

    hpp::ScheduleParams Sp(32, 32);

    int totalCorrect = 0;
    int totalSeen = 0;

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        timer.start();

        hpp::DenseMatrix* pred = hpp::gpu_sparse_nn_spmm(&batchX, W1, W2, B1, B2, Sp);

        timer.stop();

        // Calculate accuracy
        for (int i = 0; i < effectiveBatch; ++i) {
            const int outputDim = pred->n;
            const float* rowPtr = pred->data.data() + i * outputDim;
            auto it = std::max_element(rowPtr, rowPtr + outputDim);
            int argmaxIndex = static_cast<int>(std::distance(rowPtr, it));
            
            if (argmaxIndex == static_cast<int>(batchY.data[i])) {
                ++totalCorrect;
            }
            ++totalSeen;
        }

        delete pred;
    });

    double accuracy = totalSeen > 0 ? static_cast<double>(totalCorrect) / totalSeen : 0.0;
    state.add_element_count(effectiveBatch, "Samples");
    state.add_buffer_size(static_cast<int64_t>(accuracy * 10000), "Accuracy_x10000");

    printf("GPU Sparse SpMM Accuracy: %.4f%% (%d/%d)\n", accuracy * 100.0, totalCorrect, totalSeen);

    delete labels;
    delete features;
    delete W1;
    delete W2;
    delete B1;
    delete B2;
}

void nvbench_gpu_sparse_nn_spmv(nvbench::state& state)
{
    const int batchSize = static_cast<int>(state.get_int64("batch_size"));
    const float prune_threshold = 0.1f;  // Same as SpMM

    // Load data
    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);

    hpp::CSR *W1 = nullptr, *W2 = nullptr;
    hpp::DenseMatrix *B1 = nullptr, *B2 = nullptr;
    load_sparse_model(W1, W2, B1, B2, prune_threshold);

    const int totalSamples = features->m;
    const int effectiveBatch = std::min(batchSize, totalSamples);

    // Create batch
    hpp::DenseMatrix batchX(effectiveBatch, features->n);
    hpp::DenseMatrix batchY(effectiveBatch, 1);

    for (int i = 0; i < effectiveBatch; ++i) {
        batchY.data[i] = labels->data[i];
        for (int j = 0; j < features->n; ++j) {
            batchX.data[i * features->n + j] = features->data[i * features->n + j];
        }
    }

    hpp::ScheduleParams Sp(32, 32);

    int totalCorrect = 0;
    int totalSeen = 0;

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        timer.start();

        hpp::DenseMatrix* pred = hpp::gpu_sparse_nn_spmv(&batchX, W1, W2, B1, B2, Sp);

        timer.stop();

        // Calculate accuracy
        for (int i = 0; i < effectiveBatch; ++i) {
            const int outputDim = pred->n;
            const float* rowPtr = pred->data.data() + i * outputDim;
            auto it = std::max_element(rowPtr, rowPtr + outputDim);
            int argmaxIndex = static_cast<int>(std::distance(rowPtr, it));
            
            if (argmaxIndex == static_cast<int>(batchY.data[i])) {
                ++totalCorrect;
            }
            ++totalSeen;
        }

        delete pred;
    });

    double accuracy = totalSeen > 0 ? static_cast<double>(totalCorrect) / totalSeen : 0.0;
    state.add_element_count(effectiveBatch, "Samples");
    state.add_buffer_size(static_cast<int64_t>(accuracy * 10000), "Accuracy_x10000");

    printf("GPU Sparse SpMV Accuracy: %.4f%% (%d/%d)\n", accuracy * 100.0, totalCorrect, totalSeen);

    delete labels;
    delete features;
    delete W1;
    delete W2;
    delete B1;
    delete B2;
}

void nvbench_gpu_sparse_nn_cusparse(nvbench::state& state)
{
    const int batchSize = static_cast<int>(state.get_int64("batch_size"));
    const float prune_threshold = 0.1f;  // Same as custom SpMM

    // Load data
    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);

    hpp::CSR *W1 = nullptr, *W2 = nullptr;
    hpp::DenseMatrix *B1 = nullptr, *B2 = nullptr;
    load_sparse_model(W1, W2, B1, B2, prune_threshold);

    const int totalSamples = features->m;
    const int effectiveBatch = std::min(batchSize, totalSamples);

    // Create batch
    hpp::DenseMatrix batchX(effectiveBatch, features->n);
    hpp::DenseMatrix batchY(effectiveBatch, 1);

    for (int i = 0; i < effectiveBatch; ++i) {
        batchY.data[i] = labels->data[i];
        for (int j = 0; j < features->n; ++j) {
            batchX.data[i * features->n + j] = features->data[i * features->n + j];
        }
    }

    hpp::ScheduleParams Sp(32, 32);

    int totalCorrect = 0;
    int totalSeen = 0;

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        timer.start();

        hpp::DenseMatrix* pred = hpp::gpu_sparse_nn_cusparse(&batchX, W1, W2, B1, B2, Sp);

        timer.stop();

        // Calculate accuracy
        for (int i = 0; i < effectiveBatch; ++i) {
            const int outputDim = pred->n;
            const float* rowPtr = pred->data.data() + i * outputDim;
            auto it = std::max_element(rowPtr, rowPtr + outputDim);
            int argmaxIndex = static_cast<int>(std::distance(rowPtr, it));
            
            if (argmaxIndex == static_cast<int>(batchY.data[i])) {
                ++totalCorrect;
            }
            ++totalSeen;
        }

        delete pred;
    });

    double accuracy = totalSeen > 0 ? static_cast<double>(totalCorrect) / totalSeen : 0.0;
    state.add_element_count(effectiveBatch, "Samples");
    state.add_buffer_size(static_cast<int64_t>(accuracy * 10000), "Accuracy_x10000");

    printf("GPU cuSPARSE Sparse Accuracy: %.4f%% (%d/%d)\n", accuracy * 100.0, totalCorrect, totalSeen);

    delete labels;
    delete features;
    delete W1;
    delete W2;
    delete B1;
    delete B2;
}

void nvbench_gpu_sparse_nn_spmm_sparsity(nvbench::state& state)
{
    const int batchSize = 1000;  // Fixed batch for sparsity comparison
    const float sparsity_pct = static_cast<float>(state.get_int64("sparsity_pct"));

    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);

    hpp::CSR *W1 = nullptr, *W2 = nullptr;
    hpp::DenseMatrix *B1 = nullptr, *B2 = nullptr;
    load_sparse_model_target_sparsity(W1, W2, B1, B2, sparsity_pct);

    const int totalSamples = features->m;
    const int effectiveBatch = std::min(batchSize, totalSamples);

    hpp::DenseMatrix batchX(effectiveBatch, features->n);
    hpp::DenseMatrix batchY(effectiveBatch, 1);

    for (int i = 0; i < effectiveBatch; ++i) {
        batchY.data[i] = labels->data[i];
        for (int j = 0; j < features->n; ++j) {
            batchX.data[i * features->n + j] = features->data[i * features->n + j];
        }
    }

    hpp::ScheduleParams Sp(32, 32);
    int totalCorrect = 0, totalSeen = 0;

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        timer.start();
        hpp::DenseMatrix* pred = hpp::gpu_sparse_nn_spmm(&batchX, W1, W2, B1, B2, Sp);
        timer.stop();

        for (int i = 0; i < effectiveBatch; ++i) {
            const float* rowPtr = pred->data.data() + i * pred->n;
            auto it = std::max_element(rowPtr, rowPtr + pred->n);
            if (std::distance(rowPtr, it) == static_cast<int>(batchY.data[i])) ++totalCorrect;
            ++totalSeen;
        }
        delete pred;
    });

    double accuracy = totalSeen > 0 ? static_cast<double>(totalCorrect) / totalSeen : 0.0;
    printf("Sparsity %.0f%% Accuracy: %.2f%%\n", sparsity_pct, accuracy * 100.0);

    delete labels; delete features; delete W1; delete W2; delete B1; delete B2;
}

void nvbench_gpu_sparse_nn_spmv_sparsity(nvbench::state& state)
{
    const int batchSize = 1000;
    const float sparsity_pct = static_cast<float>(state.get_int64("sparsity_pct"));

    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);

    hpp::CSR *W1 = nullptr, *W2 = nullptr;
    hpp::DenseMatrix *B1 = nullptr, *B2 = nullptr;
    load_sparse_model_target_sparsity(W1, W2, B1, B2, sparsity_pct);

    const int totalSamples = features->m;
    const int effectiveBatch = std::min(batchSize, totalSamples);

    hpp::DenseMatrix batchX(effectiveBatch, features->n);
    hpp::DenseMatrix batchY(effectiveBatch, 1);

    for (int i = 0; i < effectiveBatch; ++i) {
        batchY.data[i] = labels->data[i];
        for (int j = 0; j < features->n; ++j) {
            batchX.data[i * features->n + j] = features->data[i * features->n + j];
        }
    }

    hpp::ScheduleParams Sp(32, 32);
    int totalCorrect = 0, totalSeen = 0;

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        timer.start();
        hpp::DenseMatrix* pred = hpp::gpu_sparse_nn_spmv(&batchX, W1, W2, B1, B2, Sp);
        timer.stop();

        for (int i = 0; i < effectiveBatch; ++i) {
            const float* rowPtr = pred->data.data() + i * pred->n;
            auto it = std::max_element(rowPtr, rowPtr + pred->n);
            if (std::distance(rowPtr, it) == static_cast<int>(batchY.data[i])) ++totalCorrect;
            ++totalSeen;
        }
        delete pred;
    });

    double accuracy = totalSeen > 0 ? static_cast<double>(totalCorrect) / totalSeen : 0.0;
    printf("Sparsity %.0f%% Accuracy: %.2f%%\n", sparsity_pct, accuracy * 100.0);

    delete labels; delete features; delete W1; delete W2; delete B1; delete B2;
}

// =============================================================================
// Benchmark Registration
// =============================================================================

// Dense NN GEMM benchmark with different batch sizes
NVBENCH_BENCH(nvbench_gpu_dense_nn_gemm)
    .set_name("GPU_DenseNN_GEMM")
    .add_int64_axis("batch_size", {10, 100, 1000, 10000, 60000});

// Dense NN GEMV benchmark with different batch sizes
NVBENCH_BENCH(nvbench_gpu_dense_nn_gemv)
    .set_name("GPU_DenseNN_GEMV")
    .add_int64_axis("batch_size", {10, 100, 1000, 10000, 60000});

// Dense NN cuBLAS benchmark with different batch sizes
NVBENCH_BENCH(nvbench_gpu_dense_nn_cublas)
    .set_name("GPU_DenseNN_cuBLAS")
    .add_int64_axis("batch_size", {10, 100, 1000, 10000, 60000});

// Sparse NN SpMM benchmark with different batch sizes
NVBENCH_BENCH(nvbench_gpu_sparse_nn_spmm)
    .set_name("GPU_SparseNN_SpMM")
    .add_int64_axis("batch_size", {10, 100, 1000, 10000, 60000});

// Sparse NN SpMV benchmark with different batch sizes
NVBENCH_BENCH(nvbench_gpu_sparse_nn_spmv)
    .set_name("GPU_SparseNN_SpMV")
    .add_int64_axis("batch_size", {10, 100, 1000, 10000, 60000});

// Sparse NN cuSPARSE benchmark with different batch sizes
NVBENCH_BENCH(nvbench_gpu_sparse_nn_cusparse)
    .set_name("GPU_SparseNN_cuSPARSE")
    .add_int64_axis("batch_size", {10, 100, 1000, 10000, 60000});

// Sparse NN SpMM with sparsity sweep (50%-95%, step 5%)
NVBENCH_BENCH(nvbench_gpu_sparse_nn_spmm_sparsity)
    .set_name("GPU_SparseNN_SpMM_Sparsity")
    .add_int64_axis("sparsity_pct", {50, 55, 60, 65, 70, 75, 80, 85, 90, 95});

// Sparse NN SpMV with sparsity sweep (50%-95%, step 5%)
NVBENCH_BENCH(nvbench_gpu_sparse_nn_spmv_sparsity)
    .set_name("GPU_SparseNN_SpMV_Sparsity")
    .add_int64_axis("sparsity_pct", {50, 55, 60, 65, 70, 75, 80, 85, 90, 95});

