// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Benchmarking dense NN implementations (GEMM vs GEMV)
// DO NOT DISTRIBUTE without permission.

#include "benchmark/benchmark.h"
#include "dense_nn.h"
#include "sparse_nn.h"
#include "utils.h"
#include <algorithm>
#include <memory>
#include <string>

namespace hpp = swiftware::hpp;

// Small helper to split MNIST into (labels, features) and normalize features
static void load_mnist_dataset(hpp::DenseMatrix*& labels,
                               hpp::DenseMatrix*& features)
{
    hpp::DenseMatrix* mnist = hpp::readCSV("./data/mnist_train.csv", /*hasHeader=*/true);

    labels   = new hpp::DenseMatrix(mnist->m, 1);
    features = new hpp::DenseMatrix(mnist->m, mnist->n - 1);

    for (int i = 0; i < mnist->m; ++i)
    {
        // label is first column
        labels->data[i] = mnist->data[i * mnist->n];

        // remaining columns are pixel intensities; keep in [0-255] range as per requirements
        for (int j = 0; j < features->n; ++j)
        {
            float pixel = mnist->data[i * mnist->n + (j + 1)];
            // Pixels are already in 0-255 range in the CSV, just copy them
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


static void BM_DENSENN_GEMM(benchmark::State& state)
{
    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);

    hpp::DenseMatrix *W1 = nullptr, *W2 = nullptr, *B1 = nullptr, *B2 = nullptr;
    load_model(W1, W2, B1, B2);


    hpp::ScheduleParams Sp(state.range(0), state.range(1));

    const int requestedBatch = state.range(2); 
    const int totalSamples   = features->m;


    const int batchSize =
        (requestedBatch <= 0 || requestedBatch > totalSamples)
        ? totalSamples
        : requestedBatch;

    hpp::DenseMatrix batchX(batchSize, features->n);
    hpp::DenseMatrix batchY(batchSize, 1);

    for (int i = 0; i < batchSize; ++i)
    {
        batchY.data[i] = labels->data[i];
        for (int j = 0; j < features->n; ++j)
        {
            batchX.data[i * features->n + j] =
                features->data[i * features->n + j];
        }
    }


    int fullDatasetCorrect = 0;
    hpp::DenseMatrix* fullPred = hpp::dense_nn_gemm(features, W1, W2, B1, B2, Sp);
    for (int i = 0; i < totalSamples; ++i)
    {
        const int outputDim = fullPred->n;
        const float* rowPtr = fullPred->data.data() + i * outputDim;
        auto it = std::max_element(rowPtr, rowPtr + outputDim);
        int argmaxIndex = static_cast<int>(std::distance(rowPtr, it));
        if (argmaxIndex == static_cast<int>(labels->data[i]))
        {
            ++fullDatasetCorrect;
        }
    }
    delete fullPred;
    double fullDatasetAccuracy = static_cast<double>(fullDatasetCorrect) / totalSamples;


    for (auto _ : state)
    {
        hpp::DenseMatrix* pred = hpp::dense_nn_gemm(
            &batchX, W1, W2, B1, B2, Sp);
        delete pred;
    }


    state.counters["Accuracy"] = fullDatasetAccuracy;

    delete labels;
    delete features;
    delete W1;
    delete W2;
    delete B1;
    delete B2;
}

static void BM_DENSENN_GEMV(benchmark::State& state)
{
    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);

    hpp::DenseMatrix *W1 = nullptr, *W2 = nullptr, *B1 = nullptr, *B2 = nullptr;
    load_model(W1, W2, B1, B2);

    hpp::ScheduleParams Sp(state.range(0), state.range(1));

    const int totalSamples = features->m;
    const int logicalBatch = state.range(2);   // expect 10

    const int effectiveBatch =
        (logicalBatch <= 0 || logicalBatch > totalSamples)
        ? std::min(10, totalSamples)
        : logicalBatch;


    int fullDatasetCorrect = 0;
    auto* singleXForAcc = new hpp::DenseMatrix(1, features->n);
    for (int i = 0; i < totalSamples; ++i)
    {
        for (int j = 0; j < features->n; ++j)
        {
            singleXForAcc->data[j] = features->data[i * features->n + j];
        }
        hpp::DenseMatrix* pred = hpp::dense_nn_gemv(singleXForAcc, W1, W2, B1, B2, Sp);
        const int outputDim = pred->n;
        const float* rowPtr = pred->data.data();
        auto it = std::max_element(rowPtr, rowPtr + outputDim);
        int argmaxIndex = static_cast<int>(std::distance(rowPtr, it));
        if (argmaxIndex == static_cast<int>(labels->data[i]))
        {
            ++fullDatasetCorrect;
        }
        delete pred;
    }
    delete singleXForAcc;
    double fullDatasetAccuracy = static_cast<double>(fullDatasetCorrect) / totalSamples;

    // Temporary 1×N matrix reused for each sample in benchmark
    auto* singleX = new hpp::DenseMatrix(1, features->n);

    // Benchmark loop - only measure inference time on the specified batch
    for (auto _ : state)
    {
        for (int i = 0; i < effectiveBatch; ++i)
        {
            // Copy i-th sample into singleX
            for (int j = 0; j < features->n; ++j)
            {
                singleX->data[j] =
                    features->data[i * features->n + j];
            }

            hpp::DenseMatrix* pred = hpp::dense_nn_gemv(
                singleX, W1, W2, B1, B2, Sp);

            delete pred;
        }
    }

    delete singleX;

    // Report the full dataset accuracy for all benchmarks
    state.counters["Accuracy"] = fullDatasetAccuracy;

    delete labels;
    delete features;
    delete W1;
    delete W2;
    delete B1;
    delete B2;
}


static void BM_SPARSENN_SPMM(benchmark::State& state)
{
    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);
    
    hpp::DenseMatrix *B1 = nullptr, *B2 = nullptr;
    B1 = hpp::readCSV("./data/model/biases_hidden.csv");
    B2 = hpp::readCSV("./data/model/biases_output.csv");
    
    const int sparsity = state.range(0);
    const int batchSize = state.range(1);
    hpp::ScheduleParams Sp(state.range(2), state.range(3));
    
    // Load sparse weight matrices
    std::string w1_path = "./data/model/" + std::to_string(sparsity) + "_W1.csv";
    std::string w2_path = "./data/model/" + std::to_string(sparsity) + "_W2.csv";
    
    hpp::CSR* W1 = hpp::readCSRFromDense(w1_path);
    hpp::CSR* W2 = hpp::readCSRFromDense(w2_path);
    
    const int totalSamples = features->m;
    const int effectiveBatch = (batchSize <= 0 || batchSize > totalSamples) ? totalSamples : batchSize;
    
    // Create batch
    hpp::DenseMatrix batchX(effectiveBatch, features->n);
    for (int i = 0; i < effectiveBatch; ++i) {
        for (int j = 0; j < features->n; ++j) {
            batchX.data[i * features->n + j] = features->data[i * features->n + j];
        }
    }
    
    // Calculate accuracy on full dataset once
    int fullDatasetCorrect = 0;
    hpp::DenseMatrix* fullPred = hpp::sparseNNSpmm(features, W1, W2, B1, B2, Sp);
    for (int i = 0; i < totalSamples; ++i) {
        const int outputDim = fullPred->n;
        const float* rowPtr = fullPred->data.data() + i * outputDim;
        auto it = std::max_element(rowPtr, rowPtr + outputDim);
        int argmaxIndex = static_cast<int>(std::distance(rowPtr, it));
        if (argmaxIndex == static_cast<int>(labels->data[i])) {
            ++fullDatasetCorrect;
        }
    }
    delete fullPred;
    double fullDatasetAccuracy = static_cast<double>(fullDatasetCorrect) / totalSamples;
    
    // Benchmark loop
    for (auto _ : state) {
        hpp::DenseMatrix* pred = hpp::sparseNNSpmm(&batchX, W1, W2, B1, B2, Sp);
        delete pred;
    }
    
    state.counters["Accuracy"] = fullDatasetAccuracy;
    
    delete labels;
    delete features;
    delete W1;
    delete W2;
    delete B1;
    delete B2;
}


static void BM_SPARSENN_SPMV(benchmark::State& state)
{
    hpp::DenseMatrix *labels = nullptr, *features = nullptr;
    load_mnist_dataset(labels, features);
    
    hpp::DenseMatrix *B1 = nullptr, *B2 = nullptr;
    B1 = hpp::readCSV("./data/model/biases_hidden.csv");
    B2 = hpp::readCSV("./data/model/biases_output.csv");
    
    const int sparsity = state.range(0);
    const int batchSize = state.range(1);
    hpp::ScheduleParams Sp(state.range(2), state.range(3));
    
    // Load sparse weight matrices
    std::string w1_path = "./data/model/" + std::to_string(sparsity) + "_W1.csv";
    std::string w2_path = "./data/model/" + std::to_string(sparsity) + "_W2.csv";
    
    hpp::CSR* W1 = hpp::readCSRFromDense(w1_path);
    hpp::CSR* W2 = hpp::readCSRFromDense(w2_path);
    
    const int totalSamples = features->m;
    const int effectiveBatch = (batchSize <= 0 || batchSize > totalSamples) ? std::min(10, totalSamples) : batchSize;
    
    // Calculate accuracy on full dataset once
    int fullDatasetCorrect = 0;
    auto* singleXForAcc = new hpp::DenseMatrix(1, features->n);
    for (int i = 0; i < totalSamples; ++i) {
        for (int j = 0; j < features->n; ++j) {
            singleXForAcc->data[j] = features->data[i * features->n + j];
        }
        hpp::DenseMatrix* pred = hpp::sparseNNSpmv(singleXForAcc, W1, W2, B1, B2, Sp);
        const int outputDim = pred->n;
        const float* rowPtr = pred->data.data();
        auto it = std::max_element(rowPtr, rowPtr + outputDim);
        int argmaxIndex = static_cast<int>(std::distance(rowPtr, it));
        if (argmaxIndex == static_cast<int>(labels->data[i])) {
            ++fullDatasetCorrect;
        }
        delete pred;
    }
    delete singleXForAcc;
    double fullDatasetAccuracy = static_cast<double>(fullDatasetCorrect) / totalSamples;
    
    // Benchmark loop
    auto* singleX = new hpp::DenseMatrix(1, features->n);
    for (auto _ : state) {
        for (int i = 0; i < effectiveBatch; ++i) {
            for (int j = 0; j < features->n; ++j) {
                singleX->data[j] = features->data[i * features->n + j];
            }
            hpp::DenseMatrix* pred = hpp::sparseNNSpmv(singleX, W1, W2, B1, B2, Sp);
            delete pred;
        }
    }
    delete singleX;
    
    state.counters["Accuracy"] = fullDatasetAccuracy;
    
    delete labels;
    delete features;
    delete W1;
    delete W2;
    delete B1;
    delete B2;
}

BENCHMARK(BM_DENSENN_GEMM)
    ->Args({32, 32, 10})
    ->Args({32, 32, 60000})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK(BM_DENSENN_GEMV)
    ->Args({32, 32, 10})
    ->Args({32, 32, 60000})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK(BM_SPARSENN_SPMM)
    ->Args({50, 60000, 32, 32})
    ->Args({55, 60000, 32, 32})
    ->Args({60, 60000, 32, 32})
    ->Args({65, 60000, 32, 32})
    ->Args({70, 60000, 32, 32})
    ->Args({75, 60000, 32, 32})
    ->Args({80, 60000, 32, 32})
    ->Args({85, 60000, 32, 32})
    ->Args({90, 60000, 32, 32})
    ->Args({95, 60000, 32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);

// Sparse NN benchmarks - SpMV version  
BENCHMARK(BM_SPARSENN_SPMV)
    ->Args({50, 10, 32, 32})
    ->Args({55, 10, 32, 32})
    ->Args({60, 10, 32, 32})
    ->Args({65, 10, 32, 32})
    ->Args({70, 10, 32, 32})
    ->Args({75, 10, 32, 32})
    ->Args({80, 10, 32, 32})
    ->Args({85, 10, 32, 32})
    ->Args({90, 10, 32, 32})
    ->Args({95, 10, 32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);

BENCHMARK_MAIN();
