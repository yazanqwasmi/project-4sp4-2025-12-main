// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cuda_runtime.h>

namespace swiftware::hpp
{
    constexpr int TILE_SIZE = 32;
    constexpr int BLOCK_SIZE_1D = 256;

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    __global__ void MM(const float* __restrict__ A, 
                       const float* __restrict__ B, 
                       float* __restrict__ C, 
                       int M, int N, int K)
    {
        const int bRow = blockIdx.y;
        const int bCol = blockIdx.x;

        const int numThreadsX = BN / TN; 
        const int tRow = threadIdx.x / numThreadsX;  
        const int tCol = threadIdx.x % numThreadsX;  

        __shared__ float As[BM][BK + 1];      
        __shared__ float Bs[BK][BN + 1];      

        float acc[TM][TN];
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                acc[i][j] = 0.0f;
            }
        }
        
        float fragA[TM];
        float fragB[TN];

        const int aRowBase = bRow * BM;
        const int bColBase = bCol * BN;

        const int loadRowA = threadIdx.x / BK;        
        const int loadColA = threadIdx.x % BK;        
        const int strideA = 256 / BK;                 
        
        const int loadRowB = threadIdx.x / BN;        
        const int loadColB = threadIdx.x % BN;        
        const int strideB = 256 / BN;                 

        for (int kTile = 0; kTile < K; kTile += BK) {

            #pragma unroll
            for (int i = 0; i < BM; i += strideA) {
                int aRow = aRowBase + loadRowA + i;
                int aCol = kTile + loadColA;
                float val = 0.0f;
                if (aRow < M && aCol < K) {
                    val = A[aRow * K + aCol];
                }
                As[loadRowA + i][loadColA] = val;
            }

            #pragma unroll
            for (int i = 0; i < BK; i += strideB) {
                int bRowIdx = kTile + loadRowB + i;
                int bCol = bColBase + loadColB;
                float val = 0.0f;
                if (bRowIdx < K && bCol < N) {
                    val = B[bRowIdx * N + bCol];
                }
                Bs[loadRowB + i][loadColB] = val;
            }

            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    fragA[i] = As[tRow * TM + i][kk];
                }

                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    fragB[j] = Bs[kk][tCol * TN + j];
                }

                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        acc[i][j] = fmaf(fragA[i], fragB[j], acc[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            int globalRow = aRowBase + tRow * TM + i;
            if (globalRow < M) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    int globalCol = bColBase + tCol * TN + j;
                    if (globalCol < N) {
                        C[globalRow * N + globalCol] += acc[i][j];
                    }
                }
            }
        }
    }


    __global__ void MM_Bias_Tanh(const float* __restrict__ A, 
                                 const float* __restrict__ B,
                                 const float* __restrict__ bias, 
                                 float* __restrict__ C, 
                                 int M, int N, int K)
    {
        const int bRow = blockIdx.y;
        const int bCol = blockIdx.x;

        const int numThreadsX = BN / TN;
        const int tRow = threadIdx.x / numThreadsX;
        const int tCol = threadIdx.x % numThreadsX;

        __shared__ float As[BM][BK + 1];
        __shared__ float Bs[BK][BN + 1];

        float acc[TM][TN];
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                acc[i][j] = 0.0f;
            }
        }
        
        float fragA[TM];
        float fragB[TN];

        const int aRowBase = bRow * BM;
        const int bColBase = bCol * BN;

        const int loadRowA = threadIdx.x / BK;
        const int loadColA = threadIdx.x % BK;
        const int strideA = 256 / BK;
        
        const int loadRowB = threadIdx.x / BN;
        const int loadColB = threadIdx.x % BN;
        const int strideB = 256 / BN;

        for (int kTile = 0; kTile < K; kTile += BK) {
            #pragma unroll
            for (int i = 0; i < BM; i += strideA) {
                int aRow = aRowBase + loadRowA + i;
                int aCol = kTile + loadColA;
                float val = 0.0f;
                if (aRow < M && aCol < K) {
                    val = A[aRow * K + aCol];
                }
                As[loadRowA + i][loadColA] = val;
            }

            #pragma unroll
            for (int i = 0; i < BK; i += strideB) {
                int bRowIdx = kTile + loadRowB + i;
                int bCol = bColBase + loadColB;
                float val = 0.0f;
                if (bRowIdx < K && bCol < N) {
                    val = B[bRowIdx * N + bCol];
                }
                Bs[loadRowB + i][loadColB] = val;
            }

            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    fragA[i] = As[tRow * TM + i][kk];
                }
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    fragB[j] = Bs[kk][tCol * TN + j];
                }
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        acc[i][j] = fmaf(fragA[i], fragB[j], acc[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            int globalRow = aRowBase + tRow * TM + i;
            if (globalRow < M) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    int globalCol = bColBase + tCol * TN + j;
                    if (globalCol < N) {
                        float val = acc[i][j] + bias[globalCol];
                        C[globalRow * N + globalCol] = tanhf(val);
                    }
                }
            }
        }
    }

    __global__ void MM_SmallN(const float* __restrict__ A, 
                              const float* __restrict__ B, 
                              float* __restrict__ C, 
                              int m, int n, int k)
    {
        const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        const int lane = threadIdx.x % 32;
        const int row = warpId;
        
        if (row >= m) return;
        
        float acc[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
        
        const float* Arow = A + row * k;
        
        for (int kk = lane; kk < k; kk += 32) {
            float aVal = Arow[kk];
            const float* Brow = B + kk * n;
            #pragma unroll
            for (int j = 0; j < n && j < 16; ++j) {
                acc[j] = fmaf(aVal, Brow[j], acc[j]);
            }
        }
        
        #pragma unroll
        for (int j = 0; j < n && j < 16; ++j) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[j] += __shfl_down_sync(0xFFFFFFFF, acc[j], offset);
            }
            if (lane == 0) {
                C[row * n + j] += acc[j];
            }
        }
    }

    __global__ void MM_SmallN_Bias_Sigmoid(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            const float* __restrict__ bias,
                                            float* __restrict__ C, 
                                            int m, int n, int k)
    {
        const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        const int lane = threadIdx.x % 32;
        const int row = warpId;
        
        if (row >= m) return;
        
        float acc[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
        
        const float* Arow = A + row * k;
        
        for (int kk = lane; kk < k; kk += 32) {
            float aVal = Arow[kk];
            const float* Brow = B + kk * n;
            #pragma unroll
            for (int j = 0; j < n && j < 16; ++j) {
                acc[j] = fmaf(aVal, Brow[j], acc[j]);
            }
        }
        
        #pragma unroll
        for (int j = 0; j < n && j < 16; ++j) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[j] += __shfl_down_sync(0xFFFFFFFF, acc[j], offset);
            }
            if (lane == 0) {
                float val = acc[j] + bias[j];
                C[row * n + j] = 1.0f / (1.0f + expf(-val));
            }
        }
    }

    __global__ void MV(const float* __restrict__ A, 
                       const float* __restrict__ x, 
                       float* __restrict__ y, 
                       int m, int n)
    {
        extern __shared__ float xs[];
        
        const int warpId = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        const int numWarps = blockDim.x / 32;
        const int row = blockIdx.x * numWarps + warpId;
        
        if (row >= m) return;
        
        float sum = 0.0f;
        const float* Arow = A + row * n;
        
        const int chunkSize = blockDim.x;  
        
        for (int chunk = 0; chunk < n; chunk += chunkSize) {
            int xIdx = chunk + threadIdx.x;
            if (xIdx < n) {
                xs[threadIdx.x] = x[xIdx];
            }
            __syncthreads();

            int chunkEnd = min(chunk + chunkSize, n);
            int chunkLen = chunkEnd - chunk;
            
    
            for (int j = lane; j < chunkLen; j += 32) {
                int globalJ = chunk + j;
                if (globalJ < n) {
                    sum += Arow[globalJ] * xs[j];
                }
            }
            __syncthreads();
        }
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        
        if (lane == 0) {
            y[row] += sum;
        }
    }

    __global__ void SpMM(const int* __restrict__ row_ptr, 
                         const int* __restrict__ col_idx, 
                         const float* __restrict__ values, 
                         const float* __restrict__ B, 
                         float* __restrict__ C, 
                         int m, int n, int k)
    {
        int row = blockIdx.x;
        int tid = threadIdx.x;
        int numThreads = blockDim.x;

        if (row >= m) return;

        int rowStart = row_ptr[row];
        int rowEnd = row_ptr[row + 1];

        // Each thread handles multiple columns of C
        for (int c = tid; c < n; c += numThreads) {
            float sum = 0.0f;
            
            // Iterate over non-zeros in this row of A
            for (int idx = rowStart; idx < rowEnd; ++idx) {
                int aCol = col_idx[idx];
                float aVal = values[idx];
                sum += aVal * B[aCol * n + c];
            }
            
            C[row * n + c] += sum;
        }
    }

    __global__ void SpMV(const int* __restrict__ row_ptr, 
                         const int* __restrict__ col_idx, 
                         const float* __restrict__ values, 
                         const float* __restrict__ x, 
                         float* __restrict__ y, 
                         int m, int n)
    {
        int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        int lane = threadIdx.x % 32;

        if (warpId >= m) return;

        int row = warpId;
        float sum = 0.0f;
        
        int rowStart = row_ptr[row];
        int rowEnd = row_ptr[row + 1];
        
        // Each lane processes elements with stride of 32
        for (int idx = rowStart + lane; idx < rowEnd; idx += 32) {
            int col = col_idx[idx];
            float val = values[idx];
            sum += val * x[col];
        }
        
        // Warp-level reduction using shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        
        if (lane == 0) {
            y[row] += sum;
        }
    }

    __global__ void MV_Batched(const float* __restrict__ W,   
                               const float* __restrict__ X,  
                               float* __restrict__ Y,        
                               int batchSize, int m, int n)
    {
        int row = blockIdx.x;       
        int sample = blockIdx.y;    
        int tid = threadIdx.x;
        int numThreads = blockDim.x;
        
        if (row >= m || sample >= batchSize) return;
        
        extern __shared__ float sdata[];
        
        const float* Wrow = W + row * n;
        const float* Xsample = X + sample * n;
        
        float sum = 0.0f;
        
        int j = tid * 4;
        int n4 = (n / 4) * 4;
        
        while (j < n4) {
            float4 w = *reinterpret_cast<const float4*>(Wrow + j);
            float4 x = *reinterpret_cast<const float4*>(Xsample + j);
            sum += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
            j += numThreads * 4;
        }
        
        j = n4 + tid;
        while (j < n) {
            sum += Wrow[j] * Xsample[j];
            j += numThreads;
        }
        
        sdata[tid] = sum;
        __syncthreads();
        
        for (int stride = numThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            Y[sample * m + row] += sdata[0];
        }
    }

    __global__ void CopyBiasBatched(float* __restrict__ Y, 
                                    const float* __restrict__ bias, 
                                    int batchSize, int m)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batchSize * m;
        
        if (idx < total) {
            int col = idx % m;
            Y[idx] = bias[col];
        }
    }


    __global__ void TanhActivation(float* data, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            data[idx] = tanhf(data[idx]);
        }
    }

    __global__ void SigmoidActivation(float* data, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            data[idx] = 1.0f / (1.0f + expf(-data[idx]));
        }
    }

    __global__ void BroadcastBias(float* C, const float* bias, int m, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = m * n;
        
        if (idx < total) {
            int col = idx % n;
            C[idx] = bias[col];
        }
    }

    __global__ void CopyBias(float* y, const float* bias, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            y[idx] = bias[idx];
        }
    }

    __global__ void Transpose(const float* __restrict__ A, 
                              float* __restrict__ B, 
                              int m, int n)
    {
        __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

        int x = blockIdx.x * TILE_SIZE + threadIdx.x;
        int y = blockIdx.y * TILE_SIZE + threadIdx.y;

        if (x < n && y < m) {
            tile[threadIdx.y][threadIdx.x] = A[y * n + x];
        }

        __syncthreads();

        int tx = blockIdx.y * TILE_SIZE + threadIdx.x;
        int ty = blockIdx.x * TILE_SIZE + threadIdx.y;

        if (tx < m && ty < n) {
            B[ty * m + tx] = tile[threadIdx.x][threadIdx.y];
        }
    }

}  // namespace swiftware::hpp

#endif //PROJECT_KERNELS_CUH
