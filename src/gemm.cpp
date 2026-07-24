// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gemm.h"
#include <immintrin.h>
#include <iostream>
#include <omp.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace swiftware::hpp {

    void gemm(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp) {
       
       // Block sizes from scheduling parameters
        const int BM = (Sp.TileSize1 > 0) ? Sp.TileSize1 : 64;
        const int BN = (Sp.TileSize2 > 0) ? Sp.TileSize2 : 64;
        const int BK = 64;  // can also be tuned

        // Parallelize over tiles of (ii, jj)
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ii = 0; ii < m; ii += BM) {
            for (int jj = 0; jj < n; jj += BN) {
                int iMax = (ii + BM < m) ? (ii + BM) : m;
                int jMax = (jj + BN < n) ? (jj + BN) : n;

                for (int kk = 0; kk < k; kk += BK) {
                    int kMax = (kk + BK < k) ? (kk + BK) : k;

                    for (int i = ii; i < iMax; ++i) {
                        const float* Ai = A + i * k;
                        float* Ci = C + i * n;

                        for (int p = kk; p < kMax; ++p) {
                            float a_ip = Ai[p];

                            const float* Bp = B + p * n;
                            // Inner loop over columns of C; good for SIMD
                            #pragma omp simd
                            for (int j = jj; j < jMax; ++j) {
                                Ci[j] += a_ip * Bp[j];
                            }
                        }
                    }
                }
            }
        }
    }



#ifdef USE_MKL
    void gemmMKL(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp) {

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k,
                    1.0f,        // alpha
                    A, k,        // A is m x k
                    B, n,        // B is k x n
                    1.0f,        // beta (accumulate into C)
                    C, n);       // C is m x n
    }
#endif

}