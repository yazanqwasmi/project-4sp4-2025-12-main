// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef PROJECT_GPU_UTILS_H
#define PROJECT_GPU_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace swiftware::hpp {

    inline void cuda_check(cudaError_t e, const char* file, int line) {
        if (e != cudaSuccess) {
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
            std::abort();
        }
    }

    #define CUDA_CHECK(x) swiftware::hpp::cuda_check((x), __FILE__, __LINE__)

}  // namespace swiftware::hpp

#endif //PROJECT_GPU_UTILS_H
