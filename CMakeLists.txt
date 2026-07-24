CMAKE_MINIMUM_REQUIRED(VERSION 3.26)

list(APPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_LIST_DIR}/cmake
        "${CMAKE_CURRENT_LIST_DIR}/cmake/third_party"
        )
include(DownloadProject)

get_directory_property(HAS_PARENT PARENT_DIRECTORY)

# set top level variable
#set(GOOGLE_BENCHMARK_REQUIRED OFF CACHE BOOL "GBench is needed")
set(PROFILING_WITH_PAPI OFF CACHE BOOL "PAPI is needed")
set(PROFILING_WITH_LIKWID OFF CACHE BOOL "LIKWID is needed")

project(project C CXX)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# set the c++ standard to c++11
set( CMAKE_CXX_STANDARD 11 )

## Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# fetch google benchmark
option(PROFILING_ENABLED "Enables Modules for Profiling, needs LIBFPM4" OFF)
option(OPENMP "Enables OpenMP" ON)

# Find OpenMP
if(OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# Find Intel MKL
option(USE_MKL "Use Intel MKL for reference implementations" ON)
if(USE_MKL)
    find_package(MKL CONFIG REQUIRED)
    message(STATUS "MKL found: ${MKL_ROOT}")
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx")
include(FetchContent)


FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4
)
# enable BENCHMARK_DOWNLOAD_DEPENDENCIES
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
if (PROFILING_ENABLED)
    # enable PFM
    set(BENCHMARK_ENABLE_LIBPFM ON)
endif()

FetchContent_MakeAvailable(googlebenchmark)

# enable google test
set(BENCHMARK_ENABLE_TESTS ON)
set(BENCHMARK_ENABLE_GTEST_TESTS ON)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(project
        src/utils.cpp
        src/dense_nn.cpp
        src/sparse_nn.cpp
        src/gemm.cpp
        src/gemv.cpp
        src/spmm.cpp
        src/spmv.cpp
        main_bench.cpp
)
target_compile_options(project PRIVATE -mavx2 -mfma -march=native)

target_link_libraries(project
        benchmark::benchmark
)

add_executable(nn_cpu
        src/utils.cpp
        src/dense_nn.cpp
        src/sparse_nn.cpp
        src/gemm.cpp
        src/gemv.cpp
        src/spmm.cpp
        src/spmv.cpp
        nn_cpu_bench.cpp
)
target_compile_options(nn_cpu PRIVATE -mavx2 -mfma -march=native)
target_link_libraries(nn_cpu
        benchmark::benchmark)

if(OPENMP)
    target_link_libraries(project OpenMP::OpenMP_CXX)
    target_link_libraries(nn_cpu OpenMP::OpenMP_CXX)
endif()

if(USE_MKL)
    target_link_libraries(project MKL::MKL)
    target_compile_definitions(project PRIVATE USE_MKL)
    target_link_libraries(nn_cpu MKL::MKL)
    target_compile_definitions(nn_cpu PRIVATE USE_MKL)
endif()

# GPU support
if(GPU_ENABLED)
    # Ensure CMAKE_CUDA_ARCHITECTURES is set (cannot be empty)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for")
    endif()

    # Enable CUDA language before setting CUDA-specific properties
    enable_language(CUDA OPTIONAL)
    # Set CUDA standard
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)

    # Use CPM to fetch NVBench's main branch.
    # CPM: https://github.com/cpm-cmake/CPM.cmake
    # NVBench: https://github.com/NVIDIA/nvbench
    include(cmake/CPM.cmake)
    CPMAddPackage("gh:NVIDIA/nvbench#main")

    add_executable(project_gpu
            main_bench.cu)

    target_link_libraries(project_gpu
            nvbench::main
            ${CUDA_LIBRARIES}
            CUDA::cublas)

    add_executable(nn_gpu
            src/utils.cpp
            src/gpu_dense_nn.cu
            src/gpu_sparse_nn.cu
            nn_gpu_bench.cu
    )
    target_link_libraries(nn_gpu
            nvbench::main
            ${CUDA_LIBRARIES}
            CUDA::cublas
            CUDA::cusparse)

endif()

add_subdirectory(test)