# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# TODO: add tests for MM, MV, SPMV, SPMM, DenseNN, and SparseNN

add_executable( project_gemm_test
        gemm_test.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
        )
target_compile_options(project_gemm_test PRIVATE -mavx2 -mfma -march=native)
target_link_libraries( project_gemm_test
        PRIVATE
        GTest::gtest_main
        )
if(OPENMP)
    target_link_libraries(project_gemm_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable( project_gemv_test
        gemv_test.cpp
        ${CMAKE_SOURCE_DIR}/src/gemv.cpp
        )
target_compile_options(project_gemv_test PRIVATE -mavx2 -mfma -march=native)
target_link_libraries( project_gemv_test
        PRIVATE
        GTest::gtest_main
        )
if(OPENMP)
    target_link_libraries(project_gemv_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable( project_dense_nn_test
        denseNN_test.cpp
        ${CMAKE_SOURCE_DIR}/src/dense_nn.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
        ${CMAKE_SOURCE_DIR}/src/gemv.cpp
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
        )
target_compile_options(project_dense_nn_test PRIVATE -mavx2 -mfma -march=native)
target_link_libraries( project_dense_nn_test
        PRIVATE
        GTest::gtest_main
        )
if(OPENMP)
    target_link_libraries(project_dense_nn_test PRIVATE OpenMP::OpenMP_CXX)
endif()

include(GoogleTest)
gtest_discover_tests(project_gemm_test)
gtest_discover_tests(project_gemv_test)
gtest_discover_tests(project_dense_nn_test)