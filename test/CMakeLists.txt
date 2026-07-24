# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Each suite links only the translation units it exercises, so a failure points
# at one kernel rather than the whole library.
function(hpp_add_test target_name)
    add_executable(${target_name} ${ARGN})
    target_compile_options(${target_name} PRIVATE ${HPP_ARCH_FLAGS})
    target_link_libraries(${target_name} PRIVATE GTest::gtest_main)
    if(OPENMP)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    gtest_discover_tests(${target_name})
endfunction()

include(GoogleTest)

hpp_add_test(project_gemm_test
        gemm_test.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
)

hpp_add_test(project_gemv_test
        gemv_test.cpp
        ${CMAKE_SOURCE_DIR}/src/gemv.cpp
)

hpp_add_test(project_spmv_test
        spmv_test.cpp
        ${CMAKE_SOURCE_DIR}/src/spmv.cpp
)

hpp_add_test(project_spmm_test
        spmm_test.cpp
        ${CMAKE_SOURCE_DIR}/src/spmm.cpp
)

hpp_add_test(project_dense_nn_test
        denseNN_test.cpp
        ${CMAKE_SOURCE_DIR}/src/dense_nn.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
        ${CMAKE_SOURCE_DIR}/src/gemv.cpp
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

hpp_add_test(project_sparse_nn_test
        sparseNN_test.cpp
        ${CMAKE_SOURCE_DIR}/src/sparse_nn.cpp
        ${CMAKE_SOURCE_DIR}/src/spmm.cpp
        ${CMAKE_SOURCE_DIR}/src/spmv.cpp
        ${CMAKE_SOURCE_DIR}/src/dense_nn.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
        ${CMAKE_SOURCE_DIR}/src/gemv.cpp
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
)
