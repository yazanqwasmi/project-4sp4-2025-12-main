#include "gemm.h"
#include <gtest/gtest.h>
#include <cmath>

namespace swiftware::hpp {

// Test basic gemm with identity B (already present)
TEST(GemmTest, BasicTest) {
    int m = 4, n = 4, k = 4;
    float A[16] = {1,2,3,4,
                   5,6,7,8,
                   9,10,11,12,
                   13,14,15,16};
    float B[16] = {1,0,0,0,
                   0,1,0,0,
                   0,0,1,0,
                   0,0,0,1};
    float C[16] = {0};

    ScheduleParams sp(2, 2);
    gemm(m, n, k, A, B, C, sp);

    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(C[i], A[i], 1e-5);
    }
}

// Non-square matrices: A(2x3), B(3x2)
TEST(GemmTest, NonSquareMatrices) {
    int m = 2, k = 3, n = 2;
    // A = [[1,2,3],
    //      [4,5,6]]
    float A[6] = {1,2,3,
                  4,5,6};
    // B = [[1,4],
    //      [2,5],
    //      [3,6]] = A^T
    float B[6] = {1,4,
                  2,5,
                  3,6};
    float C[4] = {0,0,
                  0,0};

    ScheduleParams sp(2, 2);
    gemm(m, n, k, A, B, C, sp);

    // C = A * B = [[14, 32],
    //              [32, 77]]
    EXPECT_NEAR(C[0], 14.0f, 1e-5);
    EXPECT_NEAR(C[1], 32.0f, 1e-5);
    EXPECT_NEAR(C[2], 32.0f, 1e-5);
    EXPECT_NEAR(C[3], 77.0f, 1e-5);
}

// Check accumulation: C = A*B + C_initial
TEST(GemmTest, Accumulation) {
    int m = 2, k = 2, n = 2;
    float A[4] = {1,2,
                  3,4};
    float B[4] = {5,6,
                  7,8};
    // A*B = [[19,22],
    //        [43,50]]
    float C[4] = {1,1,
                  1,1};

    ScheduleParams sp(-1, -1);
    gemm(m, n, k, A, B, C, sp);

    EXPECT_NEAR(C[0], 20.0f, 1e-5); // 19 + 1
    EXPECT_NEAR(C[1], 23.0f, 1e-5); // 22 + 1
    EXPECT_NEAR(C[2], 44.0f, 1e-5); // 43 + 1
    EXPECT_NEAR(C[3], 51.0f, 1e-5); // 50 + 1
}

} // namespace swiftware::hpp
