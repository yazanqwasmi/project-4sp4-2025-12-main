#include "gemv.h"
#include <gtest/gtest.h>
#include <cmath>

namespace swiftware::hpp {

// Test basic gemv
TEST(GemvTest, BasicTest) {
    int m = 4, n = 4;
    float A[16] = {1,0,0,0,
                   0,1,0,0,
                   0,0,1,0,
                   0,0,0,1};
    float x[4] = {1,2,3,4};
    float y[4] = {0,0,0,0};

    ScheduleParams sp(2, 2);
    gemv(m, n, A, x, y, sp);

    EXPECT_NEAR(y[0], 1.0f, 1e-5);
    EXPECT_NEAR(y[1], 2.0f, 1e-5);
    EXPECT_NEAR(y[2], 3.0f, 1e-5);
    EXPECT_NEAR(y[3], 4.0f, 1e-5);
}

// Non-square A: 2 x 3
TEST(GemvTest, NonSquareMatrix) {
    int m = 2, n = 3;
    // A = [[1,2,3],
    //      [4,5,6]]
    float A[6] = {1,2,3,
                  4,5,6};
    float x[3] = {1,1,1};
    float y[2] = {0,0};

    ScheduleParams sp(2, 2);
    gemv(m, n, A, x, y, sp);

    // y = [1+2+3, 4+5+6] = [6, 15]
    EXPECT_NEAR(y[0], 6.0f, 1e-5);
    EXPECT_NEAR(y[1], 15.0f, 1e-5);
}

// Check that gemv accumulates into y (y = A x + y_initial)
TEST(GemvTest, Accumulation) {
    int m = 2, n = 2;
    float A[4] = {1,2,
                  3,4};
    float x[2] = {1,1};
    float y[2] = {10, 20};  // initial values

    ScheduleParams sp(-1, -1); // tile sizes unused case
    gemv(m, n, A, x, y, sp);

    // A x = [3, 7]; y_new = [13, 27]
    EXPECT_NEAR(y[0], 13.0f, 1e-5);
    EXPECT_NEAR(y[1], 27.0f, 1e-5);
}

} // namespace swiftware::hpp
