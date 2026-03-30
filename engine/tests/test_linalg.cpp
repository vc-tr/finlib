#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/math/linalg.hpp"
#include <cmath>

using namespace qe;
using Catch::Matchers::WithinAbs;

TEST_CASE("Cholesky decomposition 2x2", "[linalg]") {
    // A = [[4, 2], [2, 3]] -> L = [[2, 0], [1, sqrt(2)]]
    Mat A = {{4.0, 2.0}, {2.0, 3.0}};
    Mat L = cholesky(A);

    REQUIRE_THAT(L[0][0], WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(L[0][1], WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(L[1][0], WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(L[1][1], WithinAbs(std::sqrt(2.0), 1e-12));
}

TEST_CASE("Cholesky: L * L^T = A", "[linalg]") {
    Mat A = {{25.0, 15.0, -5.0},
             {15.0, 18.0,  0.0},
             {-5.0,  0.0, 11.0}};

    Mat L = cholesky(A);
    Mat LT = transpose(L);
    Mat reconstructed = mat_mat_mul(L, LT);

    for (Size i = 0; i < 3; ++i) {
        for (Size j = 0; j < 3; ++j) {
            REQUIRE_THAT(reconstructed[i][j], WithinAbs(A[i][j], 1e-10));
        }
    }
}

TEST_CASE("Cholesky throws for non-positive-definite", "[linalg]") {
    Mat A = {{1.0, 2.0}, {2.0, 1.0}};  // not positive definite
    REQUIRE_THROWS(cholesky(A));
}

TEST_CASE("Thomas algorithm solves tridiagonal system", "[linalg]") {
    // System: [2 1 0] [x0]   [1]
    //         [1 3 1] [x1] = [2]
    //         [0 1 2] [x2]   [3]
    Vec a = {1.0, 1.0};        // sub-diagonal
    Vec b = {2.0, 3.0, 2.0};   // main diagonal
    Vec c = {1.0, 1.0};        // super-diagonal
    Vec d = {1.0, 2.0, 3.0};   // rhs

    Vec x = thomas_solve(a, b, c, d);
    REQUIRE(x.size() == 3);

    // Verify A * x = d
    REQUIRE_THAT(b[0] * x[0] + c[0] * x[1], WithinAbs(d[0], 1e-10));
    REQUIRE_THAT(a[0] * x[0] + b[1] * x[1] + c[1] * x[2], WithinAbs(d[1], 1e-10));
    REQUIRE_THAT(a[1] * x[1] + b[2] * x[2], WithinAbs(d[2], 1e-10));
}

TEST_CASE("Thomas algorithm larger system", "[linalg]") {
    // 5x5 tridiagonal: typical from finite difference discretization
    Size n = 5;
    Vec a(n - 1, -1.0);
    Vec b(n, 2.0);
    Vec c(n - 1, -1.0);
    Vec d = {1.0, 0.0, 0.0, 0.0, 1.0};

    Vec x = thomas_solve(a, b, c, d);
    REQUIRE(x.size() == n);

    // Verify solution
    REQUIRE_THAT(b[0] * x[0] + c[0] * x[1], WithinAbs(d[0], 1e-10));
    for (Size i = 1; i < n - 1; ++i) {
        Real val = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1];
        REQUIRE_THAT(val, WithinAbs(d[i], 1e-10));
    }
    REQUIRE_THAT(a[n - 2] * x[n - 2] + b[n - 1] * x[n - 1], WithinAbs(d[n - 1], 1e-10));
}

TEST_CASE("Matrix-vector multiplication", "[linalg]") {
    Mat A = {{1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0}};
    Vec x = {1.0, 1.0, 1.0};

    Vec y = mat_vec_mul(A, x);
    REQUIRE(y.size() == 2);
    REQUIRE_THAT(y[0], WithinAbs(6.0, 1e-12));
    REQUIRE_THAT(y[1], WithinAbs(15.0, 1e-12));
}

TEST_CASE("Matrix-matrix multiplication", "[linalg]") {
    Mat A = {{1.0, 2.0}, {3.0, 4.0}};
    Mat B = {{5.0, 6.0}, {7.0, 8.0}};

    Mat C = mat_mat_mul(A, B);
    REQUIRE_THAT(C[0][0], WithinAbs(19.0, 1e-12));
    REQUIRE_THAT(C[0][1], WithinAbs(22.0, 1e-12));
    REQUIRE_THAT(C[1][0], WithinAbs(43.0, 1e-12));
    REQUIRE_THAT(C[1][1], WithinAbs(50.0, 1e-12));
}

TEST_CASE("Identity matrix", "[linalg]") {
    Mat I = identity(3);
    REQUIRE(I.size() == 3);
    for (Size i = 0; i < 3; ++i) {
        for (Size j = 0; j < 3; ++j) {
            REQUIRE_THAT(I[i][j], WithinAbs(i == j ? 1.0 : 0.0, 1e-12));
        }
    }

    // A * I = A
    Mat A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    Mat AI = mat_mat_mul(A, I);
    for (Size i = 0; i < 3; ++i) {
        for (Size j = 0; j < 3; ++j) {
            REQUIRE_THAT(AI[i][j], WithinAbs(A[i][j], 1e-12));
        }
    }
}

TEST_CASE("Transpose", "[linalg]") {
    Mat A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    Mat AT = transpose(A);

    REQUIRE(AT.size() == 3);
    REQUIRE(AT[0].size() == 2);
    REQUIRE_THAT(AT[0][0], WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(AT[0][1], WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(AT[2][1], WithinAbs(6.0, 1e-12));
}
